"""
Op 0b: LLM Field Summarizer

Uses the configured LLM (default: Haiku/model_fast tier) to generate
natural-language summaries for each database column based on its statistical
profile. Results are cached to disk as JSON.

Batching: columns are grouped by table; within each table, batches of 5-8
columns are sent per API call to keep prompts manageable.

Structured output is obtained exclusively via tool-use (JSON schema) — no
free-text parsing.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from src.config.settings import settings
from src.llm import get_client, ToolParam, CacheableText, LLMResponse
from src.preprocessing.profiler import ColumnProfile, DatabaseProfile


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FieldSummary:
    table_name: str
    column_name: str
    short_summary: str   # 1-2 sentences, max 200 chars
    long_summary: str    # 3-5 sentences, max 1000 chars


@dataclass
class DatabaseSummary:
    db_id: str
    field_summaries: list  # list[FieldSummary]


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

_SUMMARIZE_FIELDS_TOOL = ToolParam(
    name="summarize_fields",
    description="Provide natural language summaries for database fields",
    input_schema={
        "type": "object",
        "properties": {
            "summaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column_name": {"type": "string"},
                        "short_summary": {
                            "type": "string",
                            "description": "1-2 sentences, max 200 chars",
                        },
                        "long_summary": {
                            "type": "string",
                            "description": "3-5 sentences, max 1000 chars",
                        },
                    },
                    "required": ["column_name", "short_summary", "long_summary"],
                },
            }
        },
        "required": ["summaries"],
    },
)

# Batch size: send 5-8 columns per API call
_BATCH_SIZE = 6


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_system_prompt(db_id: str, table_name: str, all_column_names: list[str]) -> str:
    cols_str = ", ".join(all_column_names)
    return (
        "You are a database documentation expert. Summarize the following database "
        "fields clearly and precisely for use by SQL query generators.\n\n"
        f"Database: {db_id}\n"
        f"Table: {table_name}\n"
        f"All table columns: {cols_str}"
    )


def _format_sample_values(sample_values: list) -> str:
    """Format top-5 sample values as comma-separated string."""
    top5 = sample_values[:5]
    return ", ".join(str(v) for v, _ in top5) if top5 else "(none)"


def _build_user_prompt(table_name: str, batch: list[ColumnProfile]) -> str:
    n = len(batch)
    lines = [f"Summarize these {n} fields from table '{table_name}':\n"]
    for col in batch:
        lines.append(f"Column: {col.column_name}")
        lines.append(f"Type: {col.data_type}")
        lines.append(f"Null rate: {col.null_rate:.1%}")
        lines.append(f"Distinct values: {col.distinct_count}")
        lines.append(f"Sample values: {_format_sample_values(col.sample_values)}")
        if col.data_type in ("INTEGER", "REAL", "NUMERIC"):
            lines.append(
                f"Min: {col.min_value}, Max: {col.max_value}, "
                f"Avg: {col.avg_value:.2f}" if col.avg_value is not None
                else f"Min: {col.min_value}, Max: {col.max_value}"
            )
        elif col.data_type == "TEXT" and col.avg_length is not None:
            lines.append(f"Avg length: {col.avg_length:.1f} chars")
        lines.append("")  # blank separator between columns
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _enforce_length_limits(short: str, long_: str) -> tuple[str, str]:
    return short[:200], long_[:1000]


def _default_summary(table_name: str, column_name: str) -> tuple[str, str]:
    msg = f"The {column_name} field in the {table_name} table."
    return msg, msg


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

async def summarize_database(
    profile: DatabaseProfile,
    output_dir: Optional[str] = None,
) -> DatabaseSummary:
    """
    Generate FieldSummary objects for every column in a DatabaseProfile.

    Parameters
    ----------
    profile:
        A DatabaseProfile produced by profile_database().
    output_dir:
        If provided, the result is loaded from / saved to
        ``{output_dir}/{db_id}.json``.

    Returns
    -------
    DatabaseSummary
    """
    db_id = profile.db_id

    # ------------------------------------------------------------------
    # Cache check (load)
    # ------------------------------------------------------------------
    cache_path: Optional[Path] = None
    if output_dir is not None:
        cache_path = Path(output_dir) / f"{db_id}.json"
        if cache_path.exists():
            return _load_summary_from_json(cache_path)

    # ------------------------------------------------------------------
    # Group columns by table
    # ------------------------------------------------------------------
    table_columns: dict[str, list[ColumnProfile]] = {}
    for col in profile.columns:
        table_columns.setdefault(col.table_name, []).append(col)

    client = get_client()
    all_summaries: list[FieldSummary] = []

    for table_name, columns in table_columns.items():
        all_col_names = [c.column_name for c in columns]
        system_prompt_text = _build_system_prompt(db_id, table_name, all_col_names)
        system = [CacheableText(text=system_prompt_text, cache=True)]

        # Split into batches of _BATCH_SIZE
        batches = [
            columns[i : i + _BATCH_SIZE]
            for i in range(0, len(columns), _BATCH_SIZE)
        ]

        # Build a lookup of summaries received so far for this table
        received: dict[str, tuple[str, str]] = {}  # col_name → (short, long)

        for batch in batches:
            user_prompt = _build_user_prompt(table_name, batch)
            response: LLMResponse = await client.generate(
                model=settings.model_fast,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
                tools=[_SUMMARIZE_FIELDS_TOOL],
                tool_choice_name="summarize_fields",
                max_tokens=2000,
            )
            result = response.tool_inputs[0] if response.tool_inputs else {"summaries": []}

            for item in result.get("summaries", []):
                col_name = item.get("column_name", "")
                short = item.get("short_summary", "")
                long_ = item.get("long_summary", "")
                short, long_ = _enforce_length_limits(short, long_)
                received[col_name] = (short, long_)

        # Build FieldSummary objects for every column in the table
        for col in columns:
            if col.column_name in received:
                short, long_ = received[col.column_name]
            else:
                short, long_ = _default_summary(table_name, col.column_name)
            all_summaries.append(
                FieldSummary(
                    table_name=table_name,
                    column_name=col.column_name,
                    short_summary=short,
                    long_summary=long_,
                )
            )

    db_summary = DatabaseSummary(db_id=db_id, field_summaries=all_summaries)

    # ------------------------------------------------------------------
    # Cache write
    # ------------------------------------------------------------------
    if cache_path is not None:
        _save_summary_to_json(db_summary, cache_path)

    return db_summary


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

def _save_summary_to_json(summary: DatabaseSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(summary)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_summary_from_json(path: Path) -> DatabaseSummary:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    field_summaries = [FieldSummary(**fs) for fs in data.get("field_summaries", [])]
    return DatabaseSummary(db_id=data["db_id"], field_summaries=field_summaries)
