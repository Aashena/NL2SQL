"""
Op 6: Adaptive Schema Linking

Takes a question + grounding context and produces two filtered schemas:
  S₁ (precise): Only columns DEFINITELY needed to answer the question.
  S₂ (recall): Columns that MIGHT be needed (wider net), always S₁ ⊆ S₂.

Two LLM iterations:
  1. Precise selection (S₁) — uses model_powerful with tool_use
  2. Recall expansion (S₂) — same model, wider net from remaining candidates

Primary keys and relevant foreign keys are auto-added for all referenced tables.
Both schemas are rendered as filtered DDL and Markdown using the full schema text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from src.llm import get_client, CacheableText, LLMError, ToolParam, sanitize_prompt_text
from src.config.settings import settings

if TYPE_CHECKING:
    from src.indexing.faiss_index import FAISSIndex, FieldMatch
    from src.grounding.context_grounder import GroundingContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definition for column selection (used in both iterations)
# ---------------------------------------------------------------------------

_SELECT_COLUMNS_TOOL = ToolParam(
    name="select_columns",
    description=(
        "Select database columns needed to answer the SQL question. "
        "For each selected field use the exact table name and column name as listed "
        "in the candidate fields (e.g. table='schools', column='Enrollment (K-12)'). "
        "The 'column' value must be the column name only — never include the table name "
        "again inside 'column' (e.g. use column='year', not column='seasons.year')."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "selected_fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string"},
                        "column": {"type": "string"},
                        "reason": {"type": "string"},
                        "role": {
                            "type": "string",
                            "enum": ["select", "where", "join", "group_by", "order_by"],
                            "description": "The SQL clause this column primarily serves",
                        },
                    },
                    "required": ["table", "column", "reason", "role"],
                },
            }
        },
        "required": ["selected_fields"],
    },
)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class LinkedSchemas:
    s1_ddl: str                          # Precise DDL schema (subset)
    s1_markdown: str                     # Precise Markdown schema (subset)
    s2_ddl: str                          # Recall DDL schema (subset)
    s2_markdown: str                     # Recall Markdown schema (subset)
    s1_fields: list[tuple[str, str]]     # (table, column) pairs in S₁
    s2_fields: list[tuple[str, str]]     # (table, column) pairs in S₂
    selection_reasoning: str             # Concatenated reasons from LLM


# ---------------------------------------------------------------------------
# DDL / Markdown parsing helpers
# ---------------------------------------------------------------------------

def _parse_ddl_tables(full_ddl: str) -> dict[str, dict]:
    """
    Parse full_ddl into a dict mapping table_name → {
        'header': str,       # '-- Table: X\nCREATE TABLE X ('
        'columns': dict[col_name, str],   # col_name → full column line (with comment)
        'footer': list[str], # lines after ');' (FK comments, example row)
        'pk_columns': set[str],           # columns with PRIMARY KEY
        'fk_refs': list[str],             # FK comment lines
    }
    """
    tables: dict[str, dict] = {}
    lines = full_ddl.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect table header comment
        if line.startswith("-- Table:"):
            table_name = line[len("-- Table:"):].strip()
            header_lines = [line]
            i += 1
            # Next non-blank line should be CREATE TABLE
            while i < len(lines) and not lines[i].strip().startswith("CREATE TABLE"):
                header_lines.append(lines[i])
                i += 1
            if i >= len(lines):
                break
            create_line = lines[i]
            header_lines.append(create_line)
            i += 1
            # Collect columns until ');\''
            col_dict: dict[str, str] = {}
            pk_cols: set[str] = set()
            body_lines: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith(");"):
                col_line = lines[i]
                body_lines.append(col_line)
                # Extract column name from line like:  name TYPE[PK][,]  [-- comment]
                stripped = col_line.strip()
                if stripped:
                    # Column name is first token
                    # Handle quoted names: "col name" or unquoted col_name
                    m = re.match(r'^\s*"([^"]+)"\s+|^\s+(\S+)\s+', col_line)
                    col_name = None
                    if col_line.strip().startswith('"'):
                        m2 = re.match(r'^\s*"([^"]+)"', col_line)
                        if m2:
                            col_name = m2.group(1)
                    else:
                        tokens = stripped.split()
                        if tokens:
                            col_name = tokens[0].rstrip(',')
                    if col_name:
                        col_dict[col_name] = col_line
                        if "PRIMARY KEY" in col_line:
                            pk_cols.add(col_name)
                i += 1
            # Consume ');'
            footer_lines: list[str] = []
            if i < len(lines) and lines[i].strip().startswith(");"):
                i += 1
            # Collect footer lines (FK comments, example row) until blank or next table
            while i < len(lines):
                fl = lines[i]
                if fl.strip() == "" or fl.startswith("-- Table:") or fl.startswith("CREATE TABLE"):
                    break
                footer_lines.append(fl)
                i += 1
            tables[table_name] = {
                "header": "\n".join(header_lines),
                "columns": col_dict,
                "footer": footer_lines,
                "pk_columns": pk_cols,
                "fk_refs": [fl for fl in footer_lines if fl.startswith("-- Foreign keys:")],
            }
        else:
            # Also detect CREATE TABLE without -- Table: comment
            if line.strip().startswith("CREATE TABLE"):
                m = re.match(r'CREATE TABLE\s+"?(\w+)"?\s*\(', line)
                if m:
                    table_name = m.group(1)
                    header_lines = [line]
                    i += 1
                    col_dict = {}
                    pk_cols: set[str] = set()
                    while i < len(lines) and not lines[i].strip().startswith(");"):
                        col_line = lines[i]
                        stripped = col_line.strip()
                        if stripped:
                            if col_line.strip().startswith('"'):
                                m2 = re.match(r'^\s*"([^"]+)"', col_line)
                                if m2:
                                    col_name = m2.group(1)
                                else:
                                    col_name = None
                            else:
                                tokens = stripped.split()
                                col_name = tokens[0].rstrip(',') if tokens else None
                            if col_name:
                                col_dict[col_name] = col_line
                                if "PRIMARY KEY" in col_line:
                                    pk_cols.add(col_name)
                        i += 1
                    footer_lines = []
                    if i < len(lines) and lines[i].strip().startswith(");"):
                        i += 1
                    while i < len(lines):
                        fl = lines[i]
                        if fl.strip() == "" or fl.startswith("-- Table:") or fl.startswith("CREATE TABLE"):
                            break
                        footer_lines.append(fl)
                        i += 1
                    tables[table_name] = {
                        "header": "\n".join(header_lines),
                        "columns": col_dict,
                        "footer": footer_lines,
                        "pk_columns": pk_cols,
                        "fk_refs": [fl for fl in footer_lines if fl.startswith("-- Foreign keys:")],
                    }
                else:
                    i += 1
            else:
                i += 1
    return tables


def _parse_markdown_tables(full_markdown: str) -> dict[str, dict]:
    """
    Parse full_markdown into a dict mapping table_name → {
        'header_lines': list[str],  # lines before the pipe table (## Table: X, *desc*, blank)
        'col_header': str,          # pipe table header line
        'col_separator': str,       # separator line
        'rows': dict[col_name, str],  # col_name → pipe table row
    }
    """
    tables: dict[str, dict] = {}
    # Split on ## Table: sections
    sections = re.split(r'(?=^## Table:)', full_markdown, flags=re.MULTILINE)
    for section in sections:
        if not section.strip():
            continue
        lines = section.splitlines()
        if not lines:
            continue
        header_line = lines[0]
        if not header_line.startswith("## Table:"):
            continue
        table_name = header_line[len("## Table:"):].strip()
        # Find where the pipe table starts
        col_header = None
        col_separator = None
        header_lines = [lines[0]]
        pipe_start = None
        for idx, ln in enumerate(lines[1:], start=1):
            if ln.strip().startswith("| Column"):
                col_header = ln
                if idx + 1 < len(lines) and lines[idx + 1].strip().startswith("|---"):
                    col_separator = lines[idx + 1]
                    pipe_start = idx + 2
                break
            else:
                header_lines.append(ln)

        rows: dict[str, str] = {}
        if pipe_start is not None:
            for ln in lines[pipe_start:]:
                if ln.strip().startswith("|") and not ln.strip().startswith("|---"):
                    # Extract column name (first cell)
                    cells = ln.strip().strip("|").split("|")
                    if cells:
                        col_name = cells[0].strip()
                        rows[col_name] = ln

        tables[table_name] = {
            "header_lines": header_lines,
            "col_header": col_header or "| Column | Type | Description | Sample Values |",
            "col_separator": col_separator or "|--------|------|-------------|---------------|",
            "rows": rows,
        }
    return tables


def _extract_pk_fk_from_ddl(
    full_ddl: str,
) -> tuple[dict[str, set[str]], dict[str, list[tuple[str, str]]]]:
    """
    Parse full_ddl to extract:
      - pk_map: table → set of PK column names
      - fk_map: table → list of (local_col, 'ref_table(ref_col)') tuples

    Returns (pk_map, fk_map).
    """
    pk_map: dict[str, set[str]] = {}
    fk_map: dict[str, list[tuple[str, str]]] = {}

    tables = _parse_ddl_tables(full_ddl)
    for table_name, info in tables.items():
        pk_map[table_name] = info["pk_columns"]
        # Parse FK comments: "-- Foreign keys: col1 REFERENCES t2(c2), col3 REFERENCES t4(c4)"
        for fk_line in info["fk_refs"]:
            prefix = "-- Foreign keys:"
            fk_content = fk_line[len(prefix):].strip()
            for part in fk_content.split(","):
                part = part.strip()
                m = re.match(r'(\S+)\s+REFERENCES\s+(\S+)', part)
                if m:
                    local_col = m.group(1)
                    ref = m.group(2)  # e.g. "departments(id)"
                    fk_map.setdefault(table_name, []).append((local_col, ref))

    return pk_map, fk_map


def _render_ddl_subset(
    ddl_tables: dict[str, dict],
    selected_by_table: dict[str, set[str]],
) -> str:
    """
    Rebuild a DDL string containing only the selected tables and columns.

    selected_by_table: table → set of selected column names
    """
    blocks: list[str] = []
    for table_name, selected_cols in selected_by_table.items():
        info = ddl_tables.get(table_name)
        if not info:
            continue
        # Rebuild CREATE TABLE block with only selected columns
        header = info["header"]
        # Ensure it ends with '('
        if not header.rstrip().endswith("("):
            # The header might already include the opening paren
            pass

        col_lines: list[str] = []
        for col_name in selected_cols:
            col_line = info["columns"].get(col_name)
            if col_line:
                col_lines.append(col_line)

        if not col_lines:
            continue

        # Fix trailing comma on last column line (same logic as schema_formatter)
        last = col_lines[-1]
        if ",  --" in last:
            last = last.replace(",  --", "  --", 1)
        elif last.rstrip().endswith(","):
            last = last.rstrip()[:-1]
        col_lines[-1] = last

        block_lines = [header]
        block_lines.extend(col_lines)
        block_lines.append(");")
        block_lines.extend(info["footer"])

        blocks.append("\n".join(block_lines))

    return "\n\n".join(blocks)


def _render_markdown_subset(
    md_tables: dict[str, dict],
    selected_by_table: dict[str, set[str]],
) -> str:
    """
    Rebuild a Markdown string containing only the selected tables and columns.
    """
    blocks: list[str] = []
    for table_name, selected_cols in selected_by_table.items():
        info = md_tables.get(table_name)
        if not info:
            continue
        lines = list(info["header_lines"])
        lines.append(info["col_header"])
        lines.append(info["col_separator"])
        for col_name in selected_cols:
            row = info["rows"].get(col_name)
            if row:
                lines.append(row)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# FAISS pre-filter + schema-hint augmentation
# ---------------------------------------------------------------------------

def _build_candidate_fields(
    question: str,
    grounding_context: "GroundingContext",
    faiss_index: "FAISSIndex",
    available_field_set: set[tuple[str, str]],
) -> list[tuple[str, str, str, str]]:
    """
    Run FAISS pre-filter, then augment with schema_hint fields.

    Returns list of (table, column, short_summary, long_summary) tuples,
    deduplicated and ordered by FAISS rank first.
    """
    faiss_results: list["FieldMatch"] = faiss_index.query(question, top_k=settings.faiss_top_k)

    seen: set[tuple[str, str]] = set()
    candidates: list[tuple[str, str, str, str]] = []

    for fm in faiss_results:
        key = (fm.table, fm.column)
        if key not in seen and key in available_field_set:
            seen.add(key)
            candidates.append((fm.table, fm.column, fm.short_summary, fm.long_summary))

    # Augment with schema_hints (fields mentioned in grounding context)
    hint_names = {h.lower() for h in grounding_context.schema_hints}
    for table, column, short_sum, long_sum in _build_all_available(available_field_set):
        key = (table, column)
        if key not in seen and column.lower() in hint_names:
            seen.add(key)
            candidates.append((table, column, short_sum, long_sum))

    return candidates


def _build_all_available(
    available_field_set: set[tuple[str, str]],
) -> list[tuple[str, str, str, str]]:
    """Helper placeholder — in practice, available_fields tuples carry summaries."""
    # This is called only for hint augmentation where we don't have summaries;
    # we'll return empty to avoid duplicating logic. The caller passes full
    # available_fields when needed.
    return []


# ---------------------------------------------------------------------------
# Main async function
# ---------------------------------------------------------------------------

async def link_schema(
    question: str,
    evidence: str,
    grounding_context: "GroundingContext",
    faiss_index: "FAISSIndex",
    full_ddl: str,
    full_markdown: str,
    available_fields: list[tuple[str, str, str, str]],
    # (table, column, short_summary, long_summary)
) -> LinkedSchemas:
    """
    Perform Op 6 adaptive schema linking.

    Parameters
    ----------
    question:
        Natural-language question.
    evidence:
        Auxiliary evidence string (may be empty or None).
    grounding_context:
        GroundingContext from Op 5 (matched_cells, schema_hints, few_shot_examples).
    faiss_index:
        Pre-built FAISSIndex for the target database.
    full_ddl:
        Full DDL schema text (output of schema_formatter).
    full_markdown:
        Full Markdown schema text (output of schema_formatter).
    available_fields:
        List of (table, column, short_summary, long_summary) tuples — all fields
        in the database.  Used for hallucination filtering and PK/FK detection.

    Returns
    -------
    LinkedSchemas with s1_fields, s2_fields, rendered DDL+Markdown for each.
    """
    evidence_str = evidence if evidence else "None"

    # Sanitize question and evidence before embedding in tool-use prompts.
    # Gemini's function-call parser fails on backtick-quoted identifiers,
    # control characters, and very long evidence blocks (100+ tokens).
    safe_question = sanitize_prompt_text(question)
    safe_evidence = sanitize_prompt_text(evidence_str)
    _MAX_EVIDENCE_CHARS = 500  # ~125 tokens; enough for all normal evidence strings
    if len(safe_evidence) > _MAX_EVIDENCE_CHARS:
        logger.debug(
            "Truncating evidence from %d to %d chars in schema linker prompt",
            len(safe_evidence), _MAX_EVIDENCE_CHARS,
        )
        safe_evidence = safe_evidence[:_MAX_EVIDENCE_CHARS] + "..."

    # Build lookup structures
    available_field_set: set[tuple[str, str]] = {
        (t, c) for t, c, _ss, _ls in available_fields
    }
    available_field_map: dict[tuple[str, str], tuple[str, str]] = {
        (t, c): (ss, ls) for t, c, ss, ls in available_fields
    }

    # ------------------------------------------------------------------
    # Step 6.1 — FAISS pre-filtering + schema-hint augmentation
    # ------------------------------------------------------------------
    # Catch OSError (includes BrokenPipeError) from SentenceTransformer model
    # load writing to a broken stdout/stderr pipe. Log as CRITICAL; fall back
    # to an empty candidate list so the S1/S2 LLM calls use full DDL context.
    try:
        faiss_results: list["FieldMatch"] = faiss_index.query(question, top_k=settings.faiss_top_k)
    except OSError as exc:
        logger.critical(
            "FAISS pre-filter query OS error (broken pipe?): %s — "
            "falling back to empty candidate list",
            exc,
        )
        faiss_results = []

    seen_keys: set[tuple[str, str]] = set()
    candidates: list[tuple[str, str, str, str]] = []  # (table, col, short, long)

    for fm in faiss_results:
        key = (fm.table, fm.column)
        if key not in seen_keys and key in available_field_set:
            seen_keys.add(key)
            candidates.append((fm.table, fm.column, fm.short_summary, fm.long_summary))

    # Augment with schema_hints
    hint_names = {h.lower() for h in grounding_context.schema_hints}
    if hint_names:
        for t, c, ss, ls in available_fields:
            key = (t, c)
            if key not in seen_keys and c.lower() in hint_names:
                seen_keys.add(key)
                candidates.append((t, c, ss, ls))

    # Format cell matches for user message
    cell_match_lines: list[str] = []
    for cm in grounding_context.matched_cells:
        cell_match_lines.append(
            f"  {cm.table}.{cm.column} = {cm.matched_value!r} (similarity={cm.similarity_score:.2f})"
        )
    formatted_cells = "\n".join(cell_match_lines) if cell_match_lines else "  (none)"

    # ------------------------------------------------------------------
    # Build shared system prompt blocks
    # ------------------------------------------------------------------
    system_block_1 = CacheableText(
        text=(
            "You are an expert database engineer selecting columns needed to answer "
            "a SQL question. Be precise and select only what is necessary."
        ),
        cache=False,
    )

    # Build candidate field list for system block 2 (cacheable).
    # Format uses pipe separators so special characters in column names
    # (parentheses, slashes, etc.) don't confuse Gemini's function-call parser.
    candidate_lines = [
        f"Table: {t} | Column: {c} | {ss}" for t, c, ss, _ls in candidates
    ]
    system_block_2 = CacheableText(
        text="Candidate fields:\n" + "\n".join(candidate_lines),
        cache=True,
    )

    # ------------------------------------------------------------------
    # Step 6.2 — Iteration 1: Precise Selection (S₁)
    # ------------------------------------------------------------------
    user_msg_s1 = (
        f"Question: {safe_question}\n"
        f"Evidence: {safe_evidence}\n"
        f"Matched values:\n{formatted_cells}\n\n"
        "Select only columns DEFINITELY needed to answer this question. "
        "Select at most 20 of the most relevant fields. "
        "Also include columns that will appear in the SELECT clause "
        "(output columns the question asks to return), not only columns "
        "needed for WHERE conditions and JOINs. "
        "For each column, set 'role' to the SQL clause it primarily serves "
        "(select, where, join, group_by, or order_by)."
    )

    client = get_client()
    try:
        response_s1 = await client.generate(
            model=settings.model_powerful_list,
            system=[system_block_1, system_block_2],
            messages=[{"role": "user", "content": user_msg_s1}],
            tools=[_SELECT_COLUMNS_TOOL],
            tool_choice_name="select_columns",
            max_tokens=8192,
            temperature=0.0,
        )
        if response_s1.tool_inputs:
            raw_s1 = response_s1.tool_inputs[0].get("selected_fields", [])
            if not isinstance(raw_s1, list):
                logger.warning(
                    "Schema linker S1: 'selected_fields' is not a list (%r); treating as empty",
                    type(raw_s1).__name__,
                )
                raw_s1 = []
            s1_raw: list[dict] = [f for f in raw_s1 if isinstance(f, dict)]
        else:
            logger.warning("Schema linker S1: no tool_inputs in response; using FAISS fallback")
            s1_raw = []
    except LLMError as exc:
        logger.error(
            "Schema linker S1 LLM call failed (%s); falling back to FAISS top-15 as S1.", exc
        )
        s1_raw = [
            {"table": t, "column": c, "reason": "FAISS-rank-fallback", "role": "where"}
            for t, c, _ss, _ls in candidates[:15]
        ]

    s1_reasons: list[str] = []
    s1_selected: set[tuple[str, str]] = set()

    for item in s1_raw:
        t = item.get("table", "")
        c = item.get("column", "")
        r = item.get("reason", "")
        role = item.get("role", "")
        key = (t, c)
        if key in available_field_set:
            s1_selected.add(key)
            role_tag = f"[{role}] " if role else ""
            s1_reasons.append(f"{role_tag}{t}.{c}: {r}")
        else:
            logger.warning(
                "Hallucinated field filtered out from S₁: %s.%s", t, c
            )

    # ------------------------------------------------------------------
    # Auto-add PKs and FKs for tables in S₁
    # ------------------------------------------------------------------
    pk_map, fk_map = _extract_pk_fk_from_ddl(full_ddl)

    def _add_pk_fk_for_tables(
        selected: set[tuple[str, str]],
        tables: set[str],
    ) -> set[tuple[str, str]]:
        """Add PK columns for all tables in `tables`. Add FK columns that bridge selected tables."""
        result = set(selected)
        for table in tables:
            # Add primary keys
            for pk_col in pk_map.get(table, set()):
                pk_key = (table, pk_col)
                if pk_key in available_field_set:
                    result.add(pk_key)
        # Add FK columns and their referenced target columns for tables in S1.
        # "FK closure": for every FK where both from_table and to_table are
        # already selected, add both the FK column (from_col) AND the target
        # column (to_col).  This handles non-PK join targets like
        # Player.player_api_id which PK auto-promotion alone would miss.
        selected_tables = {t for t, _c in result}
        for table in list(selected_tables):
            for local_col, ref_str in fk_map.get(table, []):
                # (1) always add the FK column itself
                fk_key = (table, local_col)
                if fk_key in available_field_set:
                    result.add(fk_key)
                # (2) add the referenced column when its table is already selected
                m = re.match(r"(\S+)\((\S+)\)", ref_str)  # e.g. "Player(player_api_id)"
                if m:
                    ref_table, ref_col = m.group(1), m.group(2)
                    ref_key = (ref_table, ref_col)
                    if ref_table in selected_tables and ref_key in available_field_set:
                        result.add(ref_key)
        return result

    s1_tables = {t for t, _c in s1_selected}
    s1_extended = _add_pk_fk_for_tables(s1_selected, s1_tables)

    # ------------------------------------------------------------------
    # Auto-promote high-confidence FAISS matches from schema_hints to S1
    # ------------------------------------------------------------------
    # If a schema hint (field name reference from grounding context) has
    # cosine similarity >= 0.8 to a field in the FAISS index, include it
    # in S1 directly — the LLM may not select it if the candidate list
    # is large or if the column name is ambiguous.
    _HIGH_CONF_S1_THRESHOLD = 0.8
    for hint in grounding_context.schema_hints:
        try:
            hint_results = faiss_index.query(hint, top_k=3)
        except OSError as exc:
            logger.critical(
                "FAISS hint query OS error (broken pipe?): %s — skipping hint %r",
                exc,
                hint,
            )
            continue
        for fm in hint_results:
            if fm.similarity_score >= _HIGH_CONF_S1_THRESHOLD:
                key = (fm.table, fm.column)
                if key in available_field_set and key not in s1_extended:
                    logger.debug(
                        "Auto-promoting high-confidence hint to S1: %s.%s (score=%.2f, hint=%r)",
                        fm.table, fm.column, fm.similarity_score, hint,
                    )
                    s1_extended.add(key)

    # ------------------------------------------------------------------
    # Hard cap: S₁ must not exceed 20 fields to prevent schema bloat
    # on large-schema databases.  Trim by FAISS rank; PK/FK columns
    # for non-ranked tables come last.
    # ------------------------------------------------------------------
    _S1_MAX_FIELDS = 20
    if len(s1_extended) > _S1_MAX_FIELDS:
        candidate_keys_ordered = [(t, c) for t, c, _ss, _ls in candidates]
        candidate_key_set = set(candidate_keys_ordered)
        s1_ranked = [k for k in candidate_keys_ordered if k in s1_extended]
        s1_unranked = [k for k in s1_extended if k not in candidate_key_set]
        s1_extended = set((s1_ranked + s1_unranked)[:_S1_MAX_FIELDS])
        logger.debug(
            "S1 capped to %d fields (hard cap=%d)",
            len(s1_extended),
            _S1_MAX_FIELDS,
        )

    # ------------------------------------------------------------------
    # Step 6.3 — Iteration 2: Recall Expansion (S₂)
    # ------------------------------------------------------------------
    # Show remaining candidates (not already in S₁ extended)
    remaining_candidates = [
        (t, c, ss, ls) for t, c, ss, ls in candidates
        if (t, c) not in s1_extended
    ]

    # Compute S₁ coverage over the full FAISS candidate set.
    # Skip the S₂ LLM call when it would provide little or no new signal:
    #   • total_candidates == 0  → FAISS returned nothing; nothing to expand
    #   • remaining == 0         → S₁ already consumed every candidate
    #   • s1_coverage >= 0.80   → S₁ covers ≥80% of FAISS candidates;
    #                              only a marginal tail remains
    #   • len(remaining) < 3    → fewer than 3 new fields; not worth a call
    total_candidates = len(candidates)
    s1_coverage = len(s1_extended) / max(total_candidates, 1)
    few_remaining = len(remaining_candidates) < 3

    skip_s2 = (
        total_candidates == 0
        or len(remaining_candidates) == 0
        or s1_coverage >= 0.80
        or few_remaining
    )

    if skip_s2:
        logger.debug(
            "Skipping S2 expansion: S1 covers %.0f%% of %d candidates "
            "(%d remaining)",
            s1_coverage * 100,
            total_candidates,
            len(remaining_candidates),
        )
        s2_extended = set(s1_extended)
    else:
        remaining_lines = [
            f"Table: {t} | Column: {c} | {ss}" for t, c, ss, _ls in remaining_candidates
        ]
        system_block_2_remaining = CacheableText(
            text="Candidate fields:\n" + "\n".join(remaining_lines),
            cache=True,
        )

        user_msg_s2 = (
            f"Question: {safe_question}\n"
            f"Evidence: {safe_evidence}\n"
            f"Matched values:\n{formatted_cells}\n\n"
            "Select columns that MIGHT be needed to answer this question "
            "(cast a wider net — include columns that could be relevant). "
            "Also include output columns (SELECT clause) that the question asks "
            "to return, not only columns needed for WHERE conditions and JOINs. "
            "For each column, set 'role' to the SQL clause it primarily serves "
            "(select, where, join, group_by, or order_by)."
        )

        try:
            response_s2 = await client.generate(
                model=settings.model_powerful_list,
                system=[system_block_1, system_block_2_remaining],
                messages=[{"role": "user", "content": user_msg_s2}],
                tools=[_SELECT_COLUMNS_TOOL],
                tool_choice_name="select_columns",
                max_tokens=8192,
                temperature=0.0,
            )
            if response_s2.tool_inputs:
                raw_s2 = response_s2.tool_inputs[0].get("selected_fields", [])
                if not isinstance(raw_s2, list):
                    logger.warning(
                        "Schema linker S2: 'selected_fields' is not a list (%r); treating as empty",
                        type(raw_s2).__name__,
                    )
                    raw_s2 = []
                s2_raw: list[dict] = [f for f in raw_s2 if isinstance(f, dict)]
            else:
                logger.warning("Schema linker S2: no tool_inputs in response; using S1 as S2")
                s2_raw = []
        except LLMError as exc:
            logger.error(
                "Schema linker S2 LLM call failed (%s); using S1 as S2 fallback.", exc
            )
            s2_raw = []

        s2_new: set[tuple[str, str]] = set()
        for item in s2_raw:
            t = item.get("table", "")
            c = item.get("column", "")
            r = item.get("reason", "")
            role = item.get("role", "")
            key = (t, c)
            if key in available_field_set:
                s2_new.add(key)
                role_tag = f"[{role}] " if role else ""
                s1_reasons.append(f"{role_tag}{t}.{c}: {r}")
            else:
                logger.warning(
                    "Hallucinated field filtered out from S₂: %s.%s", t, c
                )

        # S₂ = S₁ ∪ S₂_new ∪ their PKs/FKs
        s2_combined = s1_extended | s2_new
        s2_tables = {t for t, _c in s2_combined}
        s2_extended = _add_pk_fk_for_tables(s2_combined, s2_tables)

    # ------------------------------------------------------------------
    # Enforce S₁ ⊆ S₂ invariant
    # ------------------------------------------------------------------
    s2_extended = s2_extended | s1_extended  # guarantee S₁ ⊆ S₂

    # ------------------------------------------------------------------
    # Cap S₂ to prevent over-expansion on large-schema databases.
    # Formula: s2_cap = min(len(s1_extended) + 10, 25)
    # S₁ fields are always preserved; excess recall fields are trimmed
    # by FAISS rank (the order of the `candidates` list).
    # ------------------------------------------------------------------
    s2_cap = min(len(s1_extended) + 10, 25)
    if len(s2_extended) > s2_cap:
        s2_only = s2_extended - s1_extended
        slots_available = max(s2_cap - len(s1_extended), 0)
        # Prioritise by FAISS rank; unranked fields (e.g. PK/FK additions for
        # S2-only tables) come last
        candidate_keys = [(t, c) for t, c, _ss, _ls in candidates]
        candidate_key_set = set(candidate_keys)
        s2_only_ranked = [k for k in candidate_keys if k in s2_only]
        s2_only_unranked = [k for k in s2_only if k not in candidate_key_set]
        s2_trimmed = set((s2_only_ranked + s2_only_unranked)[:slots_available])
        logger.debug(
            "S₂ capped: %d → %d fields (s1=%d, cap=%d)",
            len(s2_extended),
            len(s1_extended) + len(s2_trimmed),
            len(s1_extended),
            s2_cap,
        )
        s2_extended = s1_extended | s2_trimmed

    # ------------------------------------------------------------------
    # Step 6.4 — Schema Rendering
    # ------------------------------------------------------------------
    ddl_tables = _parse_ddl_tables(full_ddl)
    md_tables = _parse_markdown_tables(full_markdown)

    def _group_by_table(fields: set[tuple[str, str]]) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for t, c in fields:
            result.setdefault(t, set()).add(c)
        return result

    s1_by_table = _group_by_table(s1_extended)
    s2_by_table = _group_by_table(s2_extended)

    s1_ddl = _render_ddl_subset(ddl_tables, s1_by_table)
    s1_markdown = _render_markdown_subset(md_tables, s1_by_table)
    s2_ddl = _render_ddl_subset(ddl_tables, s2_by_table)
    s2_markdown = _render_markdown_subset(md_tables, s2_by_table)

    # Convert sets to sorted lists for stable output
    s1_fields_list = sorted(s1_extended)
    s2_fields_list = sorted(s2_extended)

    return LinkedSchemas(
        s1_ddl=s1_ddl,
        s1_markdown=s1_markdown,
        s2_ddl=s2_ddl,
        s2_markdown=s2_markdown,
        s1_fields=s1_fields_list,
        s2_fields=s2_fields_list,
        selection_reasoning="\n".join(s1_reasons),
    )
