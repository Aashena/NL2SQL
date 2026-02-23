#!/usr/bin/env python3
"""
Checkpoint C — Ops 5+6 End-to-End Test on First 5 BIRD Dev Questions

Tests context grounding (Op 5) + schema linking (Op 6) on real BIRD data.
CACHE_LLM_RESPONSES=true is assumed (set in .env).
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.config.settings import settings
from src.data.bird_loader import load_bird_split
from src.grounding.context_grounder import ground_context
from src.indexing.example_store import ExampleStore
from src.indexing.faiss_index import FAISSIndex
from src.indexing.lsh_index import LSHIndex
from src.schema_linking.schema_linker import link_schema


def hr(char="=", width=72):
    print(char * width)


def load_california_schools_artifacts():
    """Load pre-built offline artifacts for california_schools."""
    indices_dir = Path(settings.preprocessed_dir) / "indices"

    print("Loading california_schools LSH index (1.2 GB, may take ~10s)…")
    lsh = LSHIndex.load(str(indices_dir / "california_schools_lsh.pkl"))
    print(f"  LSH loaded: {len(lsh._minhashes):,} entries")

    print("Loading california_schools FAISS index…")
    faiss = FAISSIndex.load(
        str(indices_dir / "california_schools_faiss.index"),
        str(indices_dir / "california_schools_faiss_fields.json"),
    )
    print(f"  FAISS loaded: {faiss._index.ntotal} fields")

    print("Loading example store…")
    ex_store = ExampleStore.load(
        str(indices_dir / "example_store.faiss"),
        str(indices_dir / "example_store_metadata.json"),
    )
    print(f"  Example store: {len(ex_store._metadata)} entries\n")

    return lsh, faiss, ex_store


def load_available_fields(db_id: str) -> list[tuple[str, str, str, str]]:
    """Build available_fields from pre-built summary JSON."""
    summary_path = Path(settings.preprocessed_dir) / "summaries" / f"{db_id}.json"
    with open(summary_path) as f:
        data = json.load(f)
    return [
        (fs["table_name"], fs["column_name"], fs["short_summary"], fs["long_summary"])
        for fs in data["field_summaries"]
    ]


def load_full_schemas(db_id: str) -> tuple[str, str]:
    """Load DDL and Markdown schemas from disk."""
    schemas_dir = Path(settings.preprocessed_dir) / "schemas"
    ddl = (schemas_dir / f"{db_id}_ddl.sql").read_text()
    markdown = (schemas_dir / f"{db_id}_markdown.md").read_text()
    return ddl, markdown


def extract_columns_from_sql(sql: str) -> set[str]:
    """Rough extraction of column names referenced in SQL."""
    # Remove string literals and comments
    sql_clean = re.sub(r"'[^']*'", "", sql)
    sql_clean = re.sub(r"--[^\n]*", "", sql_clean)

    # Match identifiers in backticks or double-quotes or plain words
    cols = set()
    cols |= set(re.findall(r'`([^`]+)`', sql_clean))
    cols |= set(re.findall(r'"([^"]+)"', sql_clean))
    # Plain word identifiers (uppercase, camelCase, or mixed)
    cols |= set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', sql_clean))

    # Filter out SQL keywords
    _SQL_KEYWORDS = {
        "SELECT", "FROM", "WHERE", "JOIN", "ON", "AS", "AND", "OR", "NOT",
        "IN", "IS", "NULL", "GROUP", "BY", "ORDER", "HAVING", "LIMIT",
        "DISTINCT", "COUNT", "SUM", "AVG", "MAX", "MIN", "INNER", "LEFT",
        "RIGHT", "OUTER", "UNION", "ALL", "CASE", "WHEN", "THEN", "ELSE",
        "END", "LIKE", "BETWEEN", "EXISTS", "WITH", "CTE", "ASC", "DESC",
    }
    return {c for c in cols if c.upper() not in _SQL_KEYWORDS and len(c) > 1}


async def run_checkpoint_c():
    hr()
    print("  CHECKPOINT C — Ops 5+6 End-to-End Test")
    print(f"  Provider: {settings.llm_provider}  |  Cache: {settings.cache_llm_responses}")
    hr()

    # Load BIRD dev questions
    dev_entries = load_bird_split("dev", Path(settings.bird_data_dir))
    questions = dev_entries[:5]

    # Load artifacts (all 5 questions are california_schools)
    lsh, faiss, ex_store = load_california_schools_artifacts()
    available_fields = load_available_fields("california_schools")
    full_ddl, full_markdown = load_full_schemas("california_schools")

    all_fields_set = {(t, c) for t, c, _, _ in available_fields}

    for i, entry in enumerate(questions):
        hr("-")
        print(f"\nQUESTION {i+1}/5  [db={entry.db_id}  difficulty={entry.difficulty}]")
        print(f"Q: {entry.question}")
        print(f"Evidence: {entry.evidence or '(none)'}")
        print(f"Ground truth SQL: {entry.SQL}")
        print()

        # --- Op 5: Context Grounding ---
        print("[Op 5] Running context grounding…")
        grounding = await ground_context(
            question=entry.question,
            evidence=entry.evidence,
            db_id=entry.db_id,
            lsh_index=lsh,
            example_store=ex_store,
        )

        print(f"  Extracted literals → LSH matched cells: {len(grounding.matched_cells)}")
        for cm in grounding.matched_cells[:5]:
            print(f"    {cm.table}.{cm.column} = '{cm.matched_value}' (sim={cm.similarity_score:.2f})")
        if len(grounding.matched_cells) > 5:
            print(f"    … +{len(grounding.matched_cells)-5} more")

        print(f"  Schema hints: {grounding.schema_hints}")
        print(f"  Few-shot examples retrieved: {len(grounding.few_shot_examples)}")
        print()

        # --- Op 6: Schema Linking ---
        print("[Op 6] Running schema linking…")
        try:
            schemas = await link_schema(
                question=entry.question,
                evidence=entry.evidence,
                grounding_context=grounding,
                faiss_index=faiss,
                full_ddl=full_ddl,
                full_markdown=full_markdown,
                available_fields=available_fields,
            )

            print(f"  S₁ fields ({len(schemas.s1_fields)}): {schemas.s1_fields}")
            print(f"  S₂ fields ({len(schemas.s2_fields)}): {schemas.s2_fields}")
            print()

            # S₁ markdown (first 20 lines)
            s1_md_lines = schemas.s1_markdown.strip().split("\n")[:20]
            print("  S₁ Markdown (first 20 lines):")
            for line in s1_md_lines:
                print(f"    {line}")
            print()

            # Ground truth analysis
            gt_cols = extract_columns_from_sql(entry.SQL)
            # Find columns in S₂ that match ground truth
            s2_col_names = {col for _, col in schemas.s2_fields}
            s2_table_cols = {f"{tbl}.{col}" for tbl, col in schemas.s2_fields}

            print("  Ground truth analysis:")
            print(f"    Columns referenced in SQL: {sorted(gt_cols)}")
            matched_in_s2 = gt_cols & s2_col_names
            missed = gt_cols - s2_col_names - {"FROM", "SELECT"}
            # Check if any ground truth column is NOT in S₂
            important_missed = [
                col for col in gt_cols
                if col.lower() not in {"from", "select", "where"}
                and not any(col == c for _, c in schemas.s2_fields)
                and len(col) > 3  # skip short keywords
            ]
            if not important_missed:
                print("    ✅ All important columns are in S₂")
            else:
                print(f"    ⚠️  Columns possibly missed by S₂: {important_missed}")

        except Exception as e:
            print(f"  ❌ Schema linking failed: {e}")
            import traceback
            traceback.print_exc()

        print()

    hr()
    print("  CHECKPOINT C COMPLETE")
    hr()


if __name__ == "__main__":
    asyncio.run(run_checkpoint_c())
