#!/usr/bin/env python3
"""
Checkpoint B Inspection Script
===============================
Gathers detailed information about the offline preprocessing pipeline results
for review. Run after the pipeline completes.

Usage:
    python checkpoint_B_review/inspect_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.config.settings import settings  # noqa: E402
from src.indexing.lsh_index import LSHIndex  # noqa: E402
from src.indexing.faiss_index import FAISSIndex  # noqa: E402
from src.indexing.example_store import ExampleStore  # noqa: E402


def hr(char="=", width=72):
    print(char * width)


def section(title):
    hr()
    print(f"  {title}")
    hr()


def subsection(title):
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# 1. Pipeline summary
# ---------------------------------------------------------------------------
def pipeline_summary():
    section("1. PIPELINE SUMMARY — All 11 BIRD dev databases")
    preprocessed = Path(settings.preprocessed_dir)

    profiles_dir = preprocessed / "profiles"
    summaries_dir = preprocessed / "summaries"
    schemas_dir = preprocessed / "schemas"
    indices_dir = preprocessed / "indices"

    db_ids = sorted(p.stem for p in profiles_dir.glob("*.json"))
    print(f"\nDatabases processed: {len(db_ids)}")
    print(f"  {', '.join(db_ids)}\n")

    print(f"{'DB':30s} {'Profile':>8} {'Summary':>8} {'DDL':>6} {'MD':>5} {'LSH':>10} {'FAISS':>10}")
    print("-" * 80)
    total_lsh_size = 0
    for db_id in db_ids:
        p = profiles_dir / f"{db_id}.json"
        s = summaries_dir / f"{db_id}.json"
        ddl = schemas_dir / f"{db_id}_ddl.sql"
        md = schemas_dir / f"{db_id}_markdown.md"
        lsh = indices_dir / f"{db_id}_lsh.pkl"
        faiss = indices_dir / f"{db_id}_faiss.index"

        def sz(path):
            if path.exists():
                b = path.stat().st_size
                if b > 1_000_000_000:
                    return f"{b/1e9:.1f}GB"
                elif b > 1_000_000:
                    return f"{b/1e6:.1f}MB"
                else:
                    return f"{b/1e3:.1f}KB"
            return "MISSING"

        lsh_sz = lsh.stat().st_size if lsh.exists() else 0
        total_lsh_size += lsh_sz
        print(
            f"{db_id:30s} {'✓' if p.exists() else '✗':>8} "
            f"{'✓' if s.exists() else '✗':>8} "
            f"{'✓' if ddl.exists() else '✗':>6} "
            f"{'✓' if md.exists() else '✗':>5} "
            f"{sz(lsh):>10} "
            f"{sz(faiss):>10}"
        )

    ex_faiss = indices_dir / "example_store.faiss"
    ex_meta = indices_dir / "example_store_metadata.json"
    print(f"\nExample store: {sz(ex_faiss)}, metadata: {sz(ex_meta)}")
    print(f"Total LSH index size: {total_lsh_size/1e9:.2f} GB")


# ---------------------------------------------------------------------------
# 2. california_schools — LSH breakdown
# ---------------------------------------------------------------------------
def lsh_inspection():
    section("2. CALIFORNIA_SCHOOLS — LSH Index Analysis")
    indices_dir = Path(settings.preprocessed_dir) / "indices"
    lsh_path = indices_dir / "california_schools_lsh.pkl"

    if not lsh_path.exists():
        print("ERROR: california_schools_lsh.pkl not found!")
        return

    print(f"\nLoading LSH index from {lsh_path} …")
    lsh = LSHIndex.load(str(lsh_path))
    n = len(lsh._minhashes)
    print(f"Total indexed entries: {n:,}")

    from collections import Counter
    table_col_counts: Counter[str] = Counter()
    table_counts: Counter[str] = Counter()
    for key in lsh._minhashes:
        tc = key.rsplit("::", 1)[0]
        table, col = tc.split(".", 1)
        table_col_counts[tc] += 1
        table_counts[table] += 1

    subsection("Per-table entry counts")
    for tbl, cnt in table_counts.most_common():
        print(f"  {tbl}: {cnt:,} values")

    subsection("Top 20 columns by value count")
    for tc, cnt in table_col_counts.most_common(20):
        print(f"  {tc}: {cnt:,}")

    # ------------------------------------------------------------------
    # Fuzzy query samples
    # ------------------------------------------------------------------
    subsection("Fuzzy query: 'Alameda' (county)")
    results = lsh.query("Alameda", top_k=5)
    if results:
        for r in results:
            print(f"  {r.table}.{r.column} = '{r.matched_value}'  sim={r.similarity_score:.3f}  exact={r.exact_match}")
    else:
        print("  No results")

    subsection("Fuzzy query: 'Berkley' (typo for 'Berkeley')")
    results = lsh.query("Berkley", top_k=5)
    if results:
        for r in results:
            print(f"  {r.table}.{r.column} = '{r.matched_value}'  sim={r.similarity_score:.3f}  exact={r.exact_match}")
    else:
        print("  No results")

    subsection("Fuzzy query: 'Los Angelos' (typo for 'Los Angeles')")
    results = lsh.query("Los Angelos", top_k=5)
    if results:
        for r in results:
            print(f"  {r.table}.{r.column} = '{r.matched_value}'  sim={r.similarity_score:.3f}  exact={r.exact_match}")
    else:
        print("  No results")


# ---------------------------------------------------------------------------
# 3. california_schools — FAISS query
# ---------------------------------------------------------------------------
def faiss_inspection():
    section("3. CALIFORNIA_SCHOOLS — FAISS Semantic Field Search")
    indices_dir = Path(settings.preprocessed_dir) / "indices"
    faiss_path = indices_dir / "california_schools_faiss.index"
    fields_path = indices_dir / "california_schools_faiss_fields.json"

    if not faiss_path.exists():
        print("ERROR: FAISS index not found!")
        return

    faiss_idx = FAISSIndex.load(str(faiss_path), str(fields_path))
    print(f"\nFAISS index has {faiss_idx._index.ntotal} fields")

    queries = [
        "county name",
        "free meal count in schools",
        "school district enrollment",
        "academic performance score",
        "student eligibility for reduced price lunch",
    ]
    for q in queries:
        subsection(f"Query: '{q}'")
        results = faiss_idx.query(q, top_k=5)
        for r in results:
            print(f"  {r.table}.{r.column}  score={r.similarity_score:.4f}")
            print(f"    → {r.long_summary[:80]}")


# ---------------------------------------------------------------------------
# 4. Example store — retrieval for "free meal counts in schools"
# ---------------------------------------------------------------------------
def example_store_inspection():
    section("4. EXAMPLE STORE — Few-Shot Retrieval")
    indices_dir = Path(settings.preprocessed_dir) / "indices"
    ex_faiss = indices_dir / "example_store.faiss"
    ex_meta = indices_dir / "example_store_metadata.json"

    if not ex_faiss.exists():
        print("ERROR: example_store.faiss not found!")
        return

    ex_store = ExampleStore.load(str(ex_faiss), str(ex_meta))
    print(f"\nExample store: {len(ex_store._metadata)} entries")

    queries = [
        ("free meal counts in schools", "california_schools"),
        ("how many cards have converted mana cost greater than 5", "card_games"),
        ("highest salary in the company", "financial"),
    ]
    for q, exclude_db in queries:
        subsection(f"Query: '{q}' (excluding {exclude_db})")
        results = ex_store.query(q, db_id=exclude_db, top_k=3)
        if not results:
            print("  No results")
        for ex in results:
            print(f"  [{ex.db_id}] Q: {ex.question[:70]}")
            print(f"         SQL: {ex.sql[:70]}")
            print()


# ---------------------------------------------------------------------------
# 5. Field summary quality spot-check
# ---------------------------------------------------------------------------
def summary_quality_check():
    section("5. FIELD SUMMARY QUALITY — california_schools spot check")
    summaries_dir = Path(settings.preprocessed_dir) / "summaries"
    summary_path = summaries_dir / "california_schools.json"

    if not summary_path.exists():
        print("ERROR: summary not found!")
        return

    with open(summary_path) as f:
        data = json.load(f)

    # Show first 15 field summaries
    summaries = data.get("field_summaries", [])
    print(f"\nTotal field summaries: {len(summaries)}\n")
    for fs in summaries[:15]:
        tbl = fs.get("table_name", "?")
        col = fs.get("column_name", "?")
        summ = fs.get("short_summary", fs.get("summary", ""))
        default_marker = " [DEFAULT]" if "The " in summ and " field in the " in summ and " table." in summ else ""
        print(f"  {tbl}.{col}:{default_marker}")
        print(f"    {summ[:100]}")
        print()

    # Check for default summaries
    n_default = sum(
        1 for fs in summaries
        if "The " in fs.get("short_summary", fs.get("summary", ""))
        and " field in the " in fs.get("short_summary", fs.get("summary", ""))
        and " table." in fs.get("short_summary", fs.get("summary", ""))
    )
    print(f"Fields with default (fallback) summaries: {n_default}/{len(summaries)}")


# ---------------------------------------------------------------------------
# 6. Batch failure analysis
# ---------------------------------------------------------------------------
def batch_failure_analysis():
    section("6. SUMMARIZER BATCH FAILURE ANALYSIS")
    print("""
During the pipeline run, several batch-of-6 failures occurred:
  - card_games/cards: 5 batch failures → retried individually
  - card_games/sets:  1 batch failure  → retried individually

Root cause: Gemini context window limit hit when a batch of 6 columns
includes very long purchaseUrls/text fields. Individual retry succeeds
because each column fits within the context limit alone.

Impact: Slight cost increase (individual retries use more API calls)
but correctness is maintained (individual retry always succeeds).

Recommendation: Reduce BATCH_SIZE from 6 to 4 for databases with
long free-text columns, or add a token-estimate pre-check before batching.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nCheckpoint B — Offline Pipeline Inspection Report")
    print(f"Provider: {settings.llm_provider}  |  "
          f"Fast: {settings.model_fast}  |  "
          f"Powerful: {settings.model_powerful}\n")

    pipeline_summary()
    lsh_inspection()
    faiss_inspection()
    example_store_inspection()
    summary_quality_check()
    batch_failure_analysis()

    print("\n" + "=" * 72)
    print("  END OF CHECKPOINT B INSPECTION REPORT")
    print("=" * 72 + "\n")
