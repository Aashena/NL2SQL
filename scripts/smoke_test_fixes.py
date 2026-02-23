#!/usr/bin/env python3
"""
Post-fix smoke test: re-runs the 5 questions that failed in Checkpoint D
due to LLM errors, to confirm the fixes work.

Targeted questions:
  Q23   california_schools  moderate    → MALFORMED_FUNCTION_CALL crash (0 candidates)
  Q28   california_schools  challenging → B2/C None response failures
  Q431  card_games          challenging → B2/C None response failures
  Q1027 european_football_2 simple      → S2 over-expansion (36 fields), ORACLE=NO
  Q1295 thrombosis_prediction challenging → complex query, domain knowledge
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Reuse helpers from run_checkpoint_d
from scripts.run_checkpoint_d import (
    execute_sql,
    load_available_fields,
    load_full_schemas,
    load_gt_sql,
    result_sets_match,
    run_one_question,
)

from src.config.settings import settings
from src.data.bird_loader import load_bird_split
from src.indexing.example_store import ExampleStore
from src.indexing.faiss_index import FAISSIndex
from src.indexing.lsh_index import LSHIndex

# The 5 target question IDs from Checkpoint D failures
TARGET_QIDs = {23, 28, 431, 1027, 1295}


async def main():
    print("=" * 72)
    print("  POST-FIX SMOKE TEST — 5 Checkpoint D failing questions")
    print(f"  Provider: {settings.llm_provider}  |  Cache: {settings.cache_llm_responses}")
    print("=" * 72)

    bird_data_dir = Path(settings.bird_data_dir)
    dev_entries = load_bird_split("dev", bird_data_dir)
    target_entries = [e for e in dev_entries if e.question_id in TARGET_QIDs]

    if not target_entries:
        print("ERROR: No target questions found. Check bird_data_dir in .env")
        return

    print(f"\nFound {len(target_entries)} target questions:")
    for e in sorted(target_entries, key=lambda x: x.question_id):
        print(f"  QID={e.question_id}  DB={e.db_id}  difficulty={e.difficulty}")
        print(f"    Q: {e.question[:80]}")

    gt_sqls = load_gt_sql(bird_data_dir)
    indices_dir = Path(settings.preprocessed_dir) / "indices"

    # Pre-load artifacts per database
    dbs_needed = sorted(set(e.db_id for e in target_entries))
    db_artifacts: dict[str, dict] = {}
    print(f"\nPre-loading artifacts for: {dbs_needed}")

    for db_id in dbs_needed:
        print(f"  Loading {db_id}...", end=" ", flush=True)
        lsh = LSHIndex.load(str(indices_dir / f"{db_id}_lsh.pkl"))
        faiss_idx = FAISSIndex.load(
            str(indices_dir / f"{db_id}_faiss.index"),
            str(indices_dir / f"{db_id}_faiss_fields.json"),
        )
        available_fields = load_available_fields(db_id)
        full_ddl, full_markdown = load_full_schemas(db_id)
        db_path = bird_data_dir / "dev" / "dev_databases" / db_id / f"{db_id}.sqlite"
        db_artifacts[db_id] = {
            "lsh": lsh,
            "faiss_idx": faiss_idx,
            "available_fields": available_fields,
            "full_ddl": full_ddl,
            "full_markdown": full_markdown,
            "db_path": db_path,
        }
        print(f"done ({faiss_idx._index.ntotal} FAISS fields)")

    print("Loading example store...", end=" ", flush=True)
    ex_store = ExampleStore.load(
        str(indices_dir / "example_store.faiss"),
        str(indices_dir / "example_store_metadata.json"),
    )
    print(f"done ({len(ex_store._metadata)} entries)")

    # Run each question
    all_results: list[dict] = []
    total_start = time.time()
    sorted_entries = sorted(target_entries, key=lambda x: x.question_id)

    for i, entry in enumerate(sorted_entries):
        gt_sql = gt_sqls[entry.question_id]
        art = db_artifacts[entry.db_id]
        result = await run_one_question(
            entry=entry,
            gt_sql=gt_sql,
            lsh=art["lsh"],
            faiss_idx=art["faiss_idx"],
            ex_store=ex_store,
            available_fields=art["available_fields"],
            full_ddl=art["full_ddl"],
            full_markdown=art["full_markdown"],
            db_path=art["db_path"],
            q_num=i + 1,
            total=len(sorted_entries),
        )
        all_results.append(result)

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 72)
    print("  SMOKE TEST SUMMARY")
    print("=" * 72)

    print(f"\n{'QID':>4}  {'DB':25}  {'Diff':10}  {'#Cands':>6}  {'#ExecOK':>7}  {'#Empty':>6}  {'Oracle':>6}  {'Error'}")
    print(f"  {'-'*4}  {'-'*25}  {'-'*10}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*30}")

    for r in sorted(all_results, key=lambda x: x["question_id"]):
        cands = r.get("candidates", [])
        n_exec_ok = sum(1 for c in cands if c.get("executed_ok"))
        n_empty = sum(1 for c in cands if c.get("error_flag") or not c.get("sql"))
        oracle = "YES" if r.get("oracle_match") else "NO"
        err = (r.get("error") or "")[:35]
        print(
            f"  {r['question_id']:>4}  {r['db_id'][:25]:25}  "
            f"{r.get('difficulty','?'):10}  {len(cands):>6}  {n_exec_ok:>7}  "
            f"{n_empty:>6}  {oracle:>6}  {err}"
        )

    n_oracle = sum(1 for r in all_results if r.get("oracle_match"))
    n_q = len(all_results)
    n_error = sum(1 for r in all_results if r.get("error"))

    print(f"\nOracle: {n_oracle}/{n_q}  |  Pipeline errors: {n_error}/{n_q}  |  Elapsed: {total_elapsed:.1f}s")

    # Save to JSON for inspection
    out_path = _ROOT / "checkpoint_D_review" / "smoke_test_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
