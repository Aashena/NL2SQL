#!/usr/bin/env python3
"""
Targeted test for the MAX_TOKENS fix in generators B2 and C.

Root cause (from Checkpoint D inspection report):
  Generators B2 (complex SQL, gemini-2.5-pro) and C (ICL, gemini-2.5-pro) were
  hitting `finish_reason=MAX_TOKENS` with no text output because the model's
  implicit reasoning consumed the 2000-token budget before producing any SQL.

Fixes applied:
  1. standard_generator.py: B2 max_tokens 2000 → 4096
  2. icl_generator.py:       C  max_tokens 2000 → 4096
  3. gemini_client.py:       One-shot retry with 2× token budget when MAX_TOKENS
                             hit with zero output (text=None, tool_inputs=[])

Target questions (confirmed B2/C failures in Checkpoint D smoke test):
  Q28   california_schools  challenging
  Q431  card_games          challenging
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

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

# The two questions where MAX_TOKENS was confirmed as root cause in Checkpoint D smoke test
TARGET_QIDs = {28, 431}

# Expected baseline from pre-fix smoke test (for regression comparison)
_BASELINE = {
    28:  {"n_empty_b2c": 2, "oracle": False},  # B2/C failures → improved but ORACLE=NO
    431: {"n_empty_b2c": 4, "oracle": False},  # B2+C failures → partial improvement, ORACLE=NO
}


def _count_b2c_empty(candidates: list[dict]) -> int:
    """Count candidates from generators B2 or C that are empty (error_flag=True or sql='')."""
    b2c_ids = {"complex_B2_s1", "complex_B2_s2", "icl_C1", "icl_C2", "icl_C3"}
    return sum(
        1 for c in candidates
        if c.get("generator_id") in b2c_ids and (c.get("error_flag") or not c.get("sql"))
    )


async def main():
    print("=" * 72)
    print("  MAX-TOKENS FIX TEST — Q28 + Q431 (B2/C truncation root cause)")
    print(f"  Provider: {settings.llm_provider}  |  Cache: {settings.cache_llm_responses}")
    print("=" * 72)
    print()
    print("Fixes being validated:")
    print("  • B2 max_tokens: 2000 → 4096  (standard_generator.py)")
    print("  • C  max_tokens: 2000 → 4096  (icl_generator.py)")
    print("  • gemini_client: 1-shot retry at 2× tokens when MAX_TOKENS + no output")
    print()

    bird_data_dir = Path(settings.bird_data_dir)
    dev_entries = load_bird_split("dev", bird_data_dir)
    target_entries = [e for e in dev_entries if e.question_id in TARGET_QIDs]

    if not target_entries:
        print("ERROR: No target questions found. Check bird_data_dir in .env")
        return

    print(f"Found {len(target_entries)} target questions:")
    for e in sorted(target_entries, key=lambda x: x.question_id):
        print(f"  QID={e.question_id}  DB={e.db_id}  difficulty={e.difficulty}")
        print(f"    Q: {e.question[:80]}")

    gt_sqls = load_gt_sql(bird_data_dir)
    indices_dir = Path(settings.preprocessed_dir) / "indices"

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

    # Per-question breakdown
    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)
    print(
        f"\n{'QID':>4}  {'DB':25}  {'Diff':10}  {'Cands':>5}  {'ExecOK':>6}  "
        f"{'B2/C empty':>10}  {'Oracle':>6}  Notes"
    )
    print(f"  {'-'*4}  {'-'*25}  {'-'*10}  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*6}  {'-'*30}")

    for r in sorted(all_results, key=lambda x: x["question_id"]):
        cands = r.get("candidates", [])
        n_exec_ok = sum(1 for c in cands if c.get("executed_ok"))
        n_b2c_empty = _count_b2c_empty(cands)
        oracle = "YES" if r.get("oracle_match") else "NO"
        baseline = _BASELINE.get(r["question_id"], {})
        prev_empty = baseline.get("n_empty_b2c", "?")
        prev_oracle = "YES" if baseline.get("oracle") else "NO"
        note = f"B2/C empty: {prev_empty}→{n_b2c_empty}  Oracle: {prev_oracle}→{oracle}"
        print(
            f"  {r['question_id']:>4}  {r['db_id'][:25]:25}  "
            f"{r.get('difficulty','?'):10}  {len(cands):>5}  {n_exec_ok:>6}  "
            f"{n_b2c_empty:>10}  {oracle:>6}  {note}"
        )

    n_oracle = sum(1 for r in all_results if r.get("oracle_match"))
    n_q = len(all_results)
    total_b2c_empty_before = sum(_BASELINE.get(r["question_id"], {}).get("n_empty_b2c", 0) for r in all_results)
    total_b2c_empty_after = sum(_count_b2c_empty(r.get("candidates", [])) for r in all_results)
    n_error = sum(1 for r in all_results if r.get("error"))

    print(f"\nOracle: {n_oracle}/{n_q}")
    print(f"B2/C empty candidates: {total_b2c_empty_before} → {total_b2c_empty_after} (target: 0)")
    print(f"Pipeline errors: {n_error}/{n_q}  |  Elapsed: {total_elapsed:.1f}s")

    # Verdict
    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)
    if total_b2c_empty_after == 0:
        print("  PASS: All B2 and C candidates produced output (MAX_TOKENS fix confirmed)")
    elif total_b2c_empty_after < total_b2c_empty_before:
        print(f"  PARTIAL: B2/C empty rate reduced {total_b2c_empty_before}→{total_b2c_empty_after} (improvement, but not zero)")
    else:
        print("  FAIL: B2/C empty rate unchanged — fix did not take effect")

    # Save results
    out = {
        "test": "max_tokens_fix",
        "target_qids": sorted(TARGET_QIDs),
        "b2c_empty_before": total_b2c_empty_before,
        "b2c_empty_after": total_b2c_empty_after,
        "oracle_after": n_oracle,
        "elapsed_s": round(total_elapsed, 1),
        "results": all_results,
    }
    out_path = _ROOT / "checkpoint_D_review" / "max_tokens_fix_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
