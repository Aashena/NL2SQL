#!/usr/bin/env python3
"""
Checkpoint D — Full Generation Pipeline Analysis

Runs Op 5 (Context Grounding) → Op 6 (Schema Linking) → Op 7 (all 3 generators)
on 33 BIRD dev questions: 3 per each of 11 databases (1 simple, 1 moderate,
1 challenging), using random.seed(42) for reproducibility.

Produces:
  checkpoint_D_review/results.json
  checkpoint_D_review/inspection_report.md
"""
from __future__ import annotations

import asyncio
import json
import random
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.config.settings import settings
from src.data.bird_loader import load_bird_split, BirdEntry
from src.generation.icl_generator import ICLGenerator
from src.generation.reasoning_generator import ReasoningGenerator
from src.generation.standard_generator import StandardAndComplexGenerator
from src.grounding.context_grounder import ground_context
from src.indexing.example_store import ExampleStore
from src.indexing.faiss_index import FAISSIndex
from src.indexing.lsh_index import LSHIndex

BIRD_DEV_DATABASES = [
    "california_schools",
    "card_games",
    "codebase_community",
    "debit_card_specializing",
    "european_football_2",
    "financial",
    "formula_1",
    "student_club",
    "superhero",
    "thrombosis_prediction",
    "toxicology",
]


# ---------------------------------------------------------------------------
# Sampling: 3 per DB (1 simple, 1 moderate, 1 challenging)
# ---------------------------------------------------------------------------

def sample_questions(all_entries: list[BirdEntry]) -> list[BirdEntry]:
    """Return 33 questions: 3 per DB (1s, 1m, 1c) using random.seed(42)."""
    random.seed(42)
    by_db_diff: dict[tuple[str, str], list[BirdEntry]] = defaultdict(list)
    for e in all_entries:
        by_db_diff[(e.db_id, e.difficulty)].append(e)

    selected: list[BirdEntry] = []
    for db_id in BIRD_DEV_DATABASES:
        for difficulty in ("simple", "moderate", "challenging"):
            pool = by_db_diff.get((db_id, difficulty), [])
            if pool:
                selected.append(random.choice(pool))
            else:
                # Fallback: pick any question from this DB
                any_pool = [e for e in all_entries if e.db_id == db_id]
                if any_pool:
                    selected.append(random.choice(any_pool))

    return selected


# ---------------------------------------------------------------------------
# SQL execution helper
# ---------------------------------------------------------------------------

def execute_sql(db_path: Path, sql: str, timeout: int = 30):
    """Execute SQL against SQLite DB. Returns (ok, rows, error)."""
    try:
        conn = sqlite3.connect(str(db_path), timeout=timeout)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql)
        rows = [tuple(r) for r in cur.fetchall()]
        conn.close()
        return True, rows, None
    except Exception as e:
        try:
            conn.close()
        except Exception:
            pass
        return False, [], str(e)


def normalise_result(rows: list[tuple]) -> list[tuple]:
    """Sort rows; convert all values to strings for robustness."""
    return sorted(tuple(str(v) if v is not None else "None" for v in row) for row in rows)


def result_sets_match(a: list[tuple], b: list[tuple]) -> bool:
    return normalise_result(a) == normalise_result(b)


# ---------------------------------------------------------------------------
# Artifact loading helpers
# ---------------------------------------------------------------------------

def load_available_fields(db_id: str) -> list[tuple[str, str, str, str]]:
    summary_path = Path(settings.preprocessed_dir) / "summaries" / f"{db_id}.json"
    with open(summary_path) as f:
        data = json.load(f)
    return [
        (fs["table_name"], fs["column_name"], fs["short_summary"], fs["long_summary"])
        for fs in data["field_summaries"]
    ]


def load_full_schemas(db_id: str) -> tuple[str, str]:
    schemas_dir = Path(settings.preprocessed_dir) / "schemas"
    ddl = (schemas_dir / f"{db_id}_ddl.sql").read_text()
    markdown = (schemas_dir / f"{db_id}_markdown.md").read_text()
    return ddl, markdown


def load_gt_sql(bird_data_dir: Path) -> list[str]:
    gt_sql_path = bird_data_dir / "dev" / "dev.sql"
    with open(gt_sql_path) as f:
        return [line.split("\t")[0].strip() for line in f.readlines()]


# ---------------------------------------------------------------------------
# Per-question pipeline
# ---------------------------------------------------------------------------

async def run_one_question(
    entry: BirdEntry,
    gt_sql: str,
    lsh: LSHIndex,
    faiss_idx: FAISSIndex,
    ex_store: ExampleStore,
    available_fields: list[tuple[str, str, str, str]],
    full_ddl: str,
    full_markdown: str,
    db_path: Path,
    q_num: int,
    total: int,
) -> dict:
    print(f"\n[Q{q_num}/{total}] [{entry.db_id}] [{entry.difficulty}] {entry.question[:70]}...")
    t_start = time.time()

    result: dict = {
        "question_id": entry.question_id,
        "db_id": entry.db_id,
        "difficulty": entry.difficulty,
        "question": entry.question,
        "evidence": entry.evidence,
        "ground_truth_sql": gt_sql,
        "s1_fields": [],
        "s2_fields": [],
        "candidates": [],
        "grounding_ok": False,
        "schema_linking_ok": False,
        "error": None,
        "duration_seconds": 0.0,
    }

    # Op 5: Context Grounding
    try:
        grounding = await ground_context(
            question=entry.question,
            evidence=entry.evidence,
            db_id=entry.db_id,
            lsh_index=lsh,
            example_store=ex_store,
        )
        result["grounding_ok"] = True
        result["matched_cells"] = [
            {"table": cm.table, "column": cm.column, "value": cm.matched_value, "sim": cm.similarity_score}
            for cm in grounding.matched_cells
        ]
        result["schema_hints"] = grounding.schema_hints
        result["few_shot_count"] = len(grounding.few_shot_examples)
        print(
            f"  [Op5] cells={len(grounding.matched_cells)}, hints={grounding.schema_hints}, "
            f"few_shot={len(grounding.few_shot_examples)}"
        )
    except Exception as e:
        print(f"  [Op5] FAILED: {e}")
        result["error"] = f"grounding: {e}"
        result["duration_seconds"] = time.time() - t_start
        return result

    # Op 6: Schema Linking
    try:
        schemas = await __import__("src.schema_linking.schema_linker", fromlist=["link_schema"]).link_schema(
            question=entry.question,
            evidence=entry.evidence,
            grounding_context=grounding,
            faiss_index=faiss_idx,
            full_ddl=full_ddl,
            full_markdown=full_markdown,
            available_fields=available_fields,
        )
        result["schema_linking_ok"] = True
        result["s1_fields"] = list(schemas.s1_fields)
        result["s2_fields"] = list(schemas.s2_fields)
        result["selection_reasoning"] = (schemas.selection_reasoning or "")[:500]
        print(f"  [Op6] S1={len(schemas.s1_fields)} fields, S2={len(schemas.s2_fields)} fields")
    except Exception as e:
        print(f"  [Op6] FAILED: {e}")
        result["error"] = f"schema_linking: {e}"
        result["duration_seconds"] = time.time() - t_start
        return result

    # Op 7: All 3 generators concurrently
    reasoning_gen = ReasoningGenerator()
    standard_gen = StandardAndComplexGenerator()
    icl_gen = ICLGenerator()

    try:
        gen_results = await asyncio.gather(
            reasoning_gen.generate(entry.question, entry.evidence, schemas, grounding),
            standard_gen.generate(entry.question, entry.evidence, schemas, grounding),
            icl_gen.generate(entry.question, entry.evidence, schemas, grounding),
            return_exceptions=True,
        )
    except Exception as e:
        print(f"  [Op7] Generators FAILED: {e}")
        result["error"] = f"generators: {e}"
        result["duration_seconds"] = time.time() - t_start
        return result

    all_candidates: list[dict] = []
    gen_labels = ["A_reasoning", "B_standard_complex", "C_icl"]
    for label, gen_result in zip(gen_labels, gen_results):
        if isinstance(gen_result, Exception):
            print(f"  [Op7] {label} EXCEPTION: {gen_result}")
            all_candidates.append({
                "generator_id": f"error_{label}",
                "schema_used": "unknown",
                "sql": "",
                "error_flag": True,
                "generator_error": str(gen_result),
            })
        else:
            for cand in gen_result:
                all_candidates.append({
                    "generator_id": cand.generator_id,
                    "schema_used": cand.schema_used,
                    "schema_format": cand.schema_format,
                    "sql": cand.sql,
                    "error_flag": cand.error_flag,
                    "generator_error": None,
                })

    # Execute ground truth
    gt_ok, gt_rows, gt_err = execute_sql(db_path, gt_sql)
    result["gt_executed_ok"] = gt_ok
    result["gt_row_count"] = len(gt_rows) if gt_ok else 0
    if not gt_ok:
        print(f"  WARNING: GT SQL failed: {gt_err}")

    # Execute each candidate
    oracle_match = False
    unique_sqls: set[str] = set()
    for cand in all_candidates:
        sql = cand["sql"]
        if not sql or cand["error_flag"]:
            cand["executed_ok"] = False
            cand["row_count"] = 0
            cand["exec_error"] = "error_flag_or_empty"
            cand["oracle_match"] = False
            continue

        unique_sqls.add(sql.strip().lower())
        ok, rows, err = execute_sql(db_path, sql)
        cand["executed_ok"] = ok
        cand["row_count"] = len(rows) if ok else 0
        cand["exec_error"] = err

        if ok and gt_ok and result_sets_match(rows, gt_rows):
            cand["oracle_match"] = True
            oracle_match = True
        else:
            cand["oracle_match"] = False

    result["candidates"] = all_candidates
    result["oracle_match"] = oracle_match
    result["unique_sql_count"] = len(unique_sqls)
    result["duration_seconds"] = time.time() - t_start

    exec_ok_count = sum(1 for c in all_candidates if c.get("executed_ok"))
    oracle_match_count = sum(1 for c in all_candidates if c.get("oracle_match"))
    print(
        f"  [Op7] {len(all_candidates)} candidates, exec_ok={exec_ok_count}, "
        f"oracle_match={oracle_match_count}, unique_sql={len(unique_sqls)}, "
        f"ORACLE={'YES' if oracle_match else 'NO'}"
    )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("=" * 72)
    print("  CHECKPOINT D — Generation Pipeline (33 BIRD Dev Questions)")
    print(f"  Provider: {settings.llm_provider}  |  Cache: {settings.cache_llm_responses}")
    print("=" * 72)

    output_dir = _ROOT / "checkpoint_D_review"
    output_dir.mkdir(exist_ok=True)

    # Load BIRD dev
    bird_data_dir = Path(settings.bird_data_dir)
    dev_entries = load_bird_split("dev", bird_data_dir)
    questions = sample_questions(dev_entries)

    print(f"\nSampled {len(questions)} questions:")
    for q in questions:
        print(f"  QID={q.question_id}  DB={q.db_id}  difficulty={q.difficulty}")

    gt_sqls = load_gt_sql(bird_data_dir)
    print(f"\nLoaded {len(gt_sqls)} ground truth SQL statements")

    indices_dir = Path(settings.preprocessed_dir) / "indices"

    # Pre-load artifacts per database
    db_artifacts: dict[str, dict] = {}
    dbs_needed = sorted(set(q.db_id for q in questions))
    print(f"\nPre-loading artifacts for {len(dbs_needed)} databases...")

    for db_id in dbs_needed:
        t0 = time.time()
        print(f"  Loading {db_id}...", end=" ", flush=True)
        lsh = LSHIndex.load(str(indices_dir / f"{db_id}_lsh.pkl"))
        faiss_idx = FAISSIndex.load(
            str(indices_dir / f"{db_id}_faiss.index"),
            str(indices_dir / f"{db_id}_faiss_fields.json"),
        )
        available_fields = load_available_fields(db_id)
        full_ddl, full_markdown = load_full_schemas(db_id)
        db_path = (
            bird_data_dir / "dev" / "dev_databases" / db_id / f"{db_id}.sqlite"
        )
        db_artifacts[db_id] = {
            "lsh": lsh,
            "faiss_idx": faiss_idx,
            "available_fields": available_fields,
            "full_ddl": full_ddl,
            "full_markdown": full_markdown,
            "db_path": db_path,
        }
        print(f"done in {time.time()-t0:.1f}s ({len(lsh._minhashes):,} LSH entries, {faiss_idx._index.ntotal} FAISS fields)")

    # Load shared example store
    print("Loading example store...", end=" ", flush=True)
    ex_store = ExampleStore.load(
        str(indices_dir / "example_store.faiss"),
        str(indices_dir / "example_store_metadata.json"),
    )
    print(f"done ({len(ex_store._metadata)} entries)")

    # Process questions
    all_results: list[dict] = []
    total_start = time.time()

    for i, entry in enumerate(questions):
        gt_sql = gt_sqls[entry.question_id]
        art = db_artifacts[entry.db_id]
        try:
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
                total=len(questions),
            )
            all_results.append(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results.append({
                "question_id": entry.question_id,
                "db_id": entry.db_id,
                "difficulty": entry.difficulty,
                "question": entry.question,
                "error": f"fatal: {e}",
                "oracle_match": False,
                "candidates": [],
                "duration_seconds": 0.0,
            })

    total_elapsed = time.time() - total_start

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  SUMMARY STATISTICS")
    print("=" * 72)

    n_q = len(all_results)
    n_oracle = sum(1 for r in all_results if r.get("oracle_match", False))
    oracle_pct = 100 * n_oracle / n_q if n_q > 0 else 0

    print(f"\nOracle Upper Bound: {n_oracle}/{n_q} = {oracle_pct:.1f}%")

    # Per-difficulty oracle
    for diff in ("simple", "moderate", "challenging"):
        diff_results = [r for r in all_results if r.get("difficulty") == diff]
        n_d = len(diff_results)
        n_do = sum(1 for r in diff_results if r.get("oracle_match"))
        pct = 100 * n_do / n_d if n_d else 0
        print(f"  {diff:12s}: {n_do}/{n_d} = {pct:.1f}%")

    # Per-database oracle
    print("\nPer-Database Oracle:")
    for db_id in BIRD_DEV_DATABASES:
        db_results = [r for r in all_results if r.get("db_id") == db_id]
        n_d = len(db_results)
        n_do = sum(1 for r in db_results if r.get("oracle_match"))
        pct = 100 * n_do / n_d if n_d else 0
        print(f"  {db_id:30s}: {n_do}/{n_d} = {pct:.1f}%")

    # Per-generator success rate
    gen_stats: dict[str, dict] = {}
    all_sql_strings: list[str] = []

    for result in all_results:
        for cand in result.get("candidates", []):
            gen_id = cand.get("generator_id", "unknown")
            if gen_id.startswith("reasoning"):
                gen_type = "A_reasoning"
            elif gen_id.startswith("standard_B1"):
                gen_type = "B1_standard"
            elif gen_id.startswith("complex_B2"):
                gen_type = "B2_complex"
            elif gen_id.startswith("icl_C"):
                gen_type = "C_icl"
            else:
                gen_type = "unknown"

            if gen_type not in gen_stats:
                gen_stats[gen_type] = {"total": 0, "non_empty": 0, "exec_ok": 0, "oracle_match": 0}

            gen_stats[gen_type]["total"] += 1
            if cand.get("sql") and not cand.get("error_flag"):
                gen_stats[gen_type]["non_empty"] += 1
                all_sql_strings.append(cand["sql"])
            if cand.get("executed_ok"):
                gen_stats[gen_type]["exec_ok"] += 1
            if cand.get("oracle_match"):
                gen_stats[gen_type]["oracle_match"] += 1

    print("\nPer-Generator Success Rate:")
    print(f"  {'Generator':<20} {'Total':>6} {'Non-empty':>10} {'Exec OK':>8} {'Oracle':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*8} {'-'*8}")
    for gen_type, stats in sorted(gen_stats.items()):
        t = stats["total"]
        ne = stats["non_empty"]
        eo = stats["exec_ok"]
        om = stats["oracle_match"]
        print(
            f"  {gen_type:<20} {t:>6}  {ne:>4} ({100*ne//t if t else 0:>2}%)  "
            f"{eo:>4} ({100*eo//t if t else 0:>2}%)  {om:>4} ({100*om//t if t else 0:>2}%)"
        )

    # Diversity stats
    total_cands = len(all_sql_strings)
    unique_cands = len(set(all_sql_strings))
    dupe_count = total_cands - unique_cands

    print(f"\nCandidate Diversity:")
    print(f"  Total non-error candidates: {total_cands}")
    print(f"  Unique SQL strings:         {unique_cands}")
    print(f"  Duplicates:                 {dupe_count} ({100*dupe_count//total_cands if total_cands else 0}%)")
    print(f"  Avg candidates/question:    {total_cands/n_q:.1f}")

    # Per-question table
    print("\nPer-Question Results:")
    print(f"  {'QID':>4}  {'DB':20}  {'Diff':10}  {'Cands':>5}  {'ExecOK':>6}  {'OracleMatch':>12}  {'Error'}")
    print(f"  {'-'*4}  {'-'*20}  {'-'*10}  {'-'*5}  {'-'*6}  {'-'*12}  {'-'*30}")
    for r in all_results:
        cands = r.get("candidates", [])
        n_eo = sum(1 for c in cands if c.get("executed_ok"))
        oracle_str = "YES" if r.get("oracle_match") else "NO "
        err_str = (r.get("error") or "")[:40]
        print(
            f"  {r['question_id']:>4}  {r['db_id'][:20]:20}  "
            f"{r.get('difficulty','?'):10}  {len(cands):>5}  {n_eo:>6}  {oracle_str:>12}  {err_str}"
        )

    print(f"\nTotal elapsed: {total_elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # Save results JSON
    # -----------------------------------------------------------------------
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "run_config": {
                    "provider": settings.llm_provider,
                    "model_fast": settings.model_fast,
                    "model_powerful": settings.model_powerful,
                    "model_reasoning": settings.model_reasoning,
                    "cache_enabled": settings.cache_llm_responses,
                    "n_questions": n_q,
                    "sampling": "stratified 3-per-DB (1s/1m/1c), random.seed(42)",
                    "total_elapsed_seconds": total_elapsed,
                },
                "summary": {
                    "oracle_upper_bound": n_oracle / n_q if n_q else 0,
                    "oracle_count": n_oracle,
                    "question_count": n_q,
                    "total_candidates": total_cands,
                    "unique_candidates": unique_cands,
                    "duplicate_candidates": dupe_count,
                    "gen_stats": gen_stats,
                },
                "questions": all_results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nDetailed results saved to: {results_path}")
    print("=" * 72)
    print("  CHECKPOINT D COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
