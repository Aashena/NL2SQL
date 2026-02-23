#!/usr/bin/env python3
"""
Checkpoint D: Run pipeline on first 10 BIRD dev questions, analyse candidates.

Runs: Op 5 (Context Grounding) → Op 6 (Schema Linking) → Op 7 (all 3 generators)
Produces: checkpoint_D_review/results.json + summary stats printed to stdout.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.config.settings import settings
from src.data.bird_loader import load_bird_split
from src.generation.icl_generator import ICLGenerator
from src.generation.reasoning_generator import ReasoningGenerator
from src.generation.standard_generator import StandardAndComplexGenerator
from src.grounding.context_grounder import ground_context
from src.indexing.example_store import ExampleStore
from src.indexing.faiss_index import FAISSIndex
from src.indexing.lsh_index import LSHIndex
from src.schema_linking.schema_linker import link_schema


# ---------------------------------------------------------------------------
# SQL execution helper (sqlite3 direct)
# ---------------------------------------------------------------------------

def execute_sql(db_path: Path, sql: str, timeout: int = 30):
    """Execute SQL against the given SQLite DB. Returns (ok, rows, error)."""
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
    """Sort rows for comparison; convert all values to strings for robustness."""
    normalised = []
    for row in rows:
        normalised.append(tuple(str(v) if v is not None else "None" for v in row))
    return sorted(normalised)


def result_sets_match(rows_a: list[tuple], rows_b: list[tuple]) -> bool:
    """Return True if two result sets are equal after normalisation."""
    return normalise_result(rows_a) == normalise_result(rows_b)


# ---------------------------------------------------------------------------
# Artifact loading helpers (same patterns as checkpoint_c)
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


def hr(char="=", width=72):
    print(char * width)


# ---------------------------------------------------------------------------
# Per-question pipeline
# ---------------------------------------------------------------------------

async def run_one_question(
    entry,
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
    """Run the full pipeline for one question. Returns a result dict."""
    print(f"\n[Q{q_num}/{total}] {entry.question[:80]}...")
    t_start = time.time()

    result = {
        "question_id": entry.question_id,
        "db_id": entry.db_id,
        "question": entry.question,
        "evidence": entry.evidence,
        "difficulty": entry.difficulty,
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
    print(f"  [Op5] Grounding...")
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
        print(f"    matched_cells={len(grounding.matched_cells)}, hints={grounding.schema_hints}, few_shot={len(grounding.few_shot_examples)}")
    except Exception as e:
        print(f"    FAILED: {e}")
        result["error"] = f"grounding: {e}"
        result["duration_seconds"] = time.time() - t_start
        return result

    # Op 6: Schema Linking
    print(f"  [Op6] Schema linking...")
    try:
        schemas = await link_schema(
            question=entry.question,
            evidence=entry.evidence,
            grounding_context=grounding,
            faiss_index=faiss_idx,
            full_ddl=full_ddl,
            full_markdown=full_markdown,
            available_fields=available_fields,
        )
        result["schema_linking_ok"] = True
        result["s1_fields"] = schemas.s1_fields
        result["s2_fields"] = schemas.s2_fields
        result["selection_reasoning"] = schemas.selection_reasoning[:500] if schemas.selection_reasoning else ""
        print(f"    S1={len(schemas.s1_fields)} fields, S2={len(schemas.s2_fields)} fields")
    except Exception as e:
        print(f"    FAILED: {e}")
        result["error"] = f"schema_linking: {e}"
        result["duration_seconds"] = time.time() - t_start
        return result

    # Op 7: All 3 generators concurrently
    print(f"  [Op7] Running generators (A, B, C) concurrently...")
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
        print(f"    Generators failed: {e}")
        result["error"] = f"generators: {e}"
        result["duration_seconds"] = time.time() - t_start
        return result

    # Collect all candidates, note errors
    all_candidates = []
    gen_labels = ["A (Reasoning)", "B (Standard+Complex)", "C (ICL)"]
    for label, gen_result in zip(gen_labels, gen_results):
        if isinstance(gen_result, Exception):
            print(f"    {label} RAISED EXCEPTION: {gen_result}")
            # Create error candidates to represent the failure
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

    total_cands = len(all_candidates)
    error_count = sum(1 for c in all_candidates if c["error_flag"])
    print(f"    {total_cands} candidates collected ({error_count} with error_flag)")

    # Execute ground truth SQL
    gt_ok, gt_rows, gt_err = execute_sql(db_path, gt_sql)
    result["gt_executed_ok"] = gt_ok
    result["gt_row_count"] = len(gt_rows) if gt_ok else 0
    result["gt_error"] = gt_err

    if not gt_ok:
        print(f"    WARNING: Ground truth SQL failed to execute: {gt_err}")

    # Execute each candidate SQL and check oracle match
    oracle_match = False
    for cand in all_candidates:
        sql = cand["sql"]
        if not sql or cand["error_flag"]:
            cand["executed_ok"] = False
            cand["row_count"] = 0
            cand["exec_error"] = "generator_error_flag" if cand["error_flag"] else "empty_sql"
            cand["oracle_match"] = False
            continue

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
    result["duration_seconds"] = time.time() - t_start

    match_count = sum(1 for c in all_candidates if c.get("oracle_match"))
    exec_ok_count = sum(1 for c in all_candidates if c.get("executed_ok"))
    print(f"    exec_ok={exec_ok_count}/{total_cands}, oracle_match={match_count}/{total_cands}, ORACLE={'YES' if oracle_match else 'NO'}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    hr()
    print("  CHECKPOINT D — Generation Pipeline Analysis (10 BIRD Dev Questions)")
    print(f"  Provider: {settings.llm_provider}  |  Cache: {settings.cache_llm_responses}")
    hr()

    output_dir = _ROOT / "checkpoint_D_review"
    output_dir.mkdir(exist_ok=True)

    # Load first 10 BIRD dev questions
    print("\nLoading BIRD dev questions...")
    dev_entries = load_bird_split("dev", Path(settings.bird_data_dir))
    questions = dev_entries[:10]
    print(f"  Loaded {len(questions)} questions (all db_id: {questions[0].db_id})")

    # Load ground truth SQL from dev.sql
    gt_sql_path = Path(settings.bird_data_dir) / "dev" / "dev.sql"
    with open(gt_sql_path) as f:
        gt_lines = f.readlines()
    # Each line: SQL\tdb_id
    gt_sqls = [line.split("\t")[0].strip() for line in gt_lines]
    print(f"  Loaded {len(gt_sqls)} ground truth SQL statements")

    # Load artifacts for california_schools (all 10 questions use this DB)
    db_id = "california_schools"
    indices_dir = Path(settings.preprocessed_dir) / "indices"
    db_path = (
        Path(settings.bird_data_dir) / "dev" / "dev_databases" / db_id / f"{db_id}.sqlite"
    )

    print(f"\nLoading LSH index for {db_id} (may take ~10-30s)...")
    t0 = time.time()
    lsh = LSHIndex.load(str(indices_dir / f"{db_id}_lsh.pkl"))
    print(f"  LSH loaded in {time.time()-t0:.1f}s: {len(lsh._minhashes):,} entries")

    print(f"Loading FAISS index for {db_id}...")
    faiss_idx = FAISSIndex.load(
        str(indices_dir / f"{db_id}_faiss.index"),
        str(indices_dir / f"{db_id}_faiss_fields.json"),
    )
    print(f"  FAISS loaded: {faiss_idx._index.ntotal} fields")

    print("Loading example store...")
    ex_store = ExampleStore.load(
        str(indices_dir / "example_store.faiss"),
        str(indices_dir / "example_store_metadata.json"),
    )
    print(f"  Example store: {len(ex_store._metadata)} entries")

    # Load schema artifacts
    available_fields = load_available_fields(db_id)
    full_ddl, full_markdown = load_full_schemas(db_id)
    print(f"  Available fields: {len(available_fields)}")

    # Process each question
    all_results = []
    total_start = time.time()

    for i, entry in enumerate(questions):
        gt_sql = gt_sqls[entry.question_id]
        try:
            result = await run_one_question(
                entry=entry,
                gt_sql=gt_sql,
                lsh=lsh,
                faiss_idx=faiss_idx,
                ex_store=ex_store,
                available_fields=available_fields,
                full_ddl=full_ddl,
                full_markdown=full_markdown,
                db_path=db_path,
                q_num=i + 1,
                total=len(questions),
            )
            all_results.append(result)
        except Exception as e:
            print(f"  FATAL ERROR for Q{i+1}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "question_id": entry.question_id,
                "db_id": entry.db_id,
                "question": entry.question,
                "error": f"fatal: {e}",
                "oracle_match": False,
                "candidates": [],
            })

    total_elapsed = time.time() - total_start

    # ---------------------------------------------------------------------------
    # Compute summary statistics
    # ---------------------------------------------------------------------------

    hr()
    print("  SUMMARY STATISTICS")
    hr()

    n_questions = len(all_results)
    n_oracle_match = sum(1 for r in all_results if r.get("oracle_match", False))
    oracle_pct = 100 * n_oracle_match / n_questions if n_questions > 0 else 0

    print(f"\nOracle Upper Bound: {n_oracle_match}/{n_questions} = {oracle_pct:.1f}%")

    # Per-generator success rate
    gen_stats: dict[str, dict] = {}
    all_sql_strings: list[str] = []

    for result in all_results:
        for cand in result.get("candidates", []):
            gen_id = cand.get("generator_id", "unknown")
            # Categorize generator
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
                gen_stats[gen_type] = {"total": 0, "executed_ok": 0, "non_empty": 0}

            gen_stats[gen_type]["total"] += 1
            if not cand.get("error_flag") and cand.get("sql"):
                gen_stats[gen_type]["non_empty"] += 1
            if cand.get("executed_ok"):
                gen_stats[gen_type]["executed_ok"] += 1

            if cand.get("sql") and not cand.get("error_flag"):
                all_sql_strings.append(cand["sql"])

    print("\nPer-Generator Success Rate:")
    for gen_type, stats in sorted(gen_stats.items()):
        non_empty_pct = 100 * stats["non_empty"] / stats["total"] if stats["total"] > 0 else 0
        exec_pct = 100 * stats["executed_ok"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {gen_type:<20} total={stats['total']:3d}  non_empty={stats['non_empty']:3d} ({non_empty_pct:.0f}%)  exec_ok={stats['executed_ok']:3d} ({exec_pct:.0f}%)")

    # Candidate diversity
    total_candidates = len(all_sql_strings)
    unique_candidates = len(set(all_sql_strings))
    duplicate_count = total_candidates - unique_candidates
    avg_candidates = total_candidates / n_questions if n_questions > 0 else 0

    print(f"\nCandidate Diversity:")
    print(f"  Total candidates (non-error): {total_candidates}")
    print(f"  Unique SQL strings:           {unique_candidates}")
    print(f"  Duplicate SQLs:               {duplicate_count}")
    print(f"  Avg candidates per question:  {avg_candidates:.1f}")

    # Per-question table
    print("\nPer-Question Results:")
    print(f"  {'QID':>4}  {'DB':25}  {'Cands':>6}  {'ExecOK':>7}  {'OracleMatch':>12}")
    print(f"  {'-'*4}  {'-'*25}  {'-'*6}  {'-'*7}  {'-'*12}")
    for result in all_results:
        cands = result.get("candidates", [])
        n_exec_ok = sum(1 for c in cands if c.get("executed_ok"))
        oracle_str = "YES" if result.get("oracle_match") else "NO"
        print(f"  {result['question_id']:>4}  {result['db_id'][:25]:25}  {len(cands):>6}  {n_exec_ok:>7}  {oracle_str:>12}")

    print(f"\nTotal elapsed: {total_elapsed:.1f}s")

    # Save results
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
                    "n_questions": n_questions,
                    "total_elapsed_seconds": total_elapsed,
                },
                "summary": {
                    "oracle_upper_bound": n_oracle_match / n_questions if n_questions else 0,
                    "oracle_count": n_oracle_match,
                    "question_count": n_questions,
                    "total_candidates": total_candidates,
                    "unique_candidates": unique_candidates,
                    "duplicate_candidates": duplicate_count,
                    "gen_stats": gen_stats,
                },
                "questions": all_results,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\nDetailed results saved to: {results_path}")
    hr()
    print("  CHECKPOINT D COMPLETE")
    hr()


if __name__ == "__main__":
    asyncio.run(main())
