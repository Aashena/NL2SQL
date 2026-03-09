#!/usr/bin/env python3
"""
66-Question Smoke Test — Comprehensive Component Analysis

Runs the full NL2SQL pipeline on 66 stratified BIRD dev questions
(6 per each of 11 databases: 2 simple, 2 moderate, 2 challenging)
with detailed per-component tracing and analysis output.

Unlike run_evaluation.py (which is optimized for throughput), this script
captures the full intermediate state at every pipeline stage for later
post-mortem analysis.

Output files (all written to --output_dir):
  results.json                — Per-question results; EvaluationEntry-compatible
                                (question_id, db_id, difficulty, correct, latency_s, …)
  detailed_traces.json        — Full per-question trace (all intermediate outputs
                                for every op, per-candidate details, oracle metrics)
  component_summary.json      — Aggregated metrics per pipeline component
                                (op5/op6/op7/op8/op8_verification/op9 statistics)
  op8_verification_report.json— Per-question Op8 detail: verification plan, per-candidate
                                iteration traces (exec + verif state at every fix iteration),
                                feedback sent to the fix LLM, and calibration metrics.
  failed_questions.json       — Only the questions that were answered incorrectly,
                                with full diagnostic information for failure analysis
  smoke_test.log              — Complete structured log (DEBUG level)

Usage:
  python scripts/run_smoke_test.py
  python scripts/run_smoke_test.py --output_dir results/my_run_20260301
  python scripts/run_smoke_test.py --workers 3
  python scripts/run_smoke_test.py --resume  # continue from partial run

The script respects the same .env settings as the rest of the project
(LLM_PROVIDER, ANTHROPIC_API_KEY, CACHE_LLM_RESPONSES, etc.)
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import ctypes.util
import gc
import json
import logging
import os
import psutil
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Suppress HuggingFace/tokenizer noise before any imports
# ---------------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Project imports (after path setup)
# ---------------------------------------------------------------------------
from src.config.settings import settings
from src.data.bird_loader import load_bird_split
from src.data.database import execute_sql
from src.grounding.context_grounder import ground_context, GroundingContext
from src.indexing.example_store import ExampleStore
from src.indexing.faiss_index import FAISSIndex
from src.indexing.lsh_index import LSHIndex
from src.schema_linking.schema_linker import link_schema, LinkedSchemas
from src.generation.reasoning_generator import ReasoningGenerator
from src.generation.standard_generator import StandardAndComplexGenerator
from src.generation.icl_generator import ICLGenerator
from src.fixing.query_fixer import QueryFixer, _categorize_error
from src.selection.adaptive_selector import AdaptiveSelector


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _malloc_trim() -> None:
    """Ask the C allocator to return freed memory pages to the OS."""
    try:
        if sys.platform == "darwin":
            libc = ctypes.CDLL("libc.dylib", use_errno=True)
            fn = libc.malloc_zone_pressure_relief
            fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            fn.restype = ctypes.c_size_t
            fn(None, 0)
        else:
            libc_name = ctypes.util.find_library("c") or "libc.so.6"
            libc = ctypes.CDLL(libc_name, use_errno=True)
            libc.malloc_trim(ctypes.c_size_t(0))
    except Exception:
        pass  # non-critical


def _mem_mib(label: str) -> float:
    """Log and return the process RSS in MiB at the given checkpoint label."""
    rss_bytes = psutil.Process().memory_info().rss
    rss_mib = rss_bytes / (1024 ** 2)
    print(f"  [MEM] {label}: {rss_mib:.1f} MiB RSS", flush=True)
    return rss_mib


# ---------------------------------------------------------------------------
# Per-stage timeouts.  Generator timeouts are read from settings so that
# GENERATOR_TIMEOUT_S in .env controls all three generators consistently.
# Non-generator stages keep fixed values sufficient for any provider.
# ---------------------------------------------------------------------------
_TIMEOUT_GROUNDING = 600
_TIMEOUT_SCHEMA_LINKING = 1800
_TIMEOUT_GEN_REASONING = settings.generator_timeout_s
_TIMEOUT_GEN_STANDARD = settings.generator_timeout_s
_TIMEOUT_GEN_ICL = settings.generator_timeout_s
_TIMEOUT_FIXING = 600
_TIMEOUT_SELECTION = 600
_TIMEOUT_TOTAL_SAFETY = (
    _TIMEOUT_GROUNDING + _TIMEOUT_SCHEMA_LINKING
    + max(_TIMEOUT_GEN_REASONING, _TIMEOUT_GEN_STANDARD, _TIMEOUT_GEN_ICL)
    + _TIMEOUT_FIXING + _TIMEOUT_SELECTION + 60
)


async def _check_mlx_server() -> bool:
    """Return True if the local MLX server (or any non-MLX provider) is reachable.

    Only meaningful when LLM_PROVIDER=mlx.  Performs a quick GET /v1/models with
    a 3-second connect timeout so it fails fast.
    """
    if settings.llm_provider != "mlx":
        return True
    import httpx as _httpx  # already installed as a dependency of openai
    url = f"{settings.mlx_server_url.rstrip('/')}/v1/models"
    try:
        async with _httpx.AsyncClient(timeout=3.0) as client:
            await client.get(url)
        return True
    except Exception:
        return False


# BIRD dev dataset directories
_DEV_DATABASES = _REPO_ROOT / "data" / "bird" / "dev" / "dev_databases"
_PREPROCESSED = _REPO_ROOT / "data" / "preprocessed"
_INDICES_DIR = _PREPROCESSED / "indices"
_SCHEMAS_DIR = _PREPROCESSED / "schemas"


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _setup_logging(output_dir: Path, level: str = "INFO") -> None:
    """Configure root logger to write to both console and a file."""
    log_file = output_dir / "smoke_test.log"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), encoding="utf-8"),
    ]
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=handlers,
        force=True,
    )


logger = logging.getLogger("smoke_test")


# ---------------------------------------------------------------------------
# Stratified sampling: 66 questions, 2 per difficulty per DB
# ---------------------------------------------------------------------------

def stratified_sample_66(entries) -> list:
    """
    Select 66 BIRD dev questions: 6 per each of the 11 databases,
    2 per difficulty level (simple, moderate, challenging).
    Uses random.seed(42) for reproducibility.
    """
    random.seed(42)

    # Group by (db_id, difficulty)
    groups: dict[tuple, list] = {}
    for entry in entries:
        key = (entry.db_id, entry.difficulty)
        groups.setdefault(key, []).append(entry)

    db_ids = sorted({entry.db_id for entry in entries})
    difficulties = ["simple", "moderate", "challenging"]

    selected = []
    for db_id in db_ids:
        for diff in difficulties:
            pool = groups.get((db_id, diff), [])
            if len(pool) >= 2:
                selected.extend(random.sample(pool, 2))
            elif len(pool) == 1:
                logger.warning(
                    "Only 1 %s question for db_id=%s — using it twice", diff, db_id
                )
                selected.extend(pool * 2)
            else:
                logger.warning("No %s questions found for db_id=%s", diff, db_id)

    return selected


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def load_artifacts(db_id: str) -> dict:
    """
    Load all pre-built offline artifacts for a single database.
    Returns a dict with: lsh_index, faiss_index, full_ddl, full_markdown,
    available_fields, db_path.
    """
    lsh_path = _INDICES_DIR / f"{db_id}_lsh.pkl"
    faiss_index_path = _INDICES_DIR / f"{db_id}_faiss.index"
    faiss_fields_path = _INDICES_DIR / f"{db_id}_faiss_fields.json"
    ddl_path = _SCHEMAS_DIR / f"{db_id}_ddl.sql"
    md_path = _SCHEMAS_DIR / f"{db_id}_markdown.md"
    db_path = _DEV_DATABASES / db_id / f"{db_id}.sqlite"

    lsh_index = LSHIndex.load(str(lsh_path))
    faiss_index = FAISSIndex.load(str(faiss_index_path), str(faiss_fields_path))

    full_ddl = ddl_path.read_text(encoding="utf-8")
    full_markdown = md_path.read_text(encoding="utf-8")

    available_fields = [
        (f["table"], f["column"], f.get("short_summary", ""), f.get("long_summary", ""))
        for f in faiss_index._fields
    ]

    return {
        "lsh_index": lsh_index,
        "faiss_index": faiss_index,
        "full_ddl": full_ddl,
        "full_markdown": full_markdown,
        "available_fields": available_fields,
        "db_path": str(db_path),
    }


# ---------------------------------------------------------------------------
# Result comparison helpers
# ---------------------------------------------------------------------------

def _normalize_val(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f"{v:.6f}"
    return str(v)


def _normalize_rows(rows: list) -> list:
    return sorted(
        tuple(_normalize_val(cell) for cell in row)
        for row in rows
    )


def results_match(rows_a: list, rows_b: list) -> bool:
    if not rows_a and not rows_b:
        return True
    return _normalize_rows(rows_a) == _normalize_rows(rows_b)


# ---------------------------------------------------------------------------
# Schema recall helpers (from checkpoint_e_test.py)
# ---------------------------------------------------------------------------

_ALIAS_RE = re.compile(r"^[a-z]\d*$", re.IGNORECASE)


def _get_real_table_names(db_path: str) -> set[str]:
    """Return lowercased real table names from sqlite_master. Returns empty set on error."""
    try:
        import sqlite3
        con = sqlite3.connect(db_path)
        cur = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        names = {row[0].lower() for row in cur.fetchall()}
        con.close()
        return names
    except Exception:
        return set()  # fail open — don't filter if DB is unavailable


def _extract_required_tables_columns(
    gold_sql: str, real_table_names: set[str] | None = None
) -> tuple[set, set]:
    tables: set[str] = set()
    for m in re.finditer(r"(?:FROM|JOIN)\s+(\w+)", gold_sql, re.IGNORECASE):
        name = m.group(1).lower()
        if name not in ("select", "lateral", "as"):
            tables.add(name)

    table_cols: set[tuple[str, str]] = set()
    for m in re.finditer(r"(\w+)\.(\w+)", gold_sql):
        t_name = m.group(1).lower()
        c_name = m.group(2).lower()
        if t_name.isdigit() or c_name.isdigit():
            continue
        table_cols.add((t_name, c_name))
        if not _ALIAS_RE.match(t_name):
            tables.add(t_name)

    # Filter out CTE aliases and other non-real table names (e.g. MaxBanned, MB)
    if real_table_names is not None:
        tables = {t for t in tables if t in real_table_names}
        table_cols = {(t, c) for t, c in table_cols if t in real_table_names}

    return tables, table_cols


def _compute_schema_recall(s1_ddl, s2_ddl, s1_fields, s2_fields, gold_sql, db_path=None) -> dict:
    real_table_names = _get_real_table_names(db_path) if db_path else None
    required_tables, required_cols = _extract_required_tables_columns(gold_sql, real_table_names)
    s1_ddl_lower = (s1_ddl or "").lower()
    s2_ddl_lower = (s2_ddl or "").lower()

    tables_in_s1 = 0
    tables_in_s2 = 0
    missing_in_s1: list[str] = []
    missing_in_s2: list[str] = []

    for t in required_tables:
        pat_variants = [
            f"create table {t}",
            f'create table "{t}"',
            f"create table `{t}`",
            f"create table [{t}]",
        ]
        in_s1 = any(p in s1_ddl_lower for p in pat_variants)
        in_s2 = any(p in s2_ddl_lower for p in pat_variants)
        if in_s1:
            tables_in_s1 += 1
        else:
            missing_in_s1.append(t)
        if in_s2:
            tables_in_s2 += 1
        else:
            missing_in_s2.append(t)

    s1_fields_set: set[tuple] = set()
    s2_fields_set: set[tuple] = set()
    for f in (s1_fields or []):
        s1_fields_set.add((str(f[0]).lower(), str(f[1]).lower()))
    for f in (s2_fields or []):
        s2_fields_set.add((str(f[0]).lower(), str(f[1]).lower()))

    cols_in_s1 = 0
    cols_in_s2 = 0
    missing_cols_in_s1: list[str] = []
    missing_cols_in_s2: list[str] = []

    for t, c in required_cols:
        in_s1 = (t, c) in s1_fields_set or any(sc == c for _, sc in s1_fields_set)
        in_s2 = (t, c) in s2_fields_set or any(sc == c for _, sc in s2_fields_set)
        if in_s1:
            cols_in_s1 += 1
        else:
            missing_cols_in_s1.append(f"{t}.{c}")
        if in_s2:
            cols_in_s2 += 1
        else:
            missing_cols_in_s2.append(f"{t}.{c}")

    n_tables = len(required_tables)
    n_cols = len(required_cols)

    return {
        "required_tables": sorted(required_tables),
        "required_cols": [f"{t}.{c}" for t, c in sorted(required_cols)],
        "n_required_tables": n_tables,
        "n_required_cols": n_cols,
        "table_recall_s1": round(tables_in_s1 / n_tables, 3) if n_tables > 0 else 1.0,
        "table_recall_s2": round(tables_in_s2 / n_tables, 3) if n_tables > 0 else 1.0,
        "col_recall_s1": round(cols_in_s1 / n_cols, 3) if n_cols > 0 else 1.0,
        "col_recall_s2": round(cols_in_s2 / n_cols, 3) if n_cols > 0 else 1.0,
        "missing_tables_in_s1": missing_in_s1,
        "missing_tables_in_s2": missing_in_s2,
        "missing_cols_in_s1": missing_cols_in_s1,
        "missing_cols_in_s2": missing_cols_in_s2,
        "s1_complete": (tables_in_s1 == n_tables and cols_in_s1 == n_cols),
        "s2_complete": (tables_in_s2 == n_tables and cols_in_s2 == n_cols),
    }


# ---------------------------------------------------------------------------
# Generator timeout helper
# ---------------------------------------------------------------------------

async def _run_gen_with_timeout(coro, timeout_s: int, name: str) -> list:
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning("  [Gen:%s] Timed out after %ds", name, timeout_s)
        return []


# ---------------------------------------------------------------------------
# Per-question processing — two phases so indices can be freed between them
# ---------------------------------------------------------------------------

async def _ops_5_6_body(
    entry,
    artifacts: dict,
    example_store: ExampleStore,
    question_idx: int,
    total: int,
) -> tuple:
    """
    Phase 1: Run Ops 5-6 for one question (context grounding + schema linking).

    Returns (partial_trace, grounding_ctx, linked_schemas, gold_exec, t_total_start).
    After all questions in a DB batch have completed this function, the caller
    may free artifacts["lsh_index"] and artifacts["faiss_index"] — they are
    not accessed again in Phase 2.
    """
    db_id = entry.db_id
    question = entry.question
    evidence = entry.evidence or ""
    gold_sql = entry.SQL
    db_path = artifacts["db_path"]
    t_total_start = time.monotonic()
    logger.info(
        "[%d/%d] Q#%d (%s/%s): %s",
        question_idx + 1, total,
        entry.question_id, db_id, entry.difficulty,
        question[:80],
    )

    grounding_ctx: Optional[GroundingContext] = None
    linked_schemas: Optional[LinkedSchemas] = None

    # Execute gold SQL upfront (cheap local SQLite call; reused for oracle + eval in Phase 2)
    gold_exec = execute_sql(db_path, gold_sql)
    if not gold_exec.success:
        logger.warning("  [Gold] Gold SQL failed: %s", gold_exec.error)

    # Partial trace dict — completed by _ops_7_9_body
    trace: dict = {
        "question_id": entry.question_id,
        "db_id": db_id,
        "difficulty": entry.difficulty,
        "question": question,
        "evidence": evidence,
        "truth_sql": gold_sql,
        "gold_exec_success": gold_exec.success,
        "gold_row_count": len(gold_exec.rows) if gold_exec.success else None,
        # Compact result fields (filled in below)
        "predicted_sql": "",
        "correct": False,
        "selection_method": "error",
        "winner_generator": "",
        "cluster_count": 0,
        "latency_s": 0.0,
        "error_stage": None,
        # Oracle metrics
        "oracle_pre_fix": False,
        "oracle_pre_fix_count": 0,
        "oracle_post_fix": False,
        "oracle_post_fix_count": 0,
        "oracle_total_candidates": 0,
        "selector_matched_oracle": False,
        # Per-op detailed traces (filled below)
        "op5": {},
        "op6": {},
        "op7": {},
        "op8": {},
        "op9": {},
        "eval": {},
    }

    # ===================================================================
    # Op 5: Context Grounding
    # ===================================================================
    t0 = time.monotonic()
    op5: dict = {
        "duration_s": 0.0,
        "cell_matches": [],
        "cell_match_count": 0,
        "schema_hints": [],
        "schema_hint_count": 0,
        "few_shot_count": 0,
        "error": None,
    }
    try:
        grounding_ctx = await asyncio.wait_for(
            ground_context(
                question=question,
                evidence=evidence,
                db_id=db_id,
                lsh_index=artifacts["lsh_index"],
                example_store=example_store,
            ),
            timeout=_TIMEOUT_GROUNDING,
        )
        op5["cell_matches"] = [
            {
                "table": cm.table,
                "column": cm.column,
                "value": cm.matched_value,
                "similarity": round(cm.similarity_score, 4),
            }
            for cm in grounding_ctx.matched_cells
        ]
        op5["cell_match_count"] = len(grounding_ctx.matched_cells)
        op5["schema_hints"] = list(grounding_ctx.schema_hints)
        op5["schema_hint_count"] = len(grounding_ctx.schema_hints)
        op5["few_shot_count"] = len(grounding_ctx.few_shot_examples)
        logger.info(
            "  [Op5] cell_matches=%d, schema_hints=%d, few_shot=%d",
            op5["cell_match_count"], op5["schema_hint_count"], op5["few_shot_count"],
        )
    except asyncio.TimeoutError:
        op5["error"] = f"timeout after {_TIMEOUT_GROUNDING}s"
        trace["error_stage"] = "op5_timeout"
        logger.error("  [Op5] TIMEOUT after %ds", _TIMEOUT_GROUNDING)
        grounding_ctx = GroundingContext()
    except Exception as exc:
        op5["error"] = f"{type(exc).__name__}: {exc}"
        trace["error_stage"] = "op5"
        logger.error("  [Op5] FAILED: %s", exc)
        grounding_ctx = GroundingContext()
        if not await _check_mlx_server():
            logger.critical(
                "MLX server at %s is unreachable — restart it then re-run with --resume.",
                settings.mlx_server_url,
            )
            raise SystemExit(1)
    finally:
        op5["duration_s"] = round(time.monotonic() - t0, 3)
        trace["op5"] = op5

    # ===================================================================
    # Op 6: Adaptive Schema Linking
    # ===================================================================
    t0 = time.monotonic()
    op6: dict = {
        "duration_s": 0.0,
        "s1_field_count": 0,
        "s2_field_count": 0,
        "s1_fields": [],
        "s2_fields": [],
        "schema_recall": {},
        "error": None,
    }
    try:
        linked_schemas = await asyncio.wait_for(
            link_schema(
                question=question,
                evidence=evidence,
                grounding_context=grounding_ctx,
                faiss_index=artifacts["faiss_index"],
                full_ddl=artifacts["full_ddl"],
                full_markdown=artifacts["full_markdown"],
                available_fields=artifacts["available_fields"],
            ),
            timeout=_TIMEOUT_SCHEMA_LINKING,
        )
        op6["s1_field_count"] = len(linked_schemas.s1_fields)
        op6["s2_field_count"] = len(linked_schemas.s2_fields)
        op6["s1_fields"] = [
            [str(f[0]), str(f[1])] for f in (linked_schemas.s1_fields or [])
        ]
        op6["s2_fields"] = [
            [str(f[0]), str(f[1])] for f in (linked_schemas.s2_fields or [])
        ]
        op6["schema_recall"] = _compute_schema_recall(
            linked_schemas.s1_ddl,
            linked_schemas.s2_ddl,
            linked_schemas.s1_fields,
            linked_schemas.s2_fields,
            gold_sql,
            db_path=db_path,
        )
        recall = op6["schema_recall"]
        logger.info(
            "  [Op6] s1=%d fields, s2=%d fields | "
            "table_recall S1=%.0f%% S2=%.0f%% | col_recall S1=%.0f%% S2=%.0f%%",
            op6["s1_field_count"], op6["s2_field_count"],
            recall.get("table_recall_s1", 0) * 100,
            recall.get("table_recall_s2", 0) * 100,
            recall.get("col_recall_s1", 0) * 100,
            recall.get("col_recall_s2", 0) * 100,
        )
    except asyncio.TimeoutError:
        op6["error"] = f"timeout after {_TIMEOUT_SCHEMA_LINKING}s — using full DDL"
        if not trace["error_stage"]:
            trace["error_stage"] = "op6_timeout"
        logger.error("  [Op6] TIMEOUT — falling back to full DDL")
        linked_schemas = LinkedSchemas(
            s1_ddl=artifacts["full_ddl"],
            s1_markdown=artifacts["full_markdown"],
            s2_ddl=artifacts["full_ddl"],
            s2_markdown=artifacts["full_markdown"],
            s1_fields=[],
            s2_fields=[],
            selection_reasoning="",
        )
        op6["schema_recall"] = _compute_schema_recall(
            linked_schemas.s1_ddl, linked_schemas.s2_ddl,
            [], [], gold_sql, db_path=db_path,
        )
    except Exception as exc:
        op6["error"] = f"{type(exc).__name__}: {exc}"
        if not trace["error_stage"]:
            trace["error_stage"] = "op6"
        logger.error("  [Op6] FAILED: %s", exc)
        linked_schemas = LinkedSchemas(
            s1_ddl=artifacts["full_ddl"],
            s1_markdown=artifacts["full_markdown"],
            s2_ddl=artifacts["full_ddl"],
            s2_markdown=artifacts["full_markdown"],
            s1_fields=[],
            s2_fields=[],
            selection_reasoning="",
        )
        op6["schema_recall"] = _compute_schema_recall(
            linked_schemas.s1_ddl, linked_schemas.s2_ddl,
            [], [], gold_sql, db_path=db_path,
        )
        if not await _check_mlx_server():
            logger.critical(
                "MLX server at %s is unreachable — restart it then re-run with --resume.",
                settings.mlx_server_url,
            )
            raise SystemExit(1)
    finally:
        op6["duration_s"] = round(time.monotonic() - t0, 3)
        trace["op6"] = op6

    return trace, grounding_ctx, linked_schemas, gold_exec, t_total_start


def _build_error_trace(entry, error_stage: str, error_msg: str) -> dict:
    """Build a minimal error trace for a question that failed in Phase 1."""
    gold_sql = entry.SQL
    return {
        "question_id": entry.question_id,
        "db_id": entry.db_id,
        "difficulty": entry.difficulty,
        "question": entry.question,
        "evidence": entry.evidence or "",
        "truth_sql": gold_sql,
        "gold_exec_success": False,
        "gold_row_count": None,
        "predicted_sql": "",
        "correct": False,
        "selection_method": "error",
        "winner_generator": "",
        "cluster_count": 0,
        "latency_s": 0.0,
        "error_stage": error_stage,
        "oracle_pre_fix": False,
        "oracle_pre_fix_count": 0,
        "oracle_post_fix": False,
        "oracle_post_fix_count": 0,
        "oracle_total_candidates": 0,
        "selector_matched_oracle": False,
        "op5": {"error": error_msg},
        "op6": {"error": error_msg},
        "op7": {},
        "op8": {},
        "op9": {},
        "eval": {},
    }


def _build_verif_plan_entry(spec, db_path: str) -> dict:
    """Build a verif_plan entry with full spec fields and computed upper bound.

    For grain tests, executes verification_sql_upper (or verification_sql as
    fallback) against the database and records the integer result as
    upper_bound_value so it can be compared against gold_row_count later.
    """
    upper_bound_value = None
    if spec.test_type == "grain":
        sql_to_run = spec.verification_sql_upper or spec.verification_sql
        if sql_to_run:
            try:
                vr = execute_sql(db_path, sql_to_run)
                if vr.success and vr.rows and isinstance(vr.rows[0][0], (int, float)):
                    upper_bound_value = int(vr.rows[0][0])
            except Exception:
                pass  # leave as None
    return {
        "test_type": spec.test_type,
        "description": spec.description,
        "is_critical": spec.test_type in {"grain", "duplicate", "column_alignment"},
        "fix_hint": spec.fix_hint,
        "verification_sql_upper": spec.verification_sql_upper,
        "verification_sql": spec.verification_sql,
        "upper_bound_value": upper_bound_value,
        "row_count_min": spec.row_count_min,
        "row_count_max": spec.row_count_max,
        "expected_column_count": spec.expected_column_count,
        "required_sql_keywords": spec.required_sql_keywords,
        "check_columns": spec.check_columns,
    }


async def _ops_7_9_body(
    entry,
    db_path: str,
    grounding_ctx: GroundingContext,
    linked_schemas: LinkedSchemas,
    trace: dict,
    gold_exec,
    t_total_start: float,
) -> dict:
    """
    Phase 2: Run Ops 7-9 + eval for one question (generation, fixing, selection).

    Takes no artifact references — only lightweight context objects.
    Safe to call after lsh_index and faiss_index have been freed.
    Returns the completed trace dict.
    """
    db_id = entry.db_id
    question = entry.question
    evidence = entry.evidence or ""
    gold_sql = entry.SQL
    all_candidates: list = []
    fixed_candidates: list = []
    correct_post_fix_sqls: set[str] = set()

    # ===================================================================
    # Op 7: Diverse SQL Generation (3 generators in parallel)
    # ===================================================================
    t0 = time.monotonic()
    op7: dict = {
        "duration_s": 0.0,
        "total_candidates": 0,
        "gen_reasoning_count": 0,
        "gen_standard_b1_count": 0,
        "gen_complex_b2_count": 0,
        "gen_icl_count": 0,
        "error_flag_count": 0,
        "candidates": [],
        "errors": None,
    }
    try:
        reasoning_gen = ReasoningGenerator()
        standard_gen = StandardAndComplexGenerator()
        icl_gen = ICLGenerator()

        gen_results = await asyncio.gather(
            _run_gen_with_timeout(
                reasoning_gen.generate(
                    question=question, evidence=evidence,
                    schemas=linked_schemas, grounding=grounding_ctx,
                ),
                _TIMEOUT_GEN_REASONING, "reasoning",
            ),
            _run_gen_with_timeout(
                standard_gen.generate(
                    question=question, evidence=evidence,
                    schemas=linked_schemas, grounding=grounding_ctx,
                ),
                _TIMEOUT_GEN_STANDARD, "standard",
            ),
            _run_gen_with_timeout(
                icl_gen.generate(
                    question=question, evidence=evidence,
                    schemas=linked_schemas, grounding=grounding_ctx,
                ),
                _TIMEOUT_GEN_ICL, "icl",
            ),
            return_exceptions=True,
        )

        reasoning_cands = gen_results[0] if not isinstance(gen_results[0], Exception) else []
        standard_cands = gen_results[1] if not isinstance(gen_results[1], Exception) else []
        icl_cands = gen_results[2] if not isinstance(gen_results[2], Exception) else []

        gen_errors = []
        for name, res in [("reasoning", gen_results[0]), ("standard", gen_results[1]), ("icl", gen_results[2])]:
            if isinstance(res, Exception):
                gen_errors.append(f"{name}: {type(res).__name__}: {res}")

        all_candidates = list(reasoning_cands) + list(standard_cands) + list(icl_cands)

        op7["gen_reasoning_count"] = len(reasoning_cands)
        op7["gen_standard_b1_count"] = len(
            [c for c in standard_cands if c.generator_id.startswith("standard")]
        )
        op7["gen_complex_b2_count"] = len(
            [c for c in standard_cands if c.generator_id.startswith("complex")]
        )
        op7["gen_icl_count"] = len(icl_cands)
        op7["total_candidates"] = len(all_candidates)
        op7["error_flag_count"] = sum(1 for c in all_candidates if c.error_flag)
        op7["candidates"] = [
            {
                "generator_id": c.generator_id,
                "schema_used": c.schema_used,
                "schema_format": c.schema_format,
                "sql": c.sql[:500] if c.sql else "",  # truncate long SQL
                "error_flag": c.error_flag,
            }
            for c in all_candidates
        ]
        if gen_errors:
            op7["errors"] = "; ".join(gen_errors)
            if not trace["error_stage"]:
                trace["error_stage"] = "op7"

        logger.info(
            "  [Op7] total=%d (reasoning=%d, B1=%d, B2=%d, icl=%d) error_flags=%d",
            op7["total_candidates"],
            op7["gen_reasoning_count"],
            op7["gen_standard_b1_count"],
            op7["gen_complex_b2_count"],
            op7["gen_icl_count"],
            op7["error_flag_count"],
        )

        # Oracle on pre-fix candidates
        trace["oracle_total_candidates"] = len(all_candidates)
        if gold_exec.success and all_candidates:
            pre_fix_correct = 0
            for cand in all_candidates:
                if not cand.error_flag and cand.sql:
                    cand_exec = execute_sql(db_path, cand.sql)
                    if cand_exec.success and results_match(cand_exec.rows, gold_exec.rows):
                        pre_fix_correct += 1
            trace["oracle_pre_fix"] = pre_fix_correct > 0
            trace["oracle_pre_fix_count"] = pre_fix_correct
            logger.info("  [Oracle Pre-Fix] %d/%d correct", pre_fix_correct, len(all_candidates))

    except Exception as exc:
        op7["errors"] = f"{type(exc).__name__}: {exc}"
        if not trace["error_stage"]:
            trace["error_stage"] = "op7"
        logger.error("  [Op7] FAILED: %s", exc)
        logger.debug(traceback.format_exc())
        all_candidates = []
    finally:
        op7["duration_s"] = round(time.monotonic() - t0, 3)
        trace["op7"] = op7

    # ===================================================================
    # Op 8: Query Fixer
    # ===================================================================
    t0 = time.monotonic()
    op8: dict = {
        "duration_s": 0.0,
        "total_candidates": len(all_candidates),
        "needed_fix_count": 0,
        "fixed_ok_count": 0,
        "still_failing_count": 0,
        "oracle_pre_fix": trace["oracle_pre_fix"],
        "oracle_pre_fix_count": trace["oracle_pre_fix_count"],
        "oracle_post_fix": False,
        "oracle_post_fix_count": 0,
        "candidates": [],
        "error": None,
    }

    if all_candidates:
        try:
            fixer = QueryFixer(trace=True)
            fixed_candidates = await asyncio.wait_for(
                fixer.fix_candidates(
                    candidates=all_candidates,
                    question=question,
                    evidence=evidence,
                    schemas=linked_schemas,
                    db_path=db_path,
                    cell_matches=grounding_ctx.matched_cells,
                ),
                timeout=_TIMEOUT_FIXING,
            )

            needed_fix = sum(1 for fc in fixed_candidates if fc.fix_iterations > 0)
            fixed_ok = sum(
                1 for fc in fixed_candidates
                if fc.fix_iterations > 0
                and fc.execution_result.success
                and not fc.execution_result.is_empty
            )
            still_failing = sum(
                1 for fc in fixed_candidates
                if not fc.execution_result.success or fc.execution_result.is_empty
            )

            op8["needed_fix_count"] = needed_fix
            op8["fixed_ok_count"] = fixed_ok
            op8["still_failing_count"] = still_failing

            # Per-candidate details
            op8["candidates"] = [
                {
                    "generator_id": fc.generator_id,
                    "original_sql": (fc.original_sql or "")[:500],
                    "final_sql": (fc.final_sql or "")[:500],
                    "fix_iterations": fc.fix_iterations,
                    "error_type": (
                        _categorize_error(fc.execution_result.error, fc.execution_result.is_empty)
                        if fc.fix_iterations > 0 else None
                    ),
                    "confidence": round(fc.confidence_score, 4),
                    "execution_success": fc.execution_result.success,
                    "is_empty": fc.execution_result.is_empty,
                    "row_count": (
                        len(fc.execution_result.rows)
                        if fc.execution_result.success
                        else None
                    ),
                }
                for fc in fixed_candidates
            ]

            # Oracle on post-fix candidates
            if gold_exec.success:
                post_correct = 0
                for fc in fixed_candidates:
                    if fc.execution_result.success and results_match(
                        fc.execution_result.rows, gold_exec.rows
                    ):
                        post_correct += 1
                        correct_post_fix_sqls.add(fc.final_sql)
                op8["oracle_post_fix"] = post_correct > 0
                op8["oracle_post_fix_count"] = post_correct
                trace["oracle_post_fix"] = post_correct > 0
                trace["oracle_post_fix_count"] = post_correct
                logger.info(
                    "  [Op8] needed_fix=%d, fixed_ok=%d, still_failing=%d | "
                    "oracle_post=%d/%d",
                    needed_fix, fixed_ok, still_failing,
                    post_correct, len(fixed_candidates),
                )

            # ------------------------------------------------------------------
            # Build detailed Op8 verification trace (written to op8_verification_report.json)
            # ------------------------------------------------------------------
            verif_plan_specs = fixer.last_verif_specs
            candidate_details = []
            test_type_fire_counts: Counter = Counter()
            test_type_fail_counts: Counter = Counter()
            for fc in fixed_candidates:
                is_oracle_correct = (
                    gold_exec.success
                    and fc.execution_result.success
                    and results_match(fc.execution_result.rows, gold_exec.rows)
                )
                # Determine if the loop broke early (no fix on the last pre-final iteration)
                broke_early = fc.fix_iterations == 0 and (
                    not fc.iteration_trace
                    or fc.iteration_trace[0].get("verif_all_pass") is not False
                    or fc.iteration_trace[0].get("exec_success", False)
                )
                for entry in fc.iteration_trace:
                    for r in (entry.get("verif_results") or []):
                        test_type_fire_counts[r["test_type"]] += 1
                        if r["status"] == "fail":
                            test_type_fail_counts[r["test_type"]] += 1

                final_verif = fc.verification_results
                cand_detail: dict = {
                    "generator_id": fc.generator_id,
                    "is_oracle_correct": is_oracle_correct,
                    "final_fix_iterations": fc.fix_iterations,
                    "final_confidence": round(fc.confidence_score, 4),
                    "broke_early": fc.fix_iterations == 0,
                    "iterations": fc.iteration_trace,
                    "final_exec_success": fc.execution_result.success,
                    "final_exec_is_empty": fc.execution_result.is_empty,
                    "final_exec_row_count": (
                        len(fc.execution_result.rows)
                        if fc.execution_result.success else None
                    ),
                    "final_verif_all_pass": (
                        final_verif.all_pass if final_verif else None
                    ),
                    "final_verif_confidence_adjustment": (
                        round(final_verif.confidence_adjustment, 4)
                        if final_verif else None
                    ),
                    "final_verif_test_results": (
                        [
                            {
                                "test_type": r.test_type,
                                "status": r.status,
                                "actual_outcome": r.actual_outcome,
                                "is_critical": r.is_critical,
                            }
                            for r in final_verif.test_results
                        ]
                        if final_verif else None
                    ),
                }

                # ── Per-iteration oracle evaluation + transition tracking ──────
                # Requires full SQL in iteration trace (sql_full added by query_fixer.py).
                # Only meaningful when gold execution succeeded.
                oracle_per_iter: list = []
                oracle_transitions: list = []
                verif_issues_fixed_by_iter: list = []  # fail→pass test types per iter

                if gold_exec.success and fc.iteration_trace:
                    prev_iter_verif: dict = {}  # test_type → status at previous iter
                    prev_oracle = None

                    for idx, it_entry in enumerate(fc.iteration_trace):
                        sql_full = it_entry.get("sql_full") or it_entry.get("sql", "")
                        iter_oracle = None
                        if sql_full and it_entry.get("exec_success"):
                            try:
                                iter_exec = execute_sql(db_path, sql_full)
                                if iter_exec.success:
                                    iter_oracle = results_match(iter_exec.rows, gold_exec.rows)
                            except Exception:
                                iter_oracle = None
                        oracle_per_iter.append(iter_oracle)

                        # Which test types went fail→pass compared to the previous iteration?
                        curr_verif: dict = {
                            r["test_type"]: r["status"]
                            for r in (it_entry.get("verif_results") or [])
                        }
                        fixed_this_iter = [
                            tt for tt, status in curr_verif.items()
                            if status == "pass" and prev_iter_verif.get(tt) == "fail"
                        ]
                        verif_issues_fixed_by_iter.append(fixed_this_iter)

                        # Oracle transition: previous iter triggered a fix AND SQL changed
                        if (
                            idx > 0
                            and prev_oracle is not None
                            and iter_oracle is not None
                            and iter_oracle != prev_oracle
                            and fc.iteration_trace[idx - 1].get("fix_triggered")
                            and it_entry.get("sql_changed_from_previous")
                        ):
                            failing_in_prev = [
                                r["test_type"]
                                for r in (fc.iteration_trace[idx - 1].get("verif_results") or [])
                                if r["status"] == "fail"
                            ]
                            oracle_transitions.append({
                                "iteration": idx,
                                "direction": "wrong_to_correct" if iter_oracle else "correct_to_wrong",
                                "verif_issues_triggered": failing_in_prev,
                            })

                        prev_oracle = iter_oracle
                        prev_iter_verif = curr_verif

                cand_detail["oracle_per_iteration"] = oracle_per_iter
                cand_detail["oracle_transitions"] = oracle_transitions
                cand_detail["verif_issues_fixed_by_iter"] = verif_issues_fixed_by_iter
                # ─────────────────────────────────────────────────────────────

                candidate_details.append(cand_detail)

            op8_detail: dict = {
                "question_id": entry.get("question_id") if False else trace.get("question_id"),
                "question": question,
                "db_id": db_id,
                "difficulty": entry.get("difficulty") if False else trace.get("difficulty"),
                "correct": trace.get("correct", False),          # filled in later by eval
                "oracle_post_fix": op8.get("oracle_post_fix", False),
                "gold_row_count": trace.get("gold_row_count"),
                # Verification plan
                "verif_plan_count": len(verif_plan_specs),
                "verif_plan_error": None,
                "verif_plan": [
                    _build_verif_plan_entry(s, db_path)
                    for s in verif_plan_specs
                ],
                # Per-candidate
                "candidates": candidate_details,
                # Per-question aggregates
                "summary": {
                    "total_candidates": len(fixed_candidates),
                    "candidates_needing_fix": needed_fix,
                    "candidates_fixed_ok": fixed_ok,
                    "candidates_early_break": sum(
                        1 for fc in fixed_candidates if fc.fix_iterations == 0
                    ),
                    "candidates_verif_all_pass_final": sum(
                        1 for fc in fixed_candidates
                        if fc.verification_results is not None
                        and fc.verification_results.all_pass
                    ),
                    "test_type_fire_counts": dict(test_type_fire_counts),
                    "test_type_fail_counts": dict(test_type_fail_counts),
                },
            }
            trace["op8_detail"] = op8_detail

        except asyncio.TimeoutError:
            op8["error"] = f"timeout after {_TIMEOUT_FIXING}s"
            if not trace["error_stage"]:
                trace["error_stage"] = "op8_timeout"
            logger.error("  [Op8] TIMEOUT after %ds", _TIMEOUT_FIXING)
            fixed_candidates = []
        except Exception as exc:
            op8["error"] = f"{type(exc).__name__}: {exc}"
            if not trace["error_stage"]:
                trace["error_stage"] = "op8"
            logger.error("  [Op8] FAILED: %s", exc)
            logger.debug(traceback.format_exc())
            fixed_candidates = []
    else:
        op8["error"] = "skipped — no candidates from Op7"
        logger.warning("  [Op8] Skipped (no candidates)")

    op8["duration_s"] = round(time.monotonic() - t0, 3)
    trace["op8"] = op8

    # ===================================================================
    # Op 9: Adaptive Selection
    # ===================================================================
    t0 = time.monotonic()
    op9: dict = {
        "duration_s": 0.0,
        "method": "error",
        "cluster_count": 0,
        "candidates_evaluated": 0,
        "winner_sql": "",
        "winner_generator": "",
        "tournament_wins": {},
        "selector_matched_oracle": False,
        "error": None,
    }

    if fixed_candidates:
        try:
            selector = AdaptiveSelector()
            selection = await asyncio.wait_for(
                selector.select(
                    candidates=fixed_candidates,
                    question=question,
                    evidence=evidence,
                    schemas=linked_schemas,
                    db_path=db_path,
                ),
                timeout=_TIMEOUT_SELECTION,
            )

            op9["method"] = selection.selection_method
            op9["cluster_count"] = selection.cluster_count
            op9["candidates_evaluated"] = selection.candidates_evaluated
            op9["winner_sql"] = (selection.final_sql or "")[:500]
            op9["tournament_wins"] = dict(selection.tournament_wins or {})

            # Find which generator produced the winning SQL
            winner_gen = ""
            if selection.final_sql:
                for fc in fixed_candidates:
                    if fc.final_sql == selection.final_sql:
                        winner_gen = fc.generator_id
                        break
            op9["winner_generator"] = winner_gen

            # Check if selector picked an oracle-correct candidate
            if correct_post_fix_sqls and selection.final_sql:
                op9["selector_matched_oracle"] = (
                    selection.final_sql in correct_post_fix_sqls
                )
                trace["selector_matched_oracle"] = op9["selector_matched_oracle"]

            trace["predicted_sql"] = selection.final_sql or ""
            trace["selection_method"] = selection.selection_method
            trace["winner_generator"] = winner_gen
            trace["cluster_count"] = selection.cluster_count

            logger.info(
                "  [Op9] method=%s clusters=%d winner=%s oracle_match=%s",
                op9["method"], op9["cluster_count"],
                op9["winner_generator"], op9["selector_matched_oracle"],
            )

        except asyncio.TimeoutError:
            op9["error"] = f"timeout after {_TIMEOUT_SELECTION}s"
            if not trace["error_stage"]:
                trace["error_stage"] = "op9_timeout"
            logger.error("  [Op9] TIMEOUT — using best-confidence fallback")
            best = max(fixed_candidates, key=lambda fc: fc.confidence_score)
            trace["predicted_sql"] = best.final_sql or ""
            trace["selection_method"] = "fallback_timeout"
        except Exception as exc:
            op9["error"] = f"{type(exc).__name__}: {exc}"
            if not trace["error_stage"]:
                trace["error_stage"] = "op9"
            logger.error("  [Op9] FAILED: %s", exc)
            logger.debug(traceback.format_exc())
            if fixed_candidates:
                best = max(fixed_candidates, key=lambda fc: fc.confidence_score)
                trace["predicted_sql"] = best.final_sql or ""
                trace["selection_method"] = "fallback_exception"
    else:
        op9["error"] = "skipped — no fixed candidates"
        logger.warning("  [Op9] Skipped (no fixed candidates)")

    op9["duration_s"] = round(time.monotonic() - t0, 3)
    trace["op9"] = op9

    # ===================================================================
    # Evaluation: compare predicted vs gold SQL
    # ===================================================================
    t_eval_start = time.monotonic()
    final_sql = trace["predicted_sql"]
    eval_info: dict = {
        "predicted_sql": final_sql,
        "truth_sql": gold_sql,
        "correct": False,
        "pred_success": False,
        "truth_success": gold_exec.success,
        "pred_row_count": None,
        "truth_row_count": len(gold_exec.rows) if gold_exec.success else None,
    }

    if final_sql:
        try:
            pred_result = execute_sql(db_path, final_sql)
            eval_info["pred_success"] = pred_result.success
            if pred_result.success:
                eval_info["pred_row_count"] = len(pred_result.rows)
                if gold_exec.success:
                    eval_info["correct"] = results_match(
                        pred_result.rows, gold_exec.rows
                    )
        except Exception as exc:
            logger.error("  [Eval] Execution failed: %s", exc)
    else:
        logger.warning("  [Eval] No SQL to evaluate")

    trace["correct"] = eval_info["correct"]
    trace["eval"] = eval_info

    # Backfill op8_detail["correct"] now that we know the eval result
    if "op8_detail" in trace:
        trace["op8_detail"]["correct"] = eval_info["correct"]

    # Total latency
    trace["latency_s"] = round(time.monotonic() - t_total_start, 3)

    result_str = "✓ CORRECT" if trace["correct"] else "✗ WRONG"
    logger.info(
        "  → %s | oracle_pre=%s oracle_post=%s selector_match=%s | "
        "latency=%.1fs | SQL: %.60s",
        result_str,
        trace["oracle_pre_fix"], trace["oracle_post_fix"],
        trace["selector_matched_oracle"],
        trace["latency_s"],
        final_sql or "(none)",
    )

    return trace


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def compute_component_summary(traces: list[dict]) -> dict:
    """
    Aggregate per-question traces into component-level metrics.
    """
    n = len(traces)
    if n == 0:
        return {}

    n_correct = sum(1 for t in traces if t.get("correct", False))

    # ---- By difficulty ----
    by_diff: dict[str, list] = defaultdict(list)
    for t in traces:
        by_diff[t.get("difficulty", "unknown")].append(t.get("correct", False))
    by_difficulty = {d: sum(v) / len(v) for d, v in by_diff.items()}

    # ---- By database ----
    by_db_map: dict[str, list] = defaultdict(list)
    for t in traces:
        by_db_map[t.get("db_id", "unknown")].append(t.get("correct", False))
    by_db = {db: sum(v) / len(v) for db, v in by_db_map.items()}

    # ---- Op5 ----
    op5_traces = [t.get("op5", {}) for t in traces]
    op5_summary = {
        "avg_cell_match_count": _safe_avg(op5_traces, "cell_match_count"),
        "avg_schema_hint_count": _safe_avg(op5_traces, "schema_hint_count"),
        "avg_few_shot_count": _safe_avg(op5_traces, "few_shot_count"),
        "avg_duration_s": _safe_avg(op5_traces, "duration_s"),
        "error_rate": sum(1 for o in op5_traces if o.get("error")) / n,
    }

    # ---- Op6 ----
    op6_traces = [t.get("op6", {}) for t in traces]
    recalls = [t.get("op6", {}).get("schema_recall", {}) for t in traces]
    op6_summary = {
        "avg_s1_fields": _safe_avg(op6_traces, "s1_field_count"),
        "avg_s2_fields": _safe_avg(op6_traces, "s2_field_count"),
        "avg_table_recall_s1": _safe_avg(recalls, "table_recall_s1"),
        "avg_table_recall_s2": _safe_avg(recalls, "table_recall_s2"),
        "avg_col_recall_s1": _safe_avg(recalls, "col_recall_s1"),
        "avg_col_recall_s2": _safe_avg(recalls, "col_recall_s2"),
        "s1_complete_rate": sum(1 for r in recalls if r.get("s1_complete")) / n,
        "s2_complete_rate": sum(1 for r in recalls if r.get("s2_complete")) / n,
        "avg_duration_s": _safe_avg(op6_traces, "duration_s"),
        "error_rate": sum(1 for o in op6_traces if o.get("error")) / n,
    }

    # ---- Op7 ----
    op7_traces = [t.get("op7", {}) for t in traces]
    op7_summary = {
        "avg_total_candidates": _safe_avg(op7_traces, "total_candidates"),
        "avg_reasoning_candidates": _safe_avg(op7_traces, "gen_reasoning_count"),
        "avg_standard_b1_candidates": _safe_avg(op7_traces, "gen_standard_b1_count"),
        "avg_complex_b2_candidates": _safe_avg(op7_traces, "gen_complex_b2_count"),
        "avg_icl_candidates": _safe_avg(op7_traces, "gen_icl_count"),
        "avg_error_flags": _safe_avg(op7_traces, "error_flag_count"),
        "error_flag_rate": (
            sum(t.get("op7", {}).get("error_flag_count", 0) for t in traces)
            / max(sum(t.get("op7", {}).get("total_candidates", 0) for t in traces), 1)
        ),
        "avg_duration_s": _safe_avg(op7_traces, "duration_s"),
    }

    # ---- Op8 ----
    op8_traces = [t.get("op8", {}) for t in traces]
    total_candidates_op8 = sum(o.get("total_candidates", 0) for o in op8_traces)
    total_needed_fix = sum(o.get("needed_fix_count", 0) for o in op8_traces)
    total_fixed_ok = sum(o.get("fixed_ok_count", 0) for o in op8_traces)
    op8_summary = {
        "total_candidates": total_candidates_op8,
        "total_needed_fix": total_needed_fix,
        "total_fixed_ok": total_fixed_ok,
        "candidates_needing_fix_rate": (
            total_needed_fix / total_candidates_op8 if total_candidates_op8 > 0 else 0.0
        ),
        "fix_success_rate": (
            total_fixed_ok / total_needed_fix if total_needed_fix > 0 else 0.0
        ),
        "avg_duration_s": _safe_avg(op8_traces, "duration_s"),
        "error_rate": sum(1 for o in op8_traces if o.get("error")) / n,
    }

    # ---- Op8 Verification (from op8_detail, populated when trace=True) ----
    op8_details = [t["op8_detail"] for t in traces if "op8_detail" in t]
    op8_verif_summary: dict = {}
    if op8_details:
        nd = len(op8_details)

        # Accumulate test-type counters across all questions × candidates × iterations
        tt_fire: Counter = Counter()   # evaluations run per test type
        tt_fail: Counter = Counter()   # evaluations that failed per test type
        tt_fix_trigger: Counter = Counter()  # failures that triggered a fix

        total_cands = 0
        early_break_count = 0
        one_fix_count = 0
        two_fix_count = 0

        fix_changed_total = 0      # fix was triggered and produced different SQL
        fix_triggered_total = 0    # fix was triggered (regardless of SQL change)
        verif_improved_count = 0   # verif status went from fail→pass after fix
        verif_fail_oracle_correct = 0  # verif said FAIL but candidate was oracle-correct
        verif_pass_oracle_wrong = 0    # verif said PASS but candidate was oracle-wrong

        # New: verification issue fix effectiveness + oracle transition tracking
        verif_issues_fixed_counts: Counter = Counter()  # test_type → fail→pass count
        wrong_to_correct_count = 0   # fix events where candidate went wrong→correct
        correct_to_wrong_count = 0   # fix events where candidate went correct→wrong
        issues_triggering_correct_trans: Counter = Counter()  # test_type → count
        issues_triggering_wrong_trans: Counter = Counter()    # test_type → count

        for detail in op8_details:
            for cand in detail.get("candidates", []):
                total_cands += 1
                fi = cand.get("final_fix_iterations", 0)
                if fi == 0:
                    early_break_count += 1
                elif fi == 1:
                    one_fix_count += 1
                else:
                    two_fix_count += 1

                is_oracle = cand.get("is_oracle_correct", False)
                final_verif_pass = cand.get("final_verif_all_pass")

                if final_verif_pass is False and is_oracle:
                    verif_fail_oracle_correct += 1
                if final_verif_pass is True and not is_oracle:
                    verif_pass_oracle_wrong += 1

                iters = cand.get("iterations", [])
                prev_verif_all_pass = None
                for it_entry in iters:
                    for r in (it_entry.get("verif_results") or []):
                        tt_fire[r["test_type"]] += 1
                        if r["status"] == "fail":
                            tt_fail[r["test_type"]] += 1

                    if it_entry.get("fix_triggered"):
                        fix_triggered_total += 1
                        changed = it_entry.get("fix_produced_different_sql")
                        if changed:
                            fix_changed_total += 1
                        # Check if verif improved between this iter and next
                        if prev_verif_all_pass is False and it_entry.get("verif_all_pass") is True:
                            verif_improved_count += 1
                        for r in (it_entry.get("verif_results") or []):
                            if r["status"] == "fail":
                                tt_fix_trigger[r["test_type"]] += 1

                    prev_verif_all_pass = it_entry.get("verif_all_pass")

                # Accumulate verif fail→pass fixes from per-candidate detail
                for fixed_list in cand.get("verif_issues_fixed_by_iter", []):
                    verif_issues_fixed_counts.update(fixed_list)

                # Accumulate oracle transitions
                for transition in cand.get("oracle_transitions", []):
                    direction = transition.get("direction", "")
                    triggering = transition.get("verif_issues_triggered", [])
                    if direction == "wrong_to_correct":
                        wrong_to_correct_count += 1
                        issues_triggering_correct_trans.update(triggering)
                    elif direction == "correct_to_wrong":
                        correct_to_wrong_count += 1
                        issues_triggering_wrong_trans.update(triggering)

        all_types = sorted(set(list(tt_fire.keys()) + list(tt_fail.keys())))
        op8_verif_summary = {
            "questions_with_detail": nd,
            "avg_plan_test_count": round(
                sum(d.get("verif_plan_count", 0) for d in op8_details) / nd, 2
            ),
            "questions_with_plan": sum(
                1 for d in op8_details if d.get("verif_plan_count", 0) > 0
            ),
            "questions_with_plan_error": sum(
                1 for d in op8_details if d.get("verif_plan_error")
            ),
            # Per-test-type breakdown
            "test_type_fire_counts": dict(tt_fire),
            "test_type_fail_counts": dict(tt_fail),
            "test_type_fix_trigger_counts": dict(tt_fix_trigger),
            "test_type_fail_rate": {
                tt: round(tt_fail[tt] / tt_fire[tt], 3) if tt_fire[tt] > 0 else 0.0
                for tt in all_types
            },
            "test_type_fix_trigger_rate": {
                tt: round(tt_fix_trigger[tt] / tt_fail[tt], 3) if tt_fail[tt] > 0 else 0.0
                for tt in all_types
            },
            # Calibration
            "verif_fail_but_oracle_correct_rate": round(
                verif_fail_oracle_correct / total_cands, 3
            ) if total_cands > 0 else 0.0,
            "verif_pass_but_oracle_wrong_rate": round(
                verif_pass_oracle_wrong / total_cands, 3
            ) if total_cands > 0 else 0.0,
            # Fix effectiveness
            "fix_changed_sql_rate": round(
                fix_changed_total / fix_triggered_total, 3
            ) if fix_triggered_total > 0 else 0.0,
            "fix_improved_verif_rate": round(
                verif_improved_count / fix_triggered_total, 3
            ) if fix_triggered_total > 0 else 0.0,
            # Iteration distribution (over candidates)
            "early_break_rate": round(early_break_count / total_cands, 3) if total_cands > 0 else 0.0,
            "one_fix_rate": round(one_fix_count / total_cands, 3) if total_cands > 0 else 0.0,
            "two_fix_rate": round(two_fix_count / total_cands, 3) if total_cands > 0 else 0.0,
            # Verification issue fix effectiveness
            # How many verification tests actually transitioned fail→pass after a fix
            "verif_issues_actually_fixed_counts": dict(verif_issues_fixed_counts),
            # Oracle-level impact of fix iterations (per candidate, not per question)
            "wrong_to_correct_count": wrong_to_correct_count,
            "correct_to_wrong_count": correct_to_wrong_count,
            "net_fix_oracle_impact": wrong_to_correct_count - correct_to_wrong_count,
            # Which verification issue types were present in fixes that caused transitions
            "issues_triggering_correct_trans": dict(issues_triggering_correct_trans),
            "issues_triggering_wrong_trans": dict(issues_triggering_wrong_trans),
        }

    # ---- Op9 ----
    op9_traces = [t.get("op9", {}) for t in traces]
    method_counts: Counter = Counter(o.get("method", "error") for o in op9_traces)
    generator_wins: Counter = Counter()
    for o in op9_traces:
        for gen, wins in o.get("tournament_wins", {}).items():
            generator_wins[gen] += int(wins)
    total_wins = sum(generator_wins.values())
    op9_summary = {
        "fast_path_rate": method_counts.get("fast_path", 0) / n,
        "tournament_rate": method_counts.get("tournament", 0) / n,
        "fallback_rate": method_counts.get("fallback", 0) / n,
        "error_rate": method_counts.get("error", 0) / n,
        "avg_clusters": _safe_avg(op9_traces, "cluster_count"),
        "generator_win_counts": dict(generator_wins),
        "generator_win_rates": (
            {g: c / total_wins for g, c in generator_wins.items()}
            if total_wins > 0 else {}
        ),
        "avg_duration_s": _safe_avg(op9_traces, "duration_s"),
    }

    # ---- Oracle ----
    oracle_summary = {
        "oracle_pre_fix_rate": sum(1 for t in traces if t.get("oracle_pre_fix")) / n,
        "oracle_post_fix_rate": sum(1 for t in traces if t.get("oracle_post_fix")) / n,
        "fixer_improvement_count": sum(
            1 for t in traces
            if t.get("oracle_post_fix") and not t.get("oracle_pre_fix")
        ),
        "selector_precision": (
            sum(1 for t in traces if t.get("selector_matched_oracle"))
            / max(sum(1 for t in traces if t.get("oracle_post_fix")), 1)
        ),
    }

    return {
        "n_total": n,
        "n_correct": n_correct,
        "overall_ex": n_correct / n,
        "by_difficulty": by_difficulty,
        "by_db": by_db,
        "op5": op5_summary,
        "op6": op6_summary,
        "op7": op7_summary,
        "op8": op8_summary,
        "op8_verification": op8_verif_summary,
        "op9": op9_summary,
        "oracle": oracle_summary,
        "total_latency_s": sum(t.get("latency_s", 0) for t in traces),
        "avg_latency_s": _safe_avg(traces, "latency_s"),
        "total_cost_estimate": 0.0,  # Phase 1: cost tracking not implemented
    }


def _safe_avg(items: list[dict], key: str) -> float:
    vals = [v for o in items if (v := o.get(key)) is not None and isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _trace_to_result(trace: dict) -> dict:
    """Extract the compact per-question result from a full trace."""
    return {
        "question_id": trace.get("question_id"),
        "db_id": trace.get("db_id"),
        "difficulty": trace.get("difficulty"),
        "question": trace.get("question", "")[:200],
        "predicted_sql": trace.get("predicted_sql", ""),
        "truth_sql": trace.get("truth_sql", ""),
        "correct": trace.get("correct", False),
        "selection_method": trace.get("selection_method", "error"),
        "winner_generator": trace.get("winner_generator", ""),
        "cluster_count": trace.get("cluster_count", 0),
        "latency_s": trace.get("latency_s", 0.0),
        "error_stage": trace.get("error_stage"),
        "oracle_pre_fix": trace.get("oracle_pre_fix", False),
        "oracle_post_fix": trace.get("oracle_post_fix", False),
        "selector_matched_oracle": trace.get("selector_matched_oracle", False),
        "schema_recall_s1_complete": trace.get("op6", {}).get("schema_recall", {}).get("s1_complete"),
        "schema_recall_s2_complete": trace.get("op6", {}).get("schema_recall", {}).get("s2_complete"),
        "schema_recall_col_s1": trace.get("op6", {}).get("schema_recall", {}).get("col_recall_s1"),
        "schema_recall_col_s2": trace.get("op6", {}).get("schema_recall", {}).get("col_recall_s2"),
        "total_candidates": trace.get("op7", {}).get("total_candidates", 0),
        "needed_fix_count": trace.get("op8", {}).get("needed_fix_count", 0),
    }


# ---------------------------------------------------------------------------
# Print results table (live during run)
# ---------------------------------------------------------------------------

def print_running_summary(results: list[dict]) -> None:
    n = len(results)
    n_correct = sum(1 for r in results if r.get("correct", False))
    ex = n_correct / n if n > 0 else 0.0
    by_diff: dict[str, list] = defaultdict(list)
    for r in results:
        by_diff[r.get("difficulty", "?")].append(r.get("correct", False))

    diff_str = " | ".join(
        f"{d[0].upper()}: {sum(v)}/{len(v)}"
        for d, v in sorted(by_diff.items())
    )
    print(
        f"\r  [{n}/66] EX={ex:.1%} ({n_correct}/{n}) | {diff_str}",
        end="", flush=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir, level=args.log_level)

    _mem_mib("startup baseline (imports + model loading done)")
    logger.info("=" * 70)
    logger.info("NL2SQL Smoke Test — 66 Questions (6 per DB × 11 databases)")
    logger.info("=" * 70)
    logger.info("Output dir    : %s", output_dir)
    logger.info("Workers       : %d", args.workers)
    logger.info("LLM provider  : %s", settings.llm_provider)
    logger.info("Model fast    : %s", settings.model_fast)
    logger.info("Model powerful: %s", settings.model_powerful)
    logger.info("Model reasoning: %s", settings.model_reasoning)
    logger.info("Cache enabled : %s", settings.cache_llm_responses)

    # -----------------------------------------------------------------------
    # Load BIRD dev data + sample 66 questions
    # -----------------------------------------------------------------------
    logger.info("Loading BIRD dev data …")
    dev_entries = load_bird_split("dev", str(_REPO_ROOT / "data" / "bird"))
    if not dev_entries:
        logger.error("No BIRD dev entries found — check BIRD_DATA_DIR")
        sys.exit(1)
    logger.info("Loaded %d dev questions", len(dev_entries))

    sampled = stratified_sample_66(dev_entries)
    logger.info(
        "Sampled %d questions (2 simple + 2 moderate + 2 challenging per DB)",
        len(sampled),
    )
    dist = Counter((e.db_id, e.difficulty) for e in sampled)
    for (db, diff), count in sorted(dist.items()):
        logger.debug("  %s / %s: %d", db, diff, count)

    # -----------------------------------------------------------------------
    # Load shared ExampleStore
    # -----------------------------------------------------------------------
    ex_store_faiss = str(_INDICES_DIR / "example_store.faiss")
    ex_store_meta = str(_INDICES_DIR / "example_store_metadata.json")
    logger.info("Loading ExampleStore from %s …", ex_store_faiss)
    example_store = ExampleStore.load(ex_store_faiss, ex_store_meta)
    logger.info("ExampleStore loaded (%d entries)", len(example_store._metadata))

    # -----------------------------------------------------------------------
    # Resume: load already-completed results
    # -----------------------------------------------------------------------
    results_path = output_dir / "results.json"
    traces_path = output_dir / "detailed_traces.json"
    done_ids: set[int] = set()
    completed_traces: list[dict] = []
    completed_results: list[dict] = []

    if args.resume and results_path.exists() and traces_path.exists():
        with open(results_path, encoding="utf-8") as f:
            completed_results = json.load(f)
        with open(traces_path, encoding="utf-8") as f:
            completed_traces = json.load(f)
        done_ids = {r["question_id"] for r in completed_results}
        logger.info("Resuming: %d/%d questions already done", len(done_ids), len(sampled))

    pending = [e for e in sampled if e.question_id not in done_ids]
    logger.info("Processing %d questions …", len(pending))

    # -----------------------------------------------------------------------
    # Process questions one database at a time — load artifacts, run all
    # questions for that DB concurrently, then free memory before next DB.
    # -----------------------------------------------------------------------
    # Precompute the original sampled index per question_id (for progress display)
    sampled_idx: dict[int, int] = {e.question_id: i for i, e in enumerate(sampled)}

    # Group pending questions by db_id
    pending_by_db: dict[str, list] = {}
    for entry in pending:
        pending_by_db.setdefault(entry.db_id, []).append(entry)

    def _save_incremental() -> None:
        with open(traces_path, "w", encoding="utf-8") as f:
            json.dump(completed_traces, f, ensure_ascii=False, indent=2)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(completed_results, f, ensure_ascii=False, indent=2)

    # Phase 1 timeout: grounding + schema linking + small buffer
    _P1_TIMEOUT = _TIMEOUT_GROUNDING + _TIMEOUT_SCHEMA_LINKING + 30
    # Phase 2 timeout: generation + fixing + selection + small buffer
    _P2_TIMEOUT = (
        max(_TIMEOUT_GEN_REASONING, _TIMEOUT_GEN_STANDARD, _TIMEOUT_GEN_ICL)
        + _TIMEOUT_FIXING + _TIMEOUT_SELECTION + 60
    )

    for db_id in sorted(pending_by_db.keys()):
        db_questions = pending_by_db[db_id]

        # Load artifacts for this DB only
        logger.info("  [%s] Loading artifacts for %d question(s) …", db_id, len(db_questions))
        mem_before_load = _mem_mib(f"{db_id} before loading artifacts")
        try:
            artifacts = load_artifacts(db_id)
            mem_after_load = _mem_mib(f"{db_id} after loading artifacts (FAISS: {len(artifacts['faiss_index']._fields)} fields)")
            print(f"  [{db_id}] artifacts loaded — index memory added: {mem_after_load - mem_before_load:+.1f} MiB")
            logger.info(
                "  [%s] artifacts loaded (FAISS: %d fields, LSH ready)",
                db_id,
                len(artifacts["faiss_index"]._fields),
            )
        except Exception as exc:
            logger.error("  [%s] FAILED to load artifacts: %s — skipping %d question(s)", db_id, exc, len(db_questions))
            for entry in db_questions:
                error_trace = _build_error_trace(entry, "artifacts_not_loaded", "artifacts not loaded")
                completed_traces.append(error_trace)
                completed_results.append(_trace_to_result(error_trace))
            _save_incremental()
            continue

        # ── PHASE 1: Op 5-6 for ALL questions (index-heavy) ──────────────────
        # Run concurrently; gather ALL results before freeing indices.
        phase1_semaphore = asyncio.Semaphore(args.workers)

        async def _p1(entry_=None, idx_=None):
            async with phase1_semaphore:
                return await asyncio.wait_for(
                    _ops_5_6_body(entry_, artifacts, example_store, idx_, len(sampled)),
                    timeout=_P1_TIMEOUT,
                )

        phase1_tasks = [
            _p1(entry_=e, idx_=sampled_idx[e.question_id])
            for e in db_questions
        ]
        phase1_results = await asyncio.gather(*phase1_tasks, return_exceptions=True)

        # ── FREE INDICES: LSH and FAISS not needed for Op 7-9 ────────────────
        mem_before_free = _mem_mib(f"{db_id} before freeing indexes")
        artifacts["lsh_index"] = None
        artifacts["faiss_index"] = None
        gc.collect()
        _malloc_trim()
        mem_after_free = _mem_mib(f"{db_id} after freeing indexes + gc.collect() + malloc_trim")
        print(
            f"  [{db_id}] Index memory freed: {mem_after_free - mem_before_free:+.1f} MiB "
            f"(before={mem_before_free:.1f}, after={mem_after_free:.1f}). "
            f"Phase 2: SQL generation + selection …"
        )
        logger.info("  [%s] LSH + FAISS freed after Op 5-6.", db_id)

        # ── PHASE 2: Op 7-9 for ALL questions (LLM-only) ─────────────────────
        db_path = artifacts["db_path"]
        phase2_semaphore = asyncio.Semaphore(args.workers)
        phase2_tasks = []

        for entry, p1 in zip(db_questions, phase1_results):
            if isinstance(p1, Exception):
                error_trace = _build_error_trace(entry, "phase1_error", str(p1))
                completed_traces.append(error_trace)
                completed_results.append(_trace_to_result(error_trace))
                logger.error("  [%s] Phase 1 failed for Q#%d: %s", db_id, entry.question_id, p1)
            else:
                partial_trace, grounding_ctx, linked_schemas, gold_exec, t_total_start = p1

                async def _p2(
                    entry_=None, grounding_=None, schemas_=None,
                    trace_=None, gold_=None, t_start_=None,
                ):
                    async with phase2_semaphore:
                        return await asyncio.wait_for(
                            _ops_7_9_body(
                                entry_, db_path, grounding_, schemas_,
                                trace_, gold_, t_start_,
                            ),
                            timeout=_P2_TIMEOUT,
                        )

                phase2_tasks.append(_p2(
                    entry_=entry,
                    grounding_=grounding_ctx,
                    schemas_=linked_schemas,
                    trace_=partial_trace,
                    gold_=gold_exec,
                    t_start_=t_total_start,
                ))

        for coro in asyncio.as_completed(phase2_tasks):
            try:
                trace = await coro
            except asyncio.TimeoutError:
                logger.error("Phase 2 question hit timeout of %ds", _P2_TIMEOUT)
                continue
            except Exception as exc:
                logger.error("Unexpected error in Phase 2: %s", exc)
                logger.debug(traceback.format_exc())
                continue

            completed_traces.append(trace)
            completed_results.append(_trace_to_result(trace))
            _save_incremental()
            print_running_summary(completed_results)

        # Free remaining artifacts before loading the next DB
        del artifacts
        gc.collect()
        _malloc_trim()
        mem_end = _mem_mib(f"{db_id} after Phase 2 complete + final gc")
        logger.info("  [%s] Done. All memory released.", db_id)

    print()  # newline after progress indicator

    # -----------------------------------------------------------------------
    # Compute and save aggregated summary
    # -----------------------------------------------------------------------
    logger.info("Computing component summary …")
    summary = compute_component_summary(completed_traces)
    summary_path = output_dir / "component_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # -----------------------------------------------------------------------
    # Save failed questions (with detailed diagnostics)
    # -----------------------------------------------------------------------
    failed = []
    for trace in completed_traces:
        if not trace.get("correct", False):
            failure_reason = _classify_failure(trace)
            failed.append({
                "question_id": trace.get("question_id"),
                "db_id": trace.get("db_id"),
                "difficulty": trace.get("difficulty"),
                "question": trace.get("question", ""),
                "evidence": trace.get("evidence", ""),
                "predicted_sql": trace.get("predicted_sql", ""),
                "truth_sql": trace.get("truth_sql", ""),
                "error_stage": trace.get("error_stage"),
                "oracle_pre_fix": trace.get("oracle_pre_fix", False),
                "oracle_post_fix": trace.get("oracle_post_fix", False),
                "selector_matched_oracle": trace.get("selector_matched_oracle", False),
                "schema_recall": trace.get("op6", {}).get("schema_recall", {}),
                "total_candidates": trace.get("op7", {}).get("total_candidates", 0),
                "needed_fix_count": trace.get("op8", {}).get("needed_fix_count", 0),
                "fixed_ok_count": trace.get("op8", {}).get("fixed_ok_count", 0),
                "selection_method": trace.get("selection_method", ""),
                "latency_s": trace.get("latency_s", 0),
                "failure_reason": failure_reason,
            })

    failed_path = output_dir / "failed_questions.json"
    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)

    # -----------------------------------------------------------------------
    # Save Op8 verification report
    # -----------------------------------------------------------------------
    op8_report = [
        t["op8_detail"]
        for t in completed_traces
        if "op8_detail" in t
    ]
    op8_report_path = output_dir / "op8_verification_report.json"
    with open(op8_report_path, "w", encoding="utf-8") as f:
        json.dump(op8_report, f, ensure_ascii=False, indent=2)
    logger.info("Op8 verification report: %d questions", len(op8_report))

    # -----------------------------------------------------------------------
    # Print final summary
    # -----------------------------------------------------------------------
    _print_final_summary(summary, output_dir)

    logger.info("=" * 70)
    logger.info("Smoke test complete. Output files:")
    logger.info("  results.json                 → %s", results_path)
    logger.info("  detailed_traces.json         → %s", traces_path)
    logger.info("  component_summary.json       → %s", summary_path)
    logger.info("  op8_verification_report.json → %s", op8_report_path)
    logger.info("  failed_questions.json        → %s", failed_path)
    logger.info("  smoke_test.log               → %s", output_dir / "smoke_test.log")
    logger.info(
        "\nTo analyze results: python scripts/analyze_results.py %s", results_path
    )


def _classify_failure(trace: dict) -> str:
    """Classify the primary reason a question was answered incorrectly."""
    if trace.get("error_stage"):
        return f"pipeline_error:{trace['error_stage']}"
    if not trace.get("oracle_pre_fix") and not trace.get("oracle_post_fix"):
        # Check schema recall
        recall = trace.get("op6", {}).get("schema_recall", {})
        if not recall.get("s2_complete", True):
            return "schema_miss"
        return "generation_failure"
    if trace.get("oracle_post_fix") and not trace.get("selector_matched_oracle"):
        return "selector_miss"
    if trace.get("oracle_pre_fix") and not trace.get("oracle_post_fix"):
        return "fixer_regression"
    if trace.get("oracle_post_fix") and trace.get("selector_matched_oracle"):
        return "eval_mismatch"  # selector picked right but EX still wrong
    return "unknown"


def _print_final_summary(summary: dict, output_dir: Path) -> None:
    n = summary.get("n_total", 0)
    n_correct = summary.get("n_correct", 0)
    ex = summary.get("overall_ex", 0)

    print(f"\n{'=' * 65}")
    print(f"SMOKE TEST RESULTS — {output_dir.name}")
    print(f"{'=' * 65}")
    print(f"  Total questions : {n}")
    print(f"  Correct         : {n_correct}/{n} = {ex:.1%}")
    print()

    print("  --- By Difficulty ---")
    for diff, acc in sorted(summary.get("by_difficulty", {}).items()):
        bar = "#" * int(acc * 20) + "." * (20 - int(acc * 20))
        print(f"  {diff:<12} {acc:.1%}  [{bar}]")

    print()
    print("  --- By Database (worst first) ---")
    db_items = sorted(summary.get("by_db", {}).items(), key=lambda x: x[1])
    for db_id, acc in db_items:
        bar = "#" * int(acc * 20) + "." * (20 - int(acc * 20))
        print(f"  {db_id:<30} {acc:.1%}  [{bar}]")

    op9 = summary.get("op9", {})
    print()
    print(f"  --- Selection ---")
    print(f"  Fast path    : {op9.get('fast_path_rate', 0):.1%}")
    print(f"  Tournament   : {op9.get('tournament_rate', 0):.1%}")
    print(f"  Fallback     : {op9.get('fallback_rate', 0):.1%}")
    print(f"  Avg clusters : {op9.get('avg_clusters', 0):.1f}")

    oracle = summary.get("oracle", {})
    print()
    print(f"  --- Oracle ---")
    print(f"  Oracle pre-fix  : {oracle.get('oracle_pre_fix_rate', 0):.1%}")
    print(f"  Oracle post-fix : {oracle.get('oracle_post_fix_rate', 0):.1%}")
    print(f"  Selector precision (when oracle achievable): {oracle.get('selector_precision', 0):.1%}")

    print()
    print(f"  Avg latency  : {summary.get('avg_latency_s', 0):.1f}s/question")
    total_lat = summary.get("total_latency_s", 0)
    print(f"  Total time   : {total_lat/60:.1f} min")
    print(f"{'=' * 65}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 66-question NL2SQL smoke test with detailed component analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory for output files. "
            "Defaults to results/smoke_test_66q (or timestamped if --timestamp)."
        ),
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Append YYYYMMDD_HHMMSS to output_dir to avoid overwriting previous runs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Max concurrent questions (semaphore size). Higher = faster but more API pressure.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output_dir/results.json if it exists.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.output_dir is None:
        base = _REPO_ROOT / "results" / "smoke_test_66q"
        if args.timestamp:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = str(base.parent / f"smoke_test_66q_{ts}")
        else:
            args.output_dir = str(base)

    asyncio.run(main(args))
