"""
Checkpoint E — End-to-End Component Test (v3)

Runs each component (Grounding -> Schema Linking -> Generation -> Fixing -> Selection)
in sequence on 33 stratified BIRD dev questions. Reports accuracy and stage-level diagnostics.

Extended metrics (v3):
  - Oracle performance on generated candidates (pre-fix and post-fix)
  - Query Fixer performance: effectiveness and fix success rate
  - Query Selector performance: when oracle is achievable, how often selector picks it
  - Schema Linker recall/accuracy: coverage of gold-required tables and columns in S1/S2
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

# Suppress HuggingFace/tokenizers verbose output to prevent BrokenPipeError
# when the script is run with pipe commands (e.g. | head -N).
# These must be set BEFORE any HuggingFace imports.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Handle broken pipes gracefully (e.g. when run as | head -N)
if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

# Ensure src/ is on the Python path
sys.path.insert(0, "/Users/mostafa/Documents/workplace/NL2SQL")
sys.path.insert(0, "/Users/mostafa/Documents/workplace/NL2SQL/src")

# Set up logging before imports that might trigger early log output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("checkpoint_e")

from src.config.settings import settings  # noqa: E402
from src.data.bird_loader import load_bird_split  # noqa: E402
from src.data.database import execute_sql  # noqa: E402
from src.fixing.query_fixer import QueryFixer  # noqa: E402
from src.generation.icl_generator import ICLGenerator  # noqa: E402
from src.generation.reasoning_generator import ReasoningGenerator  # noqa: E402
from src.generation.standard_generator import StandardAndComplexGenerator  # noqa: E402
from src.grounding.context_grounder import ground_context  # noqa: E402
from src.indexing.example_store import ExampleStore  # noqa: E402
from src.indexing.faiss_index import FAISSIndex  # noqa: E402
from src.indexing.lsh_index import LSHIndex  # noqa: E402
from src.schema_linking.schema_linker import link_schema  # noqa: E402
from src.selection.adaptive_selector import AdaptiveSelector  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE = Path("/Users/mostafa/Documents/workplace/NL2SQL")
_PREPROCESSED = _BASE / "data/preprocessed"
_INDICES_DIR = _PREPROCESSED / "indices"
_SCHEMAS_DIR = _PREPROCESSED / "schemas"
_DEV_JSON = _BASE / "data/bird/dev/dev.json"
_DEV_DATABASES = _BASE / "data/bird/dev/dev_databases"
_OUTPUT_DIR = _BASE / "checkpoint_E_review"

# ---------------------------------------------------------------------------
# Per-stage timeouts (seconds)
# ---------------------------------------------------------------------------
# Grounding: fast LSH + FAISS lookup, no LLM
_TIMEOUT_GROUNDING = 30
# Schema linking: up to 2 gemini-2.5-pro calls; wide schemas (115+ cols) can
# take 60-90s for S1 alone. With 3 retries + exponential backoff the worst-case
# wall time is ~3×60 + 32 ≈ 212s, so 300s gives comfortable headroom.
_TIMEOUT_SCHEMA_LINKING = 300
# Generation: 11 candidates run concurrently (4 reasoning + 2 standard +
# 2 complex + 3 ICL); reasoning uses extended thinking so allow 300s to
# handle Gemini API latency variability (was 120s, caused 4/33 timeouts).
_TIMEOUT_GENERATION = 300
# Fixing: β=2 iterations × ~15s each per candidate, but candidates are fixed
# concurrently; 120s covers 11 concurrent candidates × β=2 with API latency headroom.
_TIMEOUT_FIXING = 120
# Selection: fast_path is instant; tournament pairwise calls ~5s each × ~5
# comparisons = 25s max. 45s gives comfortable headroom.
_TIMEOUT_SELECTION = 45
# Safety-net total: sum of all per-stage timeouts + 30s overhead.
# This outer timeout only fires if a stage's internal timeout somehow leaks.
_TIMEOUT_TOTAL_SAFETY = _TIMEOUT_GROUNDING + _TIMEOUT_SCHEMA_LINKING + _TIMEOUT_GENERATION + _TIMEOUT_FIXING + _TIMEOUT_SELECTION + 30


# ---------------------------------------------------------------------------
# Stratified sampling: 33 questions, 3 per DB (1 simple, 1 moderate, 1 challenging)
# ---------------------------------------------------------------------------

def stratified_sample_33(entries) -> list:
    """
    Select 33 BIRD dev questions: 3 per each of the 11 databases,
    1 per difficulty level (simple, moderate, challenging).
    Uses random.seed(42) for reproducibility.
    """
    random.seed(42)

    # Group by (db_id, difficulty)
    groups: dict[tuple[str, str], list] = {}
    for entry in entries:
        key = (entry.db_id, entry.difficulty)
        groups.setdefault(key, []).append(entry)

    db_ids = sorted({entry.db_id for entry in entries})
    difficulties = ["simple", "moderate", "challenging"]

    selected = []
    for db_id in db_ids:
        for diff in difficulties:
            pool = groups.get((db_id, diff), [])
            if pool:
                selected.append(random.choice(pool))
            else:
                logger.warning("No %s questions found for db_id=%s", diff, db_id)

    return selected


# ---------------------------------------------------------------------------
# Load offline artifacts for a single database
# ---------------------------------------------------------------------------

def load_artifacts(db_id: str) -> dict:
    """
    Load all preprocessed artifacts for a database.
    Returns a dict with keys: lsh_index, faiss_index, full_ddl, full_markdown,
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

    # Build available_fields list: [(table, column, short_summary, long_summary)]
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
    """Normalize a single cell value for comparison."""
    if v is None:
        return "NULL"
    # Normalize float/int equivalence: 1.0 == 1
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f"{v:.6f}"
    return str(v)


def _normalize_rows(rows: list) -> list:
    """Normalize rows for comparison: sort and normalize values."""
    normalized = [
        tuple(_normalize_val(cell) for cell in row)
        for row in rows
    ]
    return sorted(normalized)


def results_match(rows_a: list, rows_b: list) -> bool:
    """Compare two result sets: sort, normalize, and compare."""
    if not rows_a and not rows_b:
        return True
    return _normalize_rows(rows_a) == _normalize_rows(rows_b)


# ---------------------------------------------------------------------------
# Schema Linker recall helpers
# ---------------------------------------------------------------------------

def _extract_required_tables_columns(gold_sql: str) -> tuple[set, set]:
    """
    Extract required tables and (table, column) pairs from a gold SQL query.

    Only real table names (from FROM/JOIN clauses) are included in required_tables.
    SQL aliases such as T1, T2, t, s, etc. are excluded from required_tables but
    their column references are still tracked in table_cols so that the bare-column
    fallback in _compute_schema_recall can match them.

    Returns:
        (required_tables: set[str], required_cols: set[tuple[str, str]])
        where required_cols contains (table_lower, column_lower) pairs.
    """
    # Alias pattern: single letter optionally followed by digits (t, t1, t2, s, a, b, ...).
    # This covers the overwhelming majority of SQL alias styles used in BIRD gold SQL.
    _ALIAS_RE = re.compile(r'^[a-z]\d*$', re.IGNORECASE)

    # Extract REAL table names from FROM and JOIN clauses.
    # The token immediately after FROM/JOIN is always the real table name; aliases
    # appear later after AS, so they are not captured here.
    tables: set[str] = set()
    for m in re.finditer(r'(?:FROM|JOIN)\s+(\w+)', gold_sql, re.IGNORECASE):
        name = m.group(1).lower()
        # Guard against edge-case keywords (subquery or LATERAL keyword after FROM)
        if name not in ('select', 'lateral', 'as'):
            tables.add(name)

    # Extract table.column patterns (e.g. T1.col or schools.CDSCode).
    # Always track the column reference; only add the table prefix to required_tables
    # when it is NOT an alias (so aliases like T1, t2 don't inflate the denominator).
    table_cols: set[tuple[str, str]] = set()
    for m in re.finditer(r'(\w+)\.(\w+)', gold_sql):
        t_name = m.group(1).lower()
        c_name = m.group(2).lower()
        if t_name.isdigit() or c_name.isdigit():
            continue
        table_cols.add((t_name, c_name))
        if not _ALIAS_RE.match(t_name):
            tables.add(t_name)

    return tables, table_cols


def _compute_schema_recall(
    s1_ddl: str,
    s2_ddl: str,
    s1_fields: list,
    s2_fields: list,
    gold_sql: str,
) -> dict:
    """
    Compute S1/S2 recall against the gold SQL requirements.

    Uses two strategies:
    1. Table-level recall: checks if each required table name appears in the DDL
       (covers tables identified via FROM/JOIN in gold SQL).
    2. Column-level recall: checks (table, column) pairs identified as "table.column"
       patterns in gold SQL against s1_fields / s2_fields.

    Returns a dict with recall metrics.
    """
    required_tables, required_cols = _extract_required_tables_columns(gold_sql)

    s1_ddl_lower = (s1_ddl or "").lower()
    s2_ddl_lower = (s2_ddl or "").lower()

    # Table recall: look for "create table tablename" or quoted variants
    tables_in_s1 = 0
    tables_in_s2 = 0
    missing_in_s1: list[str] = []
    missing_in_s2: list[str] = []

    for t in required_tables:
        # Match CREATE TABLE variants: unquoted, backtick-quoted, double-quoted
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

    # Column recall: check (table, col) pairs against s1_fields/s2_fields
    # s1_fields / s2_fields are lists of (table, column, short_summary, long_summary)
    s1_fields_set: set[tuple[str, str]] = set()
    s2_fields_set: set[tuple[str, str]] = set()
    if s1_fields:
        for f in s1_fields:
            s1_fields_set.add((str(f[0]).lower(), str(f[1]).lower()))
    if s2_fields:
        for f in s2_fields:
            s2_fields_set.add((str(f[0]).lower(), str(f[1]).lower()))

    cols_in_s1 = 0
    cols_in_s2 = 0
    missing_cols_in_s1: list[str] = []
    missing_cols_in_s2: list[str] = []

    for t, c in required_cols:
        # Also try without table prefix (bare column name) in case of aliased tables
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
        "tables_in_s1": tables_in_s1,
        "tables_in_s2": tables_in_s2,
        "cols_in_s1": cols_in_s1,
        "cols_in_s2": cols_in_s2,
        "missing_tables_in_s1": missing_in_s1,
        "missing_tables_in_s2": missing_in_s2,
        "missing_cols_in_s1": missing_cols_in_s1,
        "missing_cols_in_s2": missing_cols_in_s2,
        # Exact match: every required element is present
        "s1_complete": (tables_in_s1 == n_tables and cols_in_s1 == n_cols),
        "s2_complete": (tables_in_s2 == n_tables and cols_in_s2 == n_cols),
    }


# ---------------------------------------------------------------------------
# Per-question processing
# ---------------------------------------------------------------------------

async def process_question(
    entry,
    artifacts: dict,
    example_store: ExampleStore,
    question_idx: int,
    total: int,
) -> dict:
    """
    Run the full pipeline (Grounding -> Schema Linking -> Generation -> Fixing -> Selection)
    for a single question. Returns a diagnostics dict.
    """
    db_id = entry.db_id
    question = entry.question
    evidence = entry.evidence
    gold_sql = entry.SQL
    db_path = artifacts["db_path"]

    logger.info(
        "[%d/%d] Processing Q#%d (%s, %s): %s",
        question_idx + 1,
        total,
        entry.question_id,
        db_id,
        entry.difficulty,
        question[:80],
    )

    result_record: dict = {
        "db_id": db_id,
        "question_id": entry.question_id,
        "difficulty": entry.difficulty,
        "question": question,
        "evidence": evidence,
        "gold_sql": gold_sql,
        "stages": {
            "grounding": {
                "keywords_extracted": 0,
                "cell_matches": 0,
                "few_shot_examples": 0,
                "error": None,
            },
            "schema_linking": {
                "s1_fields": 0,
                "s2_fields": 0,
                "error": None,
            },
            "generation": {
                "total_candidates": 0,
                "by_generator": {"reasoning": 0, "standard": 0, "complex": 0, "icl": 0},
                "error_flags": 0,
                "error": None,
            },
            "fixing": {
                "needed_fix": 0,
                "fixed_successfully": 0,
                "still_failing": 0,
                "original_success_count": 0,   # candidates that succeeded before any fixing
                "post_fix_success_count": 0,    # candidates that succeed after fixing
                "error": None,
            },
            "selection": {
                "method": "N/A",
                "winner_generator": "N/A",
                "cluster_count": 0,
                "error": None,
            },
        },
        "final_sql": "",
        "correct": False,
        "error_stage": None,
        # Oracle metrics
        "oracle_pre_fix": False,           # any pre-fix candidate is correct
        "oracle_pre_fix_count": 0,         # how many pre-fix candidates are correct
        "oracle_post_fix": False,          # any post-fix candidate is correct
        "oracle_post_fix_count": 0,        # how many post-fix candidates are correct
        "oracle_total_candidates": 0,
        "selector_matched_oracle": False,  # did selector pick a correct candidate?
        # Schema linker recall
        "schema_recall": {},
        "gold_exec_success": False,
    }

    grounding_ctx = None
    linked_schemas = None
    all_candidates = []
    fixed_candidates = []
    selection_result = None
    correct_post_fix_sqls: set[str] = set()

    # ------------------------------------------------------------------
    # Execute gold SQL early — reused for oracle computation and final eval
    # ------------------------------------------------------------------
    gold_exec_result = execute_sql(db_path, gold_sql)
    result_record["gold_exec_success"] = gold_exec_result.success
    if not gold_exec_result.success:
        logger.warning("  [Gold] Gold SQL failed to execute: %s", gold_exec_result.error)

    # ------------------------------------------------------------------
    # Stage 1: Context Grounding
    # ------------------------------------------------------------------
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
        result_record["stages"]["grounding"]["cell_matches"] = len(grounding_ctx.matched_cells)
        result_record["stages"]["grounding"]["few_shot_examples"] = len(grounding_ctx.few_shot_examples)
        # Count keywords as literals + schema_references
        result_record["stages"]["grounding"]["keywords_extracted"] = (
            len(grounding_ctx.schema_hints)
        )
        logger.info(
            "  [Grounding] cell_matches=%d, few_shot=%d, schema_hints=%d",
            len(grounding_ctx.matched_cells),
            len(grounding_ctx.few_shot_examples),
            len(grounding_ctx.schema_hints),
        )
    except asyncio.TimeoutError:
        err_msg = f"Grounding timed out after {_TIMEOUT_GROUNDING}s"
        result_record["stages"]["grounding"]["error"] = err_msg
        if result_record["error_stage"] is None:
            result_record["error_stage"] = "grounding_timeout"
        logger.error("  [Grounding] TIMEOUT after %ds — using empty context", _TIMEOUT_GROUNDING)
        from src.grounding.context_grounder import GroundingContext
        grounding_ctx = GroundingContext()
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        result_record["stages"]["grounding"]["error"] = err_msg
        result_record["error_stage"] = "grounding"
        logger.error("  [Grounding] FAILED: %s", err_msg)
        logger.debug(traceback.format_exc())
        # Create minimal grounding context for downstream stages
        from src.grounding.context_grounder import GroundingContext
        grounding_ctx = GroundingContext()

    # ------------------------------------------------------------------
    # Stage 2: Schema Linking
    # ------------------------------------------------------------------
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
        result_record["stages"]["schema_linking"]["s1_fields"] = len(linked_schemas.s1_fields)
        result_record["stages"]["schema_linking"]["s2_fields"] = len(linked_schemas.s2_fields)
        logger.info(
            "  [Schema Linking] s1_fields=%d, s2_fields=%d",
            len(linked_schemas.s1_fields),
            len(linked_schemas.s2_fields),
        )
        # Compute schema recall vs gold SQL requirements
        schema_recall_data = _compute_schema_recall(
            linked_schemas.s1_ddl,
            linked_schemas.s2_ddl,
            linked_schemas.s1_fields,
            linked_schemas.s2_fields,
            gold_sql,
        )
        result_record["schema_recall"] = schema_recall_data
        logger.info(
            "  [Schema Recall] table_recall: S1=%.0f%% S2=%.0f%% | col_recall: S1=%.0f%% S2=%.0f%%",
            schema_recall_data["table_recall_s1"] * 100,
            schema_recall_data["table_recall_s2"] * 100,
            schema_recall_data["col_recall_s1"] * 100,
            schema_recall_data["col_recall_s2"] * 100,
        )
    except asyncio.TimeoutError:
        err_msg = f"Schema linking timed out after {_TIMEOUT_SCHEMA_LINKING}s — falling back to full DDL"
        result_record["stages"]["schema_linking"]["error"] = err_msg
        if result_record["error_stage"] is None:
            result_record["error_stage"] = "schema_linking_timeout"
        logger.error("  [Schema Linking] TIMEOUT after %ds — using full DDL fallback", _TIMEOUT_SCHEMA_LINKING)
        from src.schema_linking.schema_linker import LinkedSchemas
        linked_schemas = LinkedSchemas(
            s1_ddl=artifacts["full_ddl"],
            s1_markdown=artifacts["full_markdown"],
            s2_ddl=artifacts["full_ddl"],
            s2_markdown=artifacts["full_markdown"],
            s1_fields=[],
            s2_fields=[],
            selection_reasoning="",
        )
        # Still compute schema recall even with fallback (full DDL should have 100% recall)
        result_record["schema_recall"] = _compute_schema_recall(
            linked_schemas.s1_ddl, linked_schemas.s2_ddl,
            linked_schemas.s1_fields, linked_schemas.s2_fields, gold_sql,
        )
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        result_record["stages"]["schema_linking"]["error"] = err_msg
        if result_record["error_stage"] is None:
            result_record["error_stage"] = "schema_linking"
        logger.error("  [Schema Linking] FAILED: %s", err_msg)
        logger.debug(traceback.format_exc())
        # Create a minimal LinkedSchemas using full DDL as fallback
        from src.schema_linking.schema_linker import LinkedSchemas
        linked_schemas = LinkedSchemas(
            s1_ddl=artifacts["full_ddl"],
            s1_markdown=artifacts["full_markdown"],
            s2_ddl=artifacts["full_ddl"],
            s2_markdown=artifacts["full_markdown"],
            s1_fields=[],
            s2_fields=[],
            selection_reasoning="",
        )
        result_record["schema_recall"] = _compute_schema_recall(
            linked_schemas.s1_ddl, linked_schemas.s2_ddl,
            linked_schemas.s1_fields, linked_schemas.s2_fields, gold_sql,
        )

    # ------------------------------------------------------------------
    # Stage 3: SQL Generation (all 3 generators concurrently)
    # ------------------------------------------------------------------
    try:
        reasoning_gen = ReasoningGenerator()
        standard_gen = StandardAndComplexGenerator()
        icl_gen = ICLGenerator()

        gen_results = await asyncio.wait_for(
            asyncio.gather(
                reasoning_gen.generate(
                    question=question,
                    evidence=evidence,
                    schemas=linked_schemas,
                    grounding=grounding_ctx,
                ),
                standard_gen.generate(
                    question=question,
                    evidence=evidence,
                    schemas=linked_schemas,
                    grounding=grounding_ctx,
                ),
                icl_gen.generate(
                    question=question,
                    evidence=evidence,
                    schemas=linked_schemas,
                    grounding=grounding_ctx,
                ),
                return_exceptions=True,
            ),
            timeout=_TIMEOUT_GENERATION,
        )

        reasoning_candidates = gen_results[0] if not isinstance(gen_results[0], Exception) else []
        standard_candidates = gen_results[1] if not isinstance(gen_results[1], Exception) else []
        icl_candidates = gen_results[2] if not isinstance(gen_results[2], Exception) else []

        # Log generator exceptions
        gen_errors = []
        for i, (name, res) in enumerate(
            [("reasoning", gen_results[0]), ("standard", gen_results[1]), ("icl", gen_results[2])]
        ):
            if isinstance(res, Exception):
                gen_errors.append(f"{name}: {type(res).__name__}: {res}")

        all_candidates = list(reasoning_candidates) + list(standard_candidates) + list(icl_candidates)

        # Count by generator type
        r_count = len([c for c in reasoning_candidates if not c.error_flag])
        s_count = len([c for c in standard_candidates if c.generator_id.startswith("standard")])
        cx_count = len([c for c in standard_candidates if c.generator_id.startswith("complex")])
        icl_count = len([c for c in icl_candidates if not c.error_flag])

        result_record["stages"]["generation"]["total_candidates"] = len(all_candidates)
        result_record["stages"]["generation"]["by_generator"] = {
            "reasoning": r_count,
            "standard": s_count,
            "complex": cx_count,
            "icl": icl_count,
        }
        result_record["stages"]["generation"]["error_flags"] = len(
            [c for c in all_candidates if c.error_flag]
        )
        if gen_errors:
            result_record["stages"]["generation"]["error"] = "; ".join(gen_errors)
            if result_record["error_stage"] is None:
                result_record["error_stage"] = "generation"

        logger.info(
            "  [Generation] total=%d (reasoning=%d, standard=%d, complex=%d, icl=%d), errors=%d",
            len(all_candidates), r_count, s_count, cx_count, icl_count,
            result_record["stages"]["generation"]["error_flags"],
        )

        # ---- Oracle on pre-fix candidates ----
        result_record["oracle_total_candidates"] = len(all_candidates)
        if gold_exec_result.success and all_candidates:
            pre_fix_oracle_count = 0
            for cand in all_candidates:
                if not cand.error_flag and cand.sql:
                    cand_exec = execute_sql(db_path, cand.sql)
                    if cand_exec.success and results_match(cand_exec.rows, gold_exec_result.rows):
                        pre_fix_oracle_count += 1
            result_record["oracle_pre_fix"] = pre_fix_oracle_count > 0
            result_record["oracle_pre_fix_count"] = pre_fix_oracle_count
            logger.info(
                "  [Oracle Pre-Fix] %d/%d candidates are correct",
                pre_fix_oracle_count, len(all_candidates),
            )

    except asyncio.TimeoutError:
        err_msg = f"Generation timed out after {_TIMEOUT_GENERATION}s"
        result_record["stages"]["generation"]["error"] = err_msg
        if result_record["error_stage"] is None:
            result_record["error_stage"] = "generation_timeout"
        logger.error("  [Generation] TIMEOUT after %ds — no candidates produced", _TIMEOUT_GENERATION)
        all_candidates = []
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        result_record["stages"]["generation"]["error"] = err_msg
        if result_record["error_stage"] is None:
            result_record["error_stage"] = "generation"
        logger.error("  [Generation] FAILED: %s", err_msg)
        logger.debug(traceback.format_exc())
        all_candidates = []

    # ------------------------------------------------------------------
    # Stage 4: Query Fixing
    # ------------------------------------------------------------------
    if all_candidates:
        try:
            fixer = QueryFixer()
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

            # Collect diagnostics
            needed_fix = sum(1 for fc in fixed_candidates if fc.fix_iterations > 0)
            fixed_ok = sum(
                1 for fc in fixed_candidates
                if fc.fix_iterations > 0 and fc.execution_result.success and not fc.execution_result.is_empty
            )
            still_failing = sum(
                1 for fc in fixed_candidates
                if not fc.execution_result.success or fc.execution_result.is_empty
            )

            result_record["stages"]["fixing"]["needed_fix"] = needed_fix
            result_record["stages"]["fixing"]["fixed_successfully"] = fixed_ok
            result_record["stages"]["fixing"]["still_failing"] = still_failing

            # Track original vs post-fix execution success rates
            original_success_count = 0
            post_fix_success_count = 0
            for fc in fixed_candidates:
                orig_exec = execute_sql(db_path, fc.original_sql)
                if orig_exec.success and not orig_exec.is_empty:
                    original_success_count += 1
                if fc.execution_result.success and not fc.execution_result.is_empty:
                    post_fix_success_count += 1

            result_record["stages"]["fixing"]["original_success_count"] = original_success_count
            result_record["stages"]["fixing"]["post_fix_success_count"] = post_fix_success_count

            # ---- Oracle on post-fix candidates ----
            if gold_exec_result.success:
                post_fix_oracle_count = 0
                for fc in fixed_candidates:
                    if fc.execution_result.success and results_match(
                        fc.execution_result.rows, gold_exec_result.rows
                    ):
                        post_fix_oracle_count += 1
                        correct_post_fix_sqls.add(fc.final_sql)
                result_record["oracle_post_fix"] = post_fix_oracle_count > 0
                result_record["oracle_post_fix_count"] = post_fix_oracle_count
                logger.info(
                    "  [Oracle Post-Fix] %d/%d fixed candidates are correct",
                    post_fix_oracle_count, len(fixed_candidates),
                )

            logger.info(
                "  [Fixing] needed_fix=%d, fixed_ok=%d, still_failing=%d | "
                "orig_success=%d/%d, post_success=%d/%d",
                needed_fix, fixed_ok, still_failing,
                original_success_count, len(fixed_candidates),
                post_fix_success_count, len(fixed_candidates),
            )
        except asyncio.TimeoutError:
            err_msg = f"Fixing timed out after {_TIMEOUT_FIXING}s"
            result_record["stages"]["fixing"]["error"] = err_msg
            if result_record["error_stage"] is None:
                result_record["error_stage"] = "fixing_timeout"
            logger.error("  [Fixing] TIMEOUT after %ds", _TIMEOUT_FIXING)
            fixed_candidates = []
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            result_record["stages"]["fixing"]["error"] = err_msg
            if result_record["error_stage"] is None:
                result_record["error_stage"] = "fixing"
            logger.error("  [Fixing] FAILED: %s", err_msg)
            logger.debug(traceback.format_exc())
            fixed_candidates = []
    else:
        logger.warning("  [Fixing] Skipped (no candidates)")
        result_record["stages"]["fixing"]["error"] = "skipped — no candidates from generation"

    # ------------------------------------------------------------------
    # Stage 5: Adaptive Selection
    # ------------------------------------------------------------------
    if fixed_candidates:
        try:
            selector = AdaptiveSelector()
            selection_result = await asyncio.wait_for(
                selector.select(
                    candidates=fixed_candidates,
                    question=question,
                    evidence=evidence,
                    schemas=linked_schemas,
                    db_path=db_path,
                ),
                timeout=_TIMEOUT_SELECTION,
            )

            result_record["stages"]["selection"]["method"] = selection_result.selection_method
            result_record["stages"]["selection"]["cluster_count"] = selection_result.cluster_count
            result_record["stages"]["selection"]["winner_generator"] = _find_winner_generator(
                selection_result.final_sql, fixed_candidates
            )
            result_record["final_sql"] = selection_result.final_sql

            # Check if selector picked an oracle-correct candidate
            if correct_post_fix_sqls and selection_result.final_sql:
                result_record["selector_matched_oracle"] = (
                    selection_result.final_sql in correct_post_fix_sqls
                )

            logger.info(
                "  [Selection] method=%s, clusters=%d, oracle_match=%s, final_sql=%s...",
                selection_result.selection_method,
                selection_result.cluster_count,
                result_record["selector_matched_oracle"],
                (selection_result.final_sql or "")[:60],
            )
        except asyncio.TimeoutError:
            err_msg = f"Selection timed out after {_TIMEOUT_SELECTION}s — using confidence fallback"
            result_record["stages"]["selection"]["error"] = err_msg
            if result_record["error_stage"] is None:
                result_record["error_stage"] = "selection_timeout"
            logger.error("  [Selection] TIMEOUT after %ds — falling back to best confidence", _TIMEOUT_SELECTION)
            # Fallback: pick highest-confidence candidate
            best = max(fixed_candidates, key=lambda c: c.confidence_score)
            result_record["final_sql"] = best.final_sql
            result_record["stages"]["selection"]["method"] = "fallback_timeout"
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            result_record["stages"]["selection"]["error"] = err_msg
            if result_record["error_stage"] is None:
                result_record["error_stage"] = "selection"
            logger.error("  [Selection] FAILED: %s", err_msg)
            logger.debug(traceback.format_exc())
            # Fallback: pick highest-confidence candidate
            if fixed_candidates:
                best = max(fixed_candidates, key=lambda c: c.confidence_score)
                result_record["final_sql"] = best.final_sql
                result_record["stages"]["selection"]["method"] = "fallback_exception"
    else:
        logger.warning("  [Selection] Skipped (no fixed candidates)")
        result_record["stages"]["selection"]["error"] = "skipped — no fixed candidates"

    # ------------------------------------------------------------------
    # Accuracy Evaluation (reuse pre-computed gold_exec_result)
    # ------------------------------------------------------------------
    final_sql = result_record["final_sql"]
    if final_sql:
        try:
            pred_result = execute_sql(db_path, final_sql)

            if pred_result.success and gold_exec_result.success:
                result_record["correct"] = results_match(pred_result.rows, gold_exec_result.rows)
            elif not pred_result.success:
                result_record["correct"] = False
                if result_record["error_stage"] is None:
                    result_record["error_stage"] = "evaluation_pred_failed"
            elif not gold_exec_result.success:
                logger.warning(
                    "  [Eval] Gold SQL failed to execute! Error: %s", gold_exec_result.error
                )
                result_record["correct"] = False
                result_record["error_stage"] = "evaluation_gold_failed"
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.error("  [Eval] Execution comparison failed: %s", err_msg)
            result_record["correct"] = False
    else:
        result_record["correct"] = False
        if result_record["error_stage"] is None:
            result_record["error_stage"] = "no_final_sql"

    correct_str = "CORRECT" if result_record["correct"] else "WRONG"
    logger.info(
        "  -> %s | oracle_pre=%s oracle_post=%s selector_match=%s | final_sql: %s...",
        correct_str,
        result_record["oracle_pre_fix"],
        result_record["oracle_post_fix"],
        result_record["selector_matched_oracle"],
        (final_sql or "")[:60],
    )

    return result_record


def _find_winner_generator(final_sql: str, fixed_candidates: list) -> str:
    """Find which generator produced the final SQL."""
    if not final_sql:
        return "N/A"
    for fc in fixed_candidates:
        if fc.final_sql == final_sql:
            return fc.generator_id
    return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("Checkpoint E — End-to-End Component Test (v3)")
    logger.info("=" * 70)
    logger.info("LLM provider: %s", settings.llm_provider)
    logger.info("Model fast: %s", settings.model_fast)
    logger.info("Model powerful: %s", settings.model_powerful)
    logger.info("Model reasoning: %s", settings.model_reasoning)
    logger.info("Cache LLM responses: %s", settings.cache_llm_responses)

    # Load BIRD dev data
    logger.info("Loading BIRD dev data from %s", _DEV_JSON)
    dev_entries = load_bird_split("dev", str(_BASE / "data/bird"))
    logger.info("Total dev questions: %d", len(dev_entries))

    # Stratified sampling
    sampled = stratified_sample_33(dev_entries)
    logger.info("Sampled %d questions (3 per DB, 1 per difficulty)", len(sampled))

    # Show sampling summary
    from collections import Counter
    db_diff_counts = Counter((e.db_id, e.difficulty) for e in sampled)
    logger.info("Sampled distribution:")
    for (db_id, diff), count in sorted(db_diff_counts.items()):
        logger.info("  %s / %s: %d", db_id, diff, count)

    # Load example store (shared across all DBs)
    example_store_faiss = str(_INDICES_DIR / "example_store.faiss")
    example_store_meta = str(_INDICES_DIR / "example_store_metadata.json")
    logger.info("Loading ExampleStore from %s", example_store_faiss)
    example_store = ExampleStore.load(example_store_faiss, example_store_meta)
    logger.info("ExampleStore loaded with %d entries", len(example_store._metadata))

    # Pre-load all artifacts (cache per db_id to avoid repeated I/O)
    db_artifacts: dict[str, dict] = {}
    db_ids_needed = sorted(set(e.db_id for e in sampled))
    logger.info("Pre-loading artifacts for %d databases: %s", len(db_ids_needed), db_ids_needed)
    for db_id in db_ids_needed:
        logger.info("  Loading artifacts for %s...", db_id)
        try:
            db_artifacts[db_id] = load_artifacts(db_id)
        except Exception as exc:
            logger.error("  FAILED to load artifacts for %s: %s", db_id, exc)
            db_artifacts[db_id] = None

    # Process all questions
    all_results = []
    # Each stage has its own per-stage timeout inside process_question().
    # This outer value is a safety net only — it should never fire in practice.
    timeout_per_question = _TIMEOUT_TOTAL_SAFETY

    for idx, entry in enumerate(sampled):
        artifacts = db_artifacts.get(entry.db_id)
        if artifacts is None:
            logger.error(
                "Skipping Q#%d (%s) — artifacts failed to load",
                entry.question_id, entry.db_id,
            )
            all_results.append({
                "db_id": entry.db_id,
                "question_id": entry.question_id,
                "difficulty": entry.difficulty,
                "question": entry.question,
                "evidence": entry.evidence,
                "gold_sql": entry.SQL,
                "stages": {
                    "grounding": {"keywords_extracted": 0, "cell_matches": 0, "few_shot_examples": 0, "error": "artifacts not loaded"},
                    "schema_linking": {"s1_fields": 0, "s2_fields": 0, "error": "artifacts not loaded"},
                    "generation": {"total_candidates": 0, "by_generator": {"reasoning": 0, "standard": 0, "complex": 0, "icl": 0}, "error_flags": 0, "error": "artifacts not loaded"},
                    "fixing": {"needed_fix": 0, "fixed_successfully": 0, "still_failing": 0, "original_success_count": 0, "post_fix_success_count": 0, "error": "artifacts not loaded"},
                    "selection": {"method": "N/A", "winner_generator": "N/A", "cluster_count": 0, "error": "artifacts not loaded"},
                },
                "final_sql": "",
                "correct": False,
                "error_stage": "artifacts",
                "oracle_pre_fix": False, "oracle_pre_fix_count": 0,
                "oracle_post_fix": False, "oracle_post_fix_count": 0,
                "oracle_total_candidates": 0,
                "selector_matched_oracle": False,
                "schema_recall": {},
                "gold_exec_success": False,
            })
            continue

        try:
            result = await asyncio.wait_for(
                process_question(entry, artifacts, example_store, idx, len(sampled)),
                timeout=timeout_per_question,
            )
            all_results.append(result)
        except asyncio.TimeoutError:
            logger.error(
                "Q#%d (%s) hit safety-net timeout after %ds — per-stage timeouts should have fired first",
                entry.question_id, entry.db_id, timeout_per_question,
            )
            all_results.append({
                "db_id": entry.db_id,
                "question_id": entry.question_id,
                "difficulty": entry.difficulty,
                "question": entry.question,
                "evidence": entry.evidence,
                "gold_sql": entry.SQL,
                "stages": {
                    "grounding": {"keywords_extracted": 0, "cell_matches": 0, "few_shot_examples": 0, "error": None},
                    "schema_linking": {"s1_fields": 0, "s2_fields": 0, "error": None},
                    "generation": {"total_candidates": 0, "by_generator": {"reasoning": 0, "standard": 0, "complex": 0, "icl": 0}, "error_flags": 0, "error": None},
                    "fixing": {"needed_fix": 0, "fixed_successfully": 0, "still_failing": 0, "original_success_count": 0, "post_fix_success_count": 0, "error": None},
                    "selection": {"method": "N/A", "winner_generator": "N/A", "cluster_count": 0, "error": "timeout"},
                },
                "final_sql": "",
                "correct": False,
                "error_stage": "timeout",
                "oracle_pre_fix": False, "oracle_pre_fix_count": 0,
                "oracle_post_fix": False, "oracle_post_fix_count": 0,
                "oracle_total_candidates": 0,
                "selector_matched_oracle": False,
                "schema_recall": {},
                "gold_exec_success": False,
            })
        except Exception as exc:
            logger.error(
                "Unexpected error for Q#%d (%s): %s",
                entry.question_id, entry.db_id, exc,
            )
            logger.debug(traceback.format_exc())
            all_results.append({
                "db_id": entry.db_id,
                "question_id": entry.question_id,
                "difficulty": entry.difficulty,
                "question": entry.question,
                "evidence": entry.evidence,
                "gold_sql": entry.SQL,
                "stages": {
                    "grounding": {"keywords_extracted": 0, "cell_matches": 0, "few_shot_examples": 0, "error": str(exc)},
                    "schema_linking": {"s1_fields": 0, "s2_fields": 0, "error": None},
                    "generation": {"total_candidates": 0, "by_generator": {"reasoning": 0, "standard": 0, "complex": 0, "icl": 0}, "error_flags": 0, "error": None},
                    "fixing": {"needed_fix": 0, "fixed_successfully": 0, "still_failing": 0, "original_success_count": 0, "post_fix_success_count": 0, "error": None},
                    "selection": {"method": "N/A", "winner_generator": "N/A", "cluster_count": 0, "error": None},
                },
                "final_sql": "",
                "correct": False,
                "error_stage": "unexpected_exception",
                "oracle_pre_fix": False, "oracle_pre_fix_count": 0,
                "oracle_post_fix": False, "oracle_post_fix_count": 0,
                "oracle_total_candidates": 0,
                "selector_matched_oracle": False,
                "schema_recall": {},
                "gold_exec_success": False,
            })

    elapsed = time.time() - start_time

    # ------------------------------------------------------------------
    # Compute aggregate statistics
    # ------------------------------------------------------------------
    total = len(all_results)
    correct_total = sum(1 for r in all_results if r["correct"])
    accuracy_total = correct_total / total if total > 0 else 0.0

    # By difficulty
    from collections import defaultdict
    diff_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    db_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in all_results:
        diff = r["difficulty"]
        db = r["db_id"]
        diff_stats[diff]["total"] += 1
        db_stats[db]["total"] += 1
        if r["correct"]:
            diff_stats[diff]["correct"] += 1
            db_stats[db]["correct"] += 1

    # Stage failure analysis
    error_stage_counts: dict[str, int] = defaultdict(int)
    for r in all_results:
        if not r["correct"] and r.get("error_stage"):
            error_stage_counts[r["error_stage"]] += 1

    # ---- Oracle aggregate stats ----
    oracle_eligible = [r for r in all_results if r.get("gold_exec_success", False)]
    oracle_pre_fix_count_total = sum(1 for r in oracle_eligible if r.get("oracle_pre_fix", False))
    oracle_post_fix_count_total = sum(1 for r in oracle_eligible if r.get("oracle_post_fix", False))
    n_eligible = len(oracle_eligible)

    # Selector precision: among questions where oracle was achievable after fixing,
    # how often did the selector pick the correct answer?
    oracle_achievable = [r for r in all_results if r.get("oracle_post_fix", False)]
    selector_matched = sum(1 for r in oracle_achievable if r.get("selector_matched_oracle", False))
    n_oracle_achievable = len(oracle_achievable)

    # ---- Query Fixer aggregate stats ----
    total_candidates_all = sum(r["oracle_total_candidates"] for r in all_results)
    total_oracle_pre_fix_candidates = sum(r.get("oracle_pre_fix_count", 0) for r in all_results)
    total_oracle_post_fix_candidates = sum(r.get("oracle_post_fix_count", 0) for r in all_results)
    total_needed_fix = sum(r["stages"]["fixing"].get("needed_fix", 0) for r in all_results)
    total_fixed_ok = sum(r["stages"]["fixing"].get("fixed_successfully", 0) for r in all_results)
    total_still_failing = sum(r["stages"]["fixing"].get("still_failing", 0) for r in all_results)
    total_orig_success = sum(r["stages"]["fixing"].get("original_success_count", 0) for r in all_results)
    total_post_success = sum(r["stages"]["fixing"].get("post_fix_success_count", 0) for r in all_results)

    # ---- Schema Linker aggregate stats ----
    recalls_with_data = [r for r in all_results if r.get("schema_recall")]
    avg_table_recall_s1 = (
        sum(r["schema_recall"].get("table_recall_s1", 0) for r in recalls_with_data)
        / len(recalls_with_data) if recalls_with_data else 0.0
    )
    avg_col_recall_s1 = (
        sum(r["schema_recall"].get("col_recall_s1", 0) for r in recalls_with_data)
        / len(recalls_with_data) if recalls_with_data else 0.0
    )
    avg_table_recall_s2 = (
        sum(r["schema_recall"].get("table_recall_s2", 0) for r in recalls_with_data)
        / len(recalls_with_data) if recalls_with_data else 0.0
    )
    avg_col_recall_s2 = (
        sum(r["schema_recall"].get("col_recall_s2", 0) for r in recalls_with_data)
        / len(recalls_with_data) if recalls_with_data else 0.0
    )
    s1_complete_count = sum(1 for r in recalls_with_data if r["schema_recall"].get("s1_complete", False))
    s2_complete_count = sum(1 for r in recalls_with_data if r["schema_recall"].get("s2_complete", False))

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CHECKPOINT E — RESULTS SUMMARY (v3)")
    print("=" * 80)
    print(f"Total questions: {total}")
    print(f"Correct: {correct_total}/{total} ({accuracy_total:.1%})")
    print(f"Elapsed time: {elapsed:.1f}s ({elapsed/total:.1f}s per question)")

    print("\n--- Accuracy by Difficulty ---")
    for diff in ["simple", "moderate", "challenging"]:
        stats = diff_stats.get(diff, {"correct": 0, "total": 0})
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"  {diff:12s}: {stats['correct']}/{stats['total']} ({acc:.1%})")

    print("\n--- Accuracy by Database ---")
    for db_id in sorted(db_stats.keys()):
        stats = db_stats[db_id]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        bar = "#" * stats["correct"] + "." * (stats["total"] - stats["correct"])
        print(f"  {db_id:35s}: {stats['correct']}/{stats['total']} ({acc:.1%})  [{bar}]")

    print("\n--- Oracle Performance ---")
    print(f"  Questions with gold SQL executable:   {n_eligible}/{total}")
    print(f"  Oracle (pre-fix, any correct cand):  {oracle_pre_fix_count_total}/{n_eligible} ({oracle_pre_fix_count_total/n_eligible:.1%})" if n_eligible else "  N/A")
    print(f"  Oracle (post-fix, any correct cand): {oracle_post_fix_count_total}/{n_eligible} ({oracle_post_fix_count_total/n_eligible:.1%})" if n_eligible else "  N/A")
    print(f"  Selector precision (when oracle achievable): {selector_matched}/{n_oracle_achievable} ({selector_matched/n_oracle_achievable:.1%})" if n_oracle_achievable else "  N/A")

    print("\n--- Query Fixer Performance ---")
    print(f"  Total candidates generated:       {total_candidates_all}")
    print(f"  Original success (pre-fix):       {total_orig_success}/{total_candidates_all} ({total_orig_success/total_candidates_all:.1%})" if total_candidates_all else "  N/A")
    print(f"  Post-fix success:                 {total_post_success}/{total_candidates_all} ({total_post_success/total_candidates_all:.1%})" if total_candidates_all else "  N/A")
    print(f"  Candidates needing fix:           {total_needed_fix}")
    print(f"  Successfully fixed:               {total_fixed_ok}/{total_needed_fix} ({total_fixed_ok/total_needed_fix:.1%})" if total_needed_fix else "  0 candidates needed fixing")
    print(f"  Still failing after fix:          {total_still_failing}/{total_candidates_all}")

    print("\n--- Schema Linker Recall ---")
    print(f"  Avg table recall  — S1: {avg_table_recall_s1:.1%}  S2: {avg_table_recall_s2:.1%}")
    print(f"  Avg column recall — S1: {avg_col_recall_s1:.1%}  S2: {avg_col_recall_s2:.1%}")
    print(f"  S1 complete (all gold tables+cols): {s1_complete_count}/{len(recalls_with_data)}")
    print(f"  S2 complete (all gold tables+cols): {s2_complete_count}/{len(recalls_with_data)}")

    print("\n--- Per-Question Summary ---")
    print(f"  {'#':>3}  {'DB':35s}  {'Q#':>6}  {'Diff':12s}  {'Ground':>7}  {'Schema':>7}  {'Gen':>4}  {'Fix':>4}  {'Sel':>4}  {'OracleP':>7}  {'OracleF':>7}  {'SelMatch':>8}  {'Correct'}")
    print("  " + "-" * 130)
    for idx, r in enumerate(all_results):
        g_ok = "OK" if r["stages"]["grounding"]["error"] is None else "ERR"
        s_ok = "OK" if r["stages"]["schema_linking"]["error"] is None else "ERR"
        gen_ok = "OK" if r["stages"]["generation"]["error"] is None else "ERR"
        fix_ok = "OK" if r["stages"]["fixing"]["error"] is None else "ERR"
        sel_ok = "OK" if r["stages"]["selection"]["error"] is None else "ERR"
        correct_mark = "YES" if r["correct"] else "NO"
        oracle_pre = "Y" if r.get("oracle_pre_fix") else "N"
        oracle_post = "Y" if r.get("oracle_post_fix") else "N"
        sel_match = "Y" if r.get("selector_matched_oracle") else "N"
        print(
            f"  {idx+1:>3}  {r['db_id']:35s}  {r['question_id']:>6}  "
            f"{r['difficulty']:12s}  {g_ok:>7}  {s_ok:>7}  {gen_ok:>4}  "
            f"{fix_ok:>4}  {sel_ok:>4}  {oracle_pre:>7}  {oracle_post:>7}  {sel_match:>8}  {correct_mark}"
        )

    print("\n--- Stage Failure Counts (wrong answers) ---")
    if error_stage_counts:
        for stage, count in sorted(error_stage_counts.items(), key=lambda x: -x[1]):
            print(f"  {stage:40s}: {count}")
    else:
        print("  (no stage failures recorded)")

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = _OUTPUT_DIR / "checkpoint_e_results.json"
    summary = {
        "run_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed, 1),
            "llm_provider": settings.llm_provider,
            "model_fast": settings.model_fast,
            "model_powerful": settings.model_powerful,
            "model_reasoning": settings.model_reasoning,
            "cache_enabled": settings.cache_llm_responses,
            "total_questions": total,
            "script_version": "v3",
        },
        "aggregate": {
            "total": total,
            "correct": correct_total,
            "accuracy": round(accuracy_total, 4),
            "by_difficulty": {
                diff: {
                    "correct": diff_stats[diff]["correct"],
                    "total": diff_stats[diff]["total"],
                    "accuracy": round(diff_stats[diff]["correct"] / diff_stats[diff]["total"], 4)
                    if diff_stats[diff]["total"] > 0 else 0.0,
                }
                for diff in ["simple", "moderate", "challenging"]
            },
            "by_database": {
                db_id: {
                    "correct": db_stats[db_id]["correct"],
                    "total": db_stats[db_id]["total"],
                    "accuracy": round(db_stats[db_id]["correct"] / db_stats[db_id]["total"], 4)
                    if db_stats[db_id]["total"] > 0 else 0.0,
                }
                for db_id in sorted(db_stats.keys())
            },
            "error_stage_counts": dict(error_stage_counts),
            "oracle": {
                "n_gold_executable": n_eligible,
                "oracle_pre_fix": oracle_pre_fix_count_total,
                "oracle_post_fix": oracle_post_fix_count_total,
                "oracle_pre_fix_rate": round(oracle_pre_fix_count_total / n_eligible, 4) if n_eligible else 0.0,
                "oracle_post_fix_rate": round(oracle_post_fix_count_total / n_eligible, 4) if n_eligible else 0.0,
                "selector_precision_when_oracle": round(selector_matched / n_oracle_achievable, 4) if n_oracle_achievable else 0.0,
                "n_oracle_achievable": n_oracle_achievable,
                "selector_matched": selector_matched,
            },
            "fixer": {
                "total_candidates": total_candidates_all,
                "original_success": total_orig_success,
                "post_fix_success": total_post_success,
                "needed_fix": total_needed_fix,
                "fixed_ok": total_fixed_ok,
                "still_failing": total_still_failing,
                "fix_success_rate": round(total_fixed_ok / total_needed_fix, 4) if total_needed_fix else 1.0,
                "pre_oracle_candidates": total_oracle_pre_fix_candidates,
                "post_oracle_candidates": total_oracle_post_fix_candidates,
            },
            "schema_linker": {
                "n_evaluated": len(recalls_with_data),
                "avg_table_recall_s1": round(avg_table_recall_s1, 4),
                "avg_col_recall_s1": round(avg_col_recall_s1, 4),
                "avg_table_recall_s2": round(avg_table_recall_s2, 4),
                "avg_col_recall_s2": round(avg_col_recall_s2, 4),
                "s1_complete_count": s1_complete_count,
                "s2_complete_count": s2_complete_count,
            },
        },
        "questions": all_results,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {results_path}")

    # ------------------------------------------------------------------
    # Generate inspection report
    # ------------------------------------------------------------------
    write_inspection_report(
        summary, all_results, diff_stats, db_stats, error_stage_counts,
        oracle_stats=summary["aggregate"]["oracle"],
        fixer_stats=summary["aggregate"]["fixer"],
        schema_linker_stats=summary["aggregate"]["schema_linker"],
    )

    return summary


def write_inspection_report(
    summary: dict,
    all_results: list,
    diff_stats: dict,
    db_stats: dict,
    error_stage_counts: dict,
    oracle_stats: dict = None,
    fixer_stats: dict = None,
    schema_linker_stats: dict = None,
) -> None:
    """Write the Checkpoint E inspection report as Markdown."""
    report_path = _OUTPUT_DIR / "inspection_report.md"

    total = summary["aggregate"]["total"]
    correct = summary["aggregate"]["correct"]
    accuracy = summary["aggregate"]["accuracy"]

    lines = []
    lines.append("# Checkpoint E — Inspection Report (v3)")
    lines.append("")
    lines.append(f"**Date:** {summary['run_metadata']['timestamp']}")
    lines.append(f"**Script version:** {summary['run_metadata'].get('script_version', 'v3')}")
    lines.append(f"**LLM Provider:** {summary['run_metadata']['llm_provider']}")
    lines.append(f"**Models:** fast={summary['run_metadata']['model_fast']}, powerful={summary['run_metadata']['model_powerful']}, reasoning={summary['run_metadata']['model_reasoning']}")
    lines.append(f"**Cache enabled:** {summary['run_metadata']['cache_enabled']}")
    lines.append(f"**Elapsed time:** {summary['run_metadata']['elapsed_seconds']}s ({summary['run_metadata']['elapsed_seconds']/total:.1f}s/question)")
    lines.append("")

    # 1. Summary
    lines.append("## 1. Summary")
    lines.append("")
    lines.append(f"**Total accuracy:** {correct}/{total} = {accuracy:.1%}")
    lines.append("")
    lines.append("### By Difficulty")
    lines.append("")
    lines.append("| Difficulty | Correct | Total | Accuracy |")
    lines.append("|------------|---------|-------|----------|")
    for diff in ["simple", "moderate", "challenging"]:
        stats = diff_stats.get(diff, {"correct": 0, "total": 0})
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        lines.append(f"| {diff} | {stats['correct']} | {stats['total']} | {acc:.1%} |")
    lines.append("")
    lines.append("### By Database")
    lines.append("")
    lines.append("| Database | Correct | Total | Accuracy |")
    lines.append("|----------|---------|-------|----------|")
    for db_id in sorted(db_stats.keys()):
        stats = db_stats[db_id]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        lines.append(f"| {db_id} | {stats['correct']} | {stats['total']} | {acc:.1%} |")
    lines.append("")

    # 2. Oracle Performance
    lines.append("## 2. Oracle Performance on Generated Candidates")
    lines.append("")
    lines.append(
        "The oracle measures whether the correct SQL was present among the generated candidates "
        "at all — it represents the theoretical upper bound achievable by perfect selection."
    )
    lines.append("")
    if oracle_stats:
        n_elig = oracle_stats.get("n_gold_executable", 0)
        oracle_pre = oracle_stats.get("oracle_pre_fix", 0)
        oracle_post = oracle_stats.get("oracle_post_fix", 0)
        oracle_pre_rate = oracle_stats.get("oracle_pre_fix_rate", 0)
        oracle_post_rate = oracle_stats.get("oracle_post_fix_rate", 0)
        sel_prec = oracle_stats.get("selector_precision_when_oracle", 0)
        n_achievable = oracle_stats.get("n_oracle_achievable", 0)
        sel_matched = oracle_stats.get("selector_matched", 0)

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Questions with executable gold SQL | {n_elig}/{total} |")
        lines.append(f"| Oracle (pre-fix): ≥1 correct candidate | {oracle_pre}/{n_elig} = {oracle_pre_rate:.1%} |")
        lines.append(f"| Oracle (post-fix): ≥1 correct fixed candidate | {oracle_post}/{n_elig} = {oracle_post_rate:.1%} |")
        lines.append(f"| Gap: Oracle pre→post (fixer creates correct) | {oracle_post - oracle_pre} questions |")
        lines.append(f"| **Actual accuracy** | **{correct}/{total} = {accuracy:.1%}** |")
        lines.append(f"| Gap: Oracle post → Actual (selector misses) | {oracle_post - correct} questions |")
        lines.append(f"| Selector precision (when oracle achievable) | {sel_matched}/{n_achievable} = {sel_prec:.1%} |")
        lines.append("")
        lines.append("**Interpretation:**")
        lines.append(f"- Generation upper bound: {oracle_pre_rate:.1%} of questions had at least one correct candidate before fixing.")
        lines.append(f"- After fixing, the oracle rose to {oracle_post_rate:.1%} — the fixer added {oracle_post - oracle_pre} new correct candidates.")
        lines.append(f"- The selector then successfully picked the correct answer {sel_prec:.1%} of the time when it was available.")
        lines.append(f"- Accuracy gap vs oracle: {oracle_post - correct} questions were lost by the selector despite having a correct candidate available.")
        lines.append("")

    # Per-question oracle table
    lines.append("### Per-Question Oracle Detail")
    lines.append("")
    lines.append("| # | DB | Q# | Diff | Cands | OracleP | OraclePcnt | OracleF | OracleFcnt | SelMatch | Correct |")
    lines.append("|---|----|----|------|-------|---------|------------|---------|------------|----------|---------|")
    for idx, r in enumerate(all_results):
        pre_o = "Y" if r.get("oracle_pre_fix") else "N"
        post_o = "Y" if r.get("oracle_post_fix") else "N"
        sel_m = "Y" if r.get("selector_matched_oracle") else "N"
        cor = "YES" if r["correct"] else "NO"
        n_cands = r.get("oracle_total_candidates", 0)
        pre_cnt = r.get("oracle_pre_fix_count", 0)
        post_cnt = r.get("oracle_post_fix_count", 0)
        lines.append(
            f"| {idx+1} | {r['db_id']} | {r['question_id']} | {r['difficulty']} "
            f"| {n_cands} | {pre_o} | {pre_cnt}/{n_cands} | {post_o} | {post_cnt}/{n_cands} | {sel_m} | {cor} |"
        )
    lines.append("")

    # 3. Query Fixer Performance
    lines.append("## 3. Query Fixer Performance")
    lines.append("")
    if fixer_stats:
        total_c = fixer_stats.get("total_candidates", 0)
        orig_s = fixer_stats.get("original_success", 0)
        post_s = fixer_stats.get("post_fix_success", 0)
        needed = fixer_stats.get("needed_fix", 0)
        fixed_ok = fixer_stats.get("fixed_ok", 0)
        still_fail = fixer_stats.get("still_failing", 0)
        fix_rate = fixer_stats.get("fix_success_rate", 0)

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total candidates across all questions | {total_c} |")
        lines.append(f"| Succeeded before any fixing | {orig_s}/{total_c} = {orig_s/total_c:.1%} |" if total_c else "| Succeeded before any fixing | N/A |")
        lines.append(f"| Succeeded after fixing | {post_s}/{total_c} = {post_s/total_c:.1%} |" if total_c else "| Succeeded after fixing | N/A |")
        lines.append(f"| Net new successes from fixer | {post_s - orig_s} candidates |")
        lines.append(f"| Candidates that needed fixing | {needed} |")
        lines.append(f"| Successfully fixed (now succeed) | {fixed_ok}/{needed} = {fix_rate:.1%} |" if needed else "| Successfully fixed | 0/0 = N/A |")
        lines.append(f"| Still failing after fix attempts | {still_fail}/{total_c} = {still_fail/total_c:.1%} |" if total_c else "| Still failing | N/A |")
        lines.append(f"| Oracle candidates pre-fix | {fixer_stats.get('pre_oracle_candidates', 0)} |")
        lines.append(f"| Oracle candidates post-fix | {fixer_stats.get('post_oracle_candidates', 0)} |")
        lines.append("")
        lines.append("**Interpretation:**")
        if needed > 0:
            lines.append(f"- {needed} candidates had execution errors or returned empty results.")
            lines.append(f"- The fixer successfully repaired {fixed_ok} of them ({fix_rate:.1%} fix success rate).")
            improvement = post_s - orig_s
            if improvement > 0:
                lines.append(f"- Net effect: {improvement} additional candidates now execute successfully after fixing.")
            else:
                lines.append(f"- Net effect: no improvement in executable candidates (fixes may have changed SQL but not fixed underlying issues).")
        else:
            lines.append("- No candidates needed fixing (all generated SQL ran without errors).")
        lines.append("")

    # Per-question fixer table
    lines.append("### Per-Question Fixer Detail")
    lines.append("")
    lines.append("| # | DB | Q# | Diff | NeedFix | FixedOK | StillFail | OrigSucc | PostSucc |")
    lines.append("|---|----|----|------|---------|---------|-----------|----------|----------|")
    for idx, r in enumerate(all_results):
        fs = r["stages"]["fixing"]
        lines.append(
            f"| {idx+1} | {r['db_id']} | {r['question_id']} | {r['difficulty']} "
            f"| {fs.get('needed_fix', 0)} | {fs.get('fixed_successfully', 0)} | {fs.get('still_failing', 0)} "
            f"| {fs.get('original_success_count', 0)} | {fs.get('post_fix_success_count', 0)} |"
        )
    lines.append("")

    # 4. Query Selector Performance
    lines.append("## 4. Query Selector Performance")
    lines.append("")
    lines.append(
        "The selector performance measures how accurately the adaptive selector "
        "chooses the correct SQL when a correct candidate exists in the pool."
    )
    lines.append("")

    # Selection method distribution
    sel_methods: dict[str, int] = {}
    for r in all_results:
        method = r["stages"]["selection"]["method"]
        sel_methods[method] = sel_methods.get(method, 0) + 1

    lines.append("### Selection Method Distribution")
    lines.append("")
    lines.append("| Method | Count | % of Total |")
    lines.append("|--------|-------|------------|")
    for method, count in sorted(sel_methods.items(), key=lambda x: -x[1]):
        pct = count / total if total > 0 else 0.0
        lines.append(f"| {method} | {count} | {pct:.1%} |")
    lines.append("")

    # Selector accuracy per method
    lines.append("### Selector Accuracy by Method")
    lines.append("")
    method_accuracy: dict[str, dict] = {}
    for r in all_results:
        method = r["stages"]["selection"]["method"]
        if method not in method_accuracy:
            method_accuracy[method] = {"correct": 0, "total": 0, "oracle_achievable": 0, "oracle_matched": 0}
        method_accuracy[method]["total"] += 1
        if r["correct"]:
            method_accuracy[method]["correct"] += 1
        if r.get("oracle_post_fix"):
            method_accuracy[method]["oracle_achievable"] += 1
            if r.get("selector_matched_oracle"):
                method_accuracy[method]["oracle_matched"] += 1

    lines.append("| Method | Correct | Total | Accuracy | Oracle Achievable | Oracle Matched | Oracle Precision |")
    lines.append("|--------|---------|-------|----------|-------------------|----------------|-----------------|")
    for method, stats in sorted(method_accuracy.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        oa = stats["oracle_achievable"]
        om = stats["oracle_matched"]
        op = om / oa if oa else 0.0
        lines.append(f"| {method} | {stats['correct']} | {stats['total']} | {acc:.1%} | {oa} | {om} | {op:.1%} |")
    lines.append("")

    # Winner generator distribution (for all answers)
    winner_gens: dict[str, int] = {}
    correct_winner_gens: dict[str, int] = {}
    for r in all_results:
        wg = r["stages"]["selection"].get("winner_generator", "unknown")
        winner_gens[wg] = winner_gens.get(wg, 0) + 1
        if r["correct"]:
            correct_winner_gens[wg] = correct_winner_gens.get(wg, 0) + 1

    if winner_gens:
        lines.append("### Winner Generator Distribution (All Selections)")
        lines.append("")
        lines.append("| Generator | Total Wins | Correct Answers | Win Accuracy |")
        lines.append("|-----------|-----------|-----------------|-------------|")
        for gen, cnt in sorted(winner_gens.items(), key=lambda x: -x[1]):
            corr = correct_winner_gens.get(gen, 0)
            gen_acc = corr / cnt if cnt else 0.0
            lines.append(f"| {gen} | {cnt} | {corr} | {gen_acc:.1%} |")
        lines.append("")

    # Selector misses: oracle achievable but wrong answer selected
    selector_misses = [
        r for r in all_results
        if r.get("oracle_post_fix") and not r.get("selector_matched_oracle")
    ]
    if selector_misses:
        lines.append("### Selector Misses (Oracle Achievable but Wrong Answer Selected)")
        lines.append("")
        for r in selector_misses:
            lines.append(f"**Q#{r['question_id']}** ({r['db_id']} / {r['difficulty']})")
            lines.append(f"- Question: {r['question']}")
            lines.append(f"- Selection method: {r['stages']['selection']['method']}")
            lines.append(f"- Winner generator: {r['stages']['selection'].get('winner_generator', 'N/A')}")
            lines.append(f"- Final SQL: `{r['final_sql'][:200] if r['final_sql'] else '(empty)'}`")
            lines.append(f"- Gold SQL: `{r['gold_sql'][:200]}`")
            lines.append(f"- Oracle post-fix count: {r.get('oracle_post_fix_count', 0)} correct candidates available")
            lines.append("")
    else:
        lines.append("_No selector misses — selector picked correctly whenever oracle was achievable._")
        lines.append("")

    # 5. Schema Linker Recall / Accuracy
    lines.append("## 5. Schema Linker Recall and Accuracy")
    lines.append("")
    lines.append(
        "Recall measures what fraction of the tables and columns required by the gold SQL "
        "were present in S1 (precise) and S2 (recall) schemas. "
        "Column recall only counts explicit `table.column` references in the gold SQL."
    )
    lines.append("")
    if schema_linker_stats:
        n_eval = schema_linker_stats.get("n_evaluated", 0)
        lines.append("| Metric | S1 (Precise) | S2 (Recall) |")
        lines.append("|--------|-------------|------------|")
        lines.append(f"| Avg Table Recall | {schema_linker_stats['avg_table_recall_s1']:.1%} | {schema_linker_stats['avg_table_recall_s2']:.1%} |")
        lines.append(f"| Avg Column Recall | {schema_linker_stats['avg_col_recall_s1']:.1%} | {schema_linker_stats['avg_col_recall_s2']:.1%} |")
        lines.append(f"| Questions with complete coverage | {schema_linker_stats['s1_complete_count']}/{n_eval} | {schema_linker_stats['s2_complete_count']}/{n_eval} |")
        lines.append("")

    # Per-question schema recall table
    lines.append("### Per-Question Schema Recall Detail")
    lines.append("")
    lines.append("| # | DB | Q# | Diff | S1 Tables | S1 Cols | S2 Tables | S2 Cols | S1 Complete | S2 Complete | Missing S1 |")
    lines.append("|---|----|----|------|-----------|---------|-----------|---------|-------------|-------------|-----------|")
    for idx, r in enumerate(all_results):
        sr = r.get("schema_recall", {})
        n_t = sr.get("n_required_tables", 0)
        n_c = sr.get("n_required_cols", 0)
        t_s1 = sr.get("tables_in_s1", 0)
        t_s2 = sr.get("tables_in_s2", 0)
        c_s1 = sr.get("cols_in_s1", 0)
        c_s2 = sr.get("cols_in_s2", 0)
        s1_c = "Y" if sr.get("s1_complete") else "N"
        s2_c = "Y" if sr.get("s2_complete") else "N"
        missing = ", ".join(sr.get("missing_tables_in_s1", []) + sr.get("missing_cols_in_s1", []))[:60]
        lines.append(
            f"| {idx+1} | {r['db_id']} | {r['question_id']} | {r['difficulty']} "
            f"| {t_s1}/{n_t} | {c_s1}/{n_c} | {t_s2}/{n_t} | {c_s2}/{n_c} "
            f"| {s1_c} | {s2_c} | {missing} |"
        )
    lines.append("")

    # Schema linker failures: questions where S1 is missing required elements
    s1_incomplete = [
        r for r in all_results
        if r.get("schema_recall") and not r["schema_recall"].get("s1_complete", True)
    ]
    if s1_incomplete:
        lines.append("### S1 Incomplete — Missing Required Schema Elements")
        lines.append("")
        for r in s1_incomplete:
            sr = r["schema_recall"]
            lines.append(f"**Q#{r['question_id']}** ({r['db_id']} / {r['difficulty']}, correct={r['correct']})")
            lines.append(f"- Required tables: {sr.get('required_tables', [])}")
            lines.append(f"- Missing tables in S1: {sr.get('missing_tables_in_s1', [])}")
            lines.append(f"- Required cols: {sr.get('required_cols', [])}")
            lines.append(f"- Missing cols in S1: {sr.get('missing_cols_in_s1', [])}")
            lines.append(f"- S1 fields count: {r['stages']['schema_linking']['s1_fields']}")
            lines.append(f"- S2 fields count: {r['stages']['schema_linking']['s2_fields']}")
            lines.append(f"- Gold SQL: `{r['gold_sql'][:200]}`")
            lines.append("")
    lines.append("")

    # 6. Per-question full results table
    lines.append("## 6. Per-Question Full Results Table")
    lines.append("")
    lines.append("| # | DB | Q# | Difficulty | Grounding | Schema | Gen | Fix | Selection | OracleP | OracleF | SelMatch | Correct |")
    lines.append("|---|----|----|------------|-----------|--------|-----|-----|-----------|---------|---------|----------|---------|")

    for idx, r in enumerate(all_results):
        g_ok = "OK" if r["stages"]["grounding"]["error"] is None else "ERR"
        s_ok = "OK" if r["stages"]["schema_linking"]["error"] is None else "ERR"
        gen_ok = "OK" if r["stages"]["generation"]["error"] is None else "ERR"
        fix_ok = "OK" if r["stages"]["fixing"]["error"] is None else "ERR"
        sel_ok = "OK" if r["stages"]["selection"]["error"] is None else "ERR"
        correct_mark = "YES" if r["correct"] else "NO"
        sel_method = r["stages"]["selection"]["method"]
        oracle_pre = "Y" if r.get("oracle_pre_fix") else "N"
        oracle_post = "Y" if r.get("oracle_post_fix") else "N"
        sel_match = "Y" if r.get("selector_matched_oracle") else "N"
        lines.append(
            f"| {idx+1} | {r['db_id']} | {r['question_id']} | {r['difficulty']} "
            f"| {g_ok} | {s_ok} | {gen_ok} | {fix_ok} | {sel_ok} ({sel_method}) "
            f"| {oracle_pre} | {oracle_post} | {sel_match} | {correct_mark} |"
        )
    lines.append("")

    # 7. Stage failure analysis
    lines.append("## 7. Stage Failure Analysis (Wrong Answers)")
    lines.append("")
    wrong_results = [r for r in all_results if not r["correct"]]
    if wrong_results:
        lines.append(f"Total wrong answers: {len(wrong_results)}/{total}")
        lines.append("")
        lines.append("### Error Stage Distribution")
        lines.append("")
        lines.append("| Stage | Count |")
        lines.append("|-------|-------|")
        for stage, count in sorted(error_stage_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {stage} | {count} |")
        lines.append("")

        lines.append("### Wrong Answer Details")
        lines.append("")
        for r in wrong_results:
            lines.append(f"**Q#{r['question_id']}** ({r['db_id']} / {r['difficulty']})")
            lines.append(f"- Question: {r['question']}")
            lines.append(f"- Error stage: {r.get('error_stage', 'N/A')}")
            lines.append(f"- Oracle pre-fix: {r.get('oracle_pre_fix', False)} ({r.get('oracle_pre_fix_count', 0)} correct candidates)")
            lines.append(f"- Oracle post-fix: {r.get('oracle_post_fix', False)} ({r.get('oracle_post_fix_count', 0)} correct candidates)")
            lines.append(f"- Selector matched oracle: {r.get('selector_matched_oracle', False)}")
            lines.append(f"- Final SQL: `{r['final_sql'][:200] if r['final_sql'] else '(empty)'}`")
            lines.append(f"- Gold SQL: `{r['gold_sql'][:200]}`")

            # Stage-specific details
            g_stage = r["stages"]["grounding"]
            s_stage = r["stages"]["schema_linking"]
            gen_stage = r["stages"]["generation"]
            fix_stage = r["stages"]["fixing"]
            sel_stage = r["stages"]["selection"]

            if g_stage.get("error"):
                lines.append(f"- Grounding error: {g_stage['error']}")
            if s_stage.get("error"):
                lines.append(f"- Schema linking error: {s_stage['error']}")
            if gen_stage.get("error"):
                lines.append(f"- Generation error: {gen_stage['error']}")
            if fix_stage.get("error"):
                lines.append(f"- Fixing error: {fix_stage['error']}")
            if sel_stage.get("error"):
                lines.append(f"- Selection error: {sel_stage['error']}")

            # Schema recall details for wrong answers
            sr = r.get("schema_recall", {})
            if sr and not sr.get("s1_complete"):
                lines.append(f"- S1 schema missing: tables={sr.get('missing_tables_in_s1', [])}, cols={sr.get('missing_cols_in_s1', [])}")

            lines.append("")
    else:
        lines.append("No wrong answers!")
        lines.append("")

    # 8. Interface issues analysis
    lines.append("## 8. Interface Issues Observed")
    lines.append("")

    # Collect interface observations
    gen_errors_total = sum(1 for r in all_results if r["stages"]["generation"]["error"] is not None)
    fix_errors_total = sum(1 for r in all_results if r["stages"]["fixing"]["error"] is not None)
    sel_errors_total = sum(1 for r in all_results if r["stages"]["selection"]["error"] is not None)
    ground_errors_total = sum(1 for r in all_results if r["stages"]["grounding"]["error"] is not None)
    schema_errors_total = sum(1 for r in all_results if r["stages"]["schema_linking"]["error"] is not None)

    lines.append(f"- Grounding stage errors: {ground_errors_total}/{total}")
    lines.append(f"- Schema linking stage errors: {schema_errors_total}/{total}")
    lines.append(f"- Generation stage errors (partial): {gen_errors_total}/{total}")
    lines.append(f"- Fixing stage errors: {fix_errors_total}/{total}")
    lines.append(f"- Selection stage errors: {sel_errors_total}/{total}")
    lines.append("")

    # Collect stage statistics
    avg_cell_matches = sum(r["stages"]["grounding"]["cell_matches"] for r in all_results) / total if total > 0 else 0
    avg_few_shot = sum(r["stages"]["grounding"]["few_shot_examples"] for r in all_results) / total if total > 0 else 0
    avg_s1 = sum(r["stages"]["schema_linking"]["s1_fields"] for r in all_results) / total if total > 0 else 0
    avg_s2 = sum(r["stages"]["schema_linking"]["s2_fields"] for r in all_results) / total if total > 0 else 0
    avg_candidates = sum(r["stages"]["generation"]["total_candidates"] for r in all_results) / total if total > 0 else 0

    lines.append("### Stage Statistics")
    lines.append("")
    lines.append(f"- Avg cell matches per question: {avg_cell_matches:.1f}")
    lines.append(f"- Avg few-shot examples per question: {avg_few_shot:.1f}")
    lines.append(f"- Avg S1 fields: {avg_s1:.1f}")
    lines.append(f"- Avg S2 fields: {avg_s2:.1f}")
    lines.append(f"- Avg total generation candidates: {avg_candidates:.1f}")
    lines.append("")

    # 9. Issues found
    lines.append("## 9. Issues Found")
    lines.append("")
    lines.append("### Priority Legend: P0 = must fix, P1 = important, P2 = nice to have")
    lines.append("")

    # Automatically identify issues from results
    issues = _identify_issues(all_results, diff_stats, error_stage_counts, oracle_stats, schema_linker_stats)
    if issues:
        for issue in issues:
            lines.append(f"**{issue['priority']}: {issue['title']}**")
            lines.append(f"- Description: {issue['description']}")
            lines.append(f"- Suggested fix: {issue['fix']}")
            lines.append("")
    else:
        lines.append("No critical issues identified.")
        lines.append("")

    # 10. Suggested improvements
    lines.append("## 10. Suggested Improvements Before Pipeline Wiring (Prompt 13)")
    lines.append("")
    suggestions = _generate_suggestions(all_results, diff_stats, error_stage_counts, oracle_stats, schema_linker_stats, fixer_stats)
    for s in suggestions:
        lines.append(f"- {s}")
    lines.append("")

    report_content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nInspection report saved to: {report_path}")


def _identify_issues(
    all_results: list,
    diff_stats: dict,
    error_stage_counts: dict,
    oracle_stats: dict = None,
    schema_linker_stats: dict = None,
) -> list:
    """Automatically identify issues from results."""
    issues = []
    total = len(all_results)

    # Check for timeout issues
    timeout_count = sum(1 for r in all_results if r.get("error_stage") == "timeout")
    if timeout_count > 0:
        issues.append({
            "priority": "P0",
            "title": f"Question timeouts: {timeout_count}/{total} questions timed out",
            "description": f"{timeout_count} questions exceeded the 120-second timeout per question.",
            "fix": "Increase per-question timeout, optimize API calls, or add concurrency limits.",
        })

    # Check for high generation error rate
    error_flag_totals = sum(r["stages"]["generation"]["error_flags"] for r in all_results)
    total_candidates = sum(r["stages"]["generation"]["total_candidates"] for r in all_results)
    if total_candidates > 0:
        error_flag_rate = error_flag_totals / total_candidates
        if error_flag_rate > 0.2:
            issues.append({
                "priority": "P1",
                "title": f"High generation error_flag rate: {error_flag_rate:.1%} of candidates have errors",
                "description": f"{error_flag_totals} out of {total_candidates} candidates have error_flag=True.",
                "fix": "Review generator prompts and model temperature settings.",
            })

    # Check for zero-candidate generation
    zero_cand = sum(1 for r in all_results if r["stages"]["generation"]["total_candidates"] == 0)
    if zero_cand > 0:
        issues.append({
            "priority": "P0",
            "title": f"Zero candidates generated: {zero_cand}/{total} questions",
            "description": f"{zero_cand} questions produced no SQL candidates at all.",
            "fix": "Add fallback SQL generation or better error handling.",
        })

    # Check for schema linking failures
    schema_err = sum(1 for r in all_results if r["stages"]["schema_linking"]["error"] is not None)
    if schema_err > 0:
        issues.append({
            "priority": "P0",
            "title": f"Schema linking failures: {schema_err}/{total} questions",
            "description": f"Schema linker raised exceptions for {schema_err} questions.",
            "fix": "Review schema linker error handling and fallback logic.",
        })

    # Check oracle vs actual accuracy gap (selector misses)
    if oracle_stats:
        oracle_post = oracle_stats.get("oracle_post_fix", 0)
        n_elig = oracle_stats.get("n_gold_executable", total)
        correct = sum(1 for r in all_results if r["correct"])
        selector_gap = oracle_post - correct
        if selector_gap > 2:
            issues.append({
                "priority": "P1",
                "title": f"Selector missing {selector_gap} achievable correct answers",
                "description": (
                    f"Oracle post-fix is {oracle_post}/{n_elig} but actual accuracy is {correct}/{total}. "
                    f"{selector_gap} questions had a correct candidate but the selector chose wrong."
                ),
                "fix": "Review tournament prompt quality and cluster comparison logic.",
            })

        sel_prec = oracle_stats.get("selector_precision_when_oracle", 0)
        if sel_prec < 0.7 and oracle_stats.get("n_oracle_achievable", 0) > 3:
            issues.append({
                "priority": "P1",
                "title": f"Low selector precision: {sel_prec:.1%} when oracle achievable",
                "description": "Selector is picking the wrong answer more than 30% of the time when a correct answer exists.",
                "fix": "Review tournament pairwise comparison prompt. Consider using a stronger model for selection.",
            })

    # Check schema linker recall
    if schema_linker_stats:
        col_recall_s1 = schema_linker_stats.get("avg_col_recall_s1", 1.0)
        table_recall_s1 = schema_linker_stats.get("avg_table_recall_s1", 1.0)
        s1_complete = schema_linker_stats.get("s1_complete_count", total)
        n_eval = schema_linker_stats.get("n_evaluated", total)

        if table_recall_s1 < 0.85:
            issues.append({
                "priority": "P0",
                "title": f"Low S1 table recall: {table_recall_s1:.1%}",
                "description": "Schema linker S1 is missing required tables from gold SQL.",
                "fix": "Increase FAISS top_k, review S1 pruning logic, or ensure PK/FK auto-add works correctly.",
            })
        elif table_recall_s1 < 0.95:
            issues.append({
                "priority": "P1",
                "title": f"Imperfect S1 table recall: {table_recall_s1:.1%}",
                "description": "S1 occasionally misses required tables.",
                "fix": "Review edge cases in schema linking (e.g. aliases, junction tables).",
            })

        if col_recall_s1 < 0.75:
            issues.append({
                "priority": "P1",
                "title": f"Low S1 column recall: {col_recall_s1:.1%}",
                "description": "S1 schema is missing many required columns from gold SQL.",
                "fix": "Investigate S1 over-pruning. Consider making S1 more inclusive.",
            })

        s1_incomplete_rate = 1.0 - (s1_complete / n_eval) if n_eval else 0.0
        if s1_incomplete_rate > 0.3:
            issues.append({
                "priority": "P1",
                "title": f"S1 schema incomplete for {int(s1_incomplete_rate * 100)}% of questions",
                "description": f"S1 is missing at least one required table or column for {total - s1_complete}/{total} questions.",
                "fix": "Schema linker needs to be more conservative in S1 pruning.",
            })

    # Check for challenging difficulty issues
    challenging_stats = diff_stats.get("challenging", {"correct": 0, "total": 0})
    if challenging_stats["total"] > 0:
        challenging_acc = challenging_stats["correct"] / challenging_stats["total"]
        if challenging_acc < 0.3:
            issues.append({
                "priority": "P1",
                "title": f"Low accuracy on challenging questions: {challenging_acc:.1%}",
                "description": "Challenging questions have significantly lower accuracy than simple/moderate.",
                "fix": "Review reasoning generator prompt and extended thinking token budget.",
            })

    # Check if selection method distribution is problematic
    all_sel_methods = [r["stages"]["selection"]["method"] for r in all_results]
    fallback_count = sum(1 for m in all_sel_methods if "fallback" in m)
    if fallback_count > total * 0.3:
        issues.append({
            "priority": "P1",
            "title": f"High selection fallback rate: {fallback_count}/{total}",
            "description": "More than 30% of questions fall back to the simple fallback selection (not tournament).",
            "fix": "Review fixing stage — too many candidates may be failing execution.",
        })

    # Check for empty final SQL
    empty_sql = sum(1 for r in all_results if not r["final_sql"])
    if empty_sql > 0:
        issues.append({
            "priority": "P0",
            "title": f"Empty final SQL: {empty_sql}/{total} questions",
            "description": f"{empty_sql} questions produced no final SQL at all.",
            "fix": "Add guaranteed fallback SQL (e.g. SELECT * from first table).",
        })

    return issues


def _generate_suggestions(
    all_results: list,
    diff_stats: dict,
    error_stage_counts: dict,
    oracle_stats: dict = None,
    schema_linker_stats: dict = None,
    fixer_stats: dict = None,
) -> list:
    """Generate improvement suggestions for Prompt 13."""
    suggestions = []
    total = len(all_results)

    # Check accuracy
    correct = sum(1 for r in all_results if r["correct"])
    accuracy = correct / total if total > 0 else 0.0
    if accuracy < 0.5:
        suggestions.append(
            f"Overall accuracy ({accuracy:.1%}) is below 50%. Consider reviewing the schema linker and "
            "generator prompts before wiring the full online pipeline."
        )
    elif accuracy < 0.68:
        suggestions.append(
            f"Overall accuracy ({accuracy:.1%}) is below the 68% target. "
            "Review wrong answers to identify patterns before wiring Prompt 13."
        )
    else:
        suggestions.append(
            f"Overall accuracy ({accuracy:.1%}) meets or approaches the 68% target. "
            "Proceed to Prompt 13 pipeline wiring."
        )

    # Oracle-based suggestions
    if oracle_stats:
        oracle_pre_rate = oracle_stats.get("oracle_pre_fix_rate", 0.0)
        oracle_post_rate = oracle_stats.get("oracle_post_fix_rate", 0.0)
        sel_prec = oracle_stats.get("selector_precision_when_oracle", 0.0)

        if oracle_pre_rate < 0.6:
            suggestions.append(
                f"Generator oracle pre-fix ({oracle_pre_rate:.1%}) is low. "
                "The generators are not producing correct candidates for many questions. "
                "Consider reviewing generator prompts, schema quality (S1/S2), and cell matching."
            )

        if oracle_post_rate - oracle_pre_rate > 0.05:
            suggestions.append(
                f"The query fixer adds {(oracle_post_rate - oracle_pre_rate):.1%} oracle coverage — "
                "it's actively helping. Consider increasing β (fix iterations) from 2 to 3."
            )

        if sel_prec < 0.75 and oracle_stats.get("n_oracle_achievable", 0) > 3:
            suggestions.append(
                f"Selector precision ({sel_prec:.1%}) is below 75% when oracle is achievable. "
                "Review the tournament comparison prompt and consider: "
                "(1) using a stronger model (model_powerful) for selection, "
                "(2) adding execution result statistics to the comparison prompt."
            )

    # Schema linker recall suggestions
    if schema_linker_stats:
        col_s1 = schema_linker_stats.get("avg_col_recall_s1", 1.0)
        table_s1 = schema_linker_stats.get("avg_table_recall_s1", 1.0)
        s1_complete = schema_linker_stats.get("s1_complete_count", total)
        n_eval = schema_linker_stats.get("n_evaluated", total)

        if table_s1 < 0.90 or col_s1 < 0.80:
            suggestions.append(
                f"Schema linker S1 has imperfect recall (table={table_s1:.1%}, col={col_s1:.1%}). "
                f"S1 is complete for only {s1_complete}/{n_eval} questions. "
                "Consider: (1) relaxing S1 pruning threshold, (2) ensuring join tables are always included, "
                "(3) adding explicit PK/FK pair inclusion."
            )

    # Fixer suggestions
    if fixer_stats:
        fix_rate = fixer_stats.get("fix_success_rate", 1.0)
        needed = fixer_stats.get("needed_fix", 0)
        total_c = fixer_stats.get("total_candidates", 0)

        if needed > 0 and fix_rate < 0.5:
            suggestions.append(
                f"Query fixer success rate is low ({fix_rate:.1%}). "
                f"{needed} candidates needed fixing but only {fixer_stats.get('fixed_ok', 0)} were fixed. "
                "Review fix prompt construction — it may need more context about the error."
            )

    # Error stage suggestions
    if "timeout" in error_stage_counts and error_stage_counts["timeout"] > 2:
        suggestions.append(
            "Several questions timed out. Consider running generators concurrently with a semaphore "
            "limit to reduce API rate-limit stalls."
        )

    # Schema linking suggestion if S1/S2 are consistently empty
    avg_s1 = sum(r["stages"]["schema_linking"]["s1_fields"] for r in all_results) / total if total > 0 else 0
    if avg_s1 < 3:
        suggestions.append(
            "Average S1 fields is very low (<3). Schema linker may be over-pruning. "
            "Consider increasing faiss_top_k or reducing S1 hard cap."
        )

    # ICL generator suggestion
    icl_errors = sum(
        r["stages"]["generation"]["total_candidates"] - r["stages"]["generation"]["by_generator"]["icl"]
        for r in all_results
        if r["stages"]["generation"]["by_generator"]["icl"] == 0
        and r["stages"]["generation"]["total_candidates"] > 0
    )
    if icl_errors > 3:
        suggestions.append(
            "ICL generator produced 0 candidates for several questions. "
            "Verify that the ExampleStore is populated with training examples and that "
            "same-db filtering is not over-aggressive."
        )

    suggestions.append(
        "Run the full online_pipeline.py integration (Prompt 13) once the component interfaces "
        "are confirmed correct from this checkpoint."
    )

    suggestions.append(
        "Consider adding a configurable semaphore (e.g. max_concurrent_questions=3) in the "
        "online pipeline to prevent API rate limit errors during batch evaluation."
    )

    return suggestions


if __name__ == "__main__":
    asyncio.run(main())
