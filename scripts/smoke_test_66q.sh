#!/usr/bin/env bash
# =============================================================================
# NL2SQL Smoke Test — 66 Questions (6 per each of 11 BIRD dev databases)
#
# This script runs the full pipeline on 66 stratified questions and produces
# comprehensive log and result files for post-run analysis.
#
# Usage:
#   cd /Users/mostafa/Documents/workplace/NL2SQL
#   bash scripts/smoke_test_66q.sh                  # fresh run, default dir
#   bash scripts/smoke_test_66q.sh --timestamp      # timestamped output dir
#   bash scripts/smoke_test_66q.sh --resume         # resume a partial run
#   bash scripts/smoke_test_66q.sh --workers 3      # 3 concurrent questions
#   bash scripts/smoke_test_66q.sh --output_dir results/my_run
#
# Output directory (default: results/smoke_test_66q):
#   results.json            — per-question results (EvaluationEntry-compatible)
#   detailed_traces.json    — full per-question traces with every op's inputs/outputs
#   component_summary.json  — aggregated metrics per pipeline component (Op5–Op9)
#   failed_questions.json   — only the incorrect answers with diagnostic info
#   smoke_test.log          — complete structured log at DEBUG level
#
# After the run, analyze results with:
#   python scripts/analyze_results.py results/smoke_test_66q/results.json
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Script config
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/run_smoke_test.py"
DEFAULT_OUTPUT_DIR="$REPO_ROOT/results/smoke_test_66q"
ANALYZE_SCRIPT="$REPO_ROOT/scripts/analyze_results.py"
ENV_FILE="$REPO_ROOT/.env"

# Default values for options (overridden by CLI flags below)
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
WORKERS=2
RESUME_FLAG=""
TIMESTAMP_FLAG=""
LOG_LEVEL="INFO"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --resume)
            RESUME_FLAG="--resume"
            shift
            ;;
        --timestamp)
            TIMESTAMP_FLAG="--timestamp"
            shift
            ;;
        --log_level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --help|-h)
            head -30 "${BASH_SOURCE[0]}" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash scripts/smoke_test_66q.sh [--output_dir DIR] [--workers N] [--resume] [--timestamp] [--log_level LEVEL]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "  NL2SQL Phase 1 — 66-Question Smoke Test"
echo "=================================================================="
echo "  Repo     : $REPO_ROOT"
echo "  Output   : $OUTPUT_DIR"
echo "  Workers  : $WORKERS"
echo "  Resume   : ${RESUME_FLAG:-no}"
echo "  Log level: $LOG_LEVEL"
echo "  Time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="
echo ""

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "[1/6] Pre-flight checks …"

# Python
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "ERROR: python3 / python not found in PATH"
    exit 1
fi
PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" &>/dev/null || PYTHON=python

echo "  Python : $($PYTHON --version 2>&1)"

# Check .env file
if [[ ! -f "$ENV_FILE" ]]; then
    echo "  WARNING: .env not found at $ENV_FILE — using environment variables"
else
    echo "  .env   : $ENV_FILE (found)"
fi

# Check ANTHROPIC_API_KEY or GEMINI_API_KEY
if [[ -z "${ANTHROPIC_API_KEY:-}" ]] && [[ -z "${GEMINI_API_KEY:-}" ]]; then
    # Try loading from .env
    if [[ -f "$ENV_FILE" ]]; then
        if grep -q "ANTHROPIC_API_KEY" "$ENV_FILE" || grep -q "GEMINI_API_KEY" "$ENV_FILE"; then
            echo "  API key: found in .env (will be loaded by pydantic-settings)"
        else
            echo "  ERROR: No ANTHROPIC_API_KEY or GEMINI_API_KEY found in .env or environment"
            exit 1
        fi
    else
        echo "  ERROR: No API key set. Set ANTHROPIC_API_KEY or GEMINI_API_KEY in .env or environment"
        exit 1
    fi
else
    echo "  API key: found in environment"
fi

# Check BIRD dev data
BIRD_DEV_JSON="$REPO_ROOT/data/bird/dev/dev.json"
if [[ ! -f "$BIRD_DEV_JSON" ]]; then
    echo "  ERROR: BIRD dev data not found at $BIRD_DEV_JSON"
    echo "  Run the offline preprocessing first:"
    echo "    python scripts/run_offline_preprocessing.py --split dev --step all"
    exit 1
fi
echo "  BIRD dev: $BIRD_DEV_JSON (found)"

# Check preprocessed artifacts
INDICES_DIR="$REPO_ROOT/data/preprocessed/indices"
SCHEMAS_DIR="$REPO_ROOT/data/preprocessed/schemas"
if [[ ! -d "$INDICES_DIR" ]] || [[ -z "$(ls -A "$INDICES_DIR" 2>/dev/null)" ]]; then
    echo "  ERROR: Preprocessed indices not found at $INDICES_DIR"
    echo "  Run: python scripts/run_offline_preprocessing.py --split dev --step all"
    exit 1
fi
echo "  Indices : $INDICES_DIR (found)"

# Check example store
if [[ ! -f "$INDICES_DIR/example_store.faiss" ]]; then
    echo "  WARNING: example_store.faiss not found — few-shot examples will be empty"
fi

echo "  All pre-flight checks passed."
echo ""

# ---------------------------------------------------------------------------
# Create output directory
# ---------------------------------------------------------------------------
echo "[2/6] Preparing output directory …"
mkdir -p "$OUTPUT_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo ""

# ---------------------------------------------------------------------------
# Run the smoke test
# ---------------------------------------------------------------------------
echo "[3/6] Running 66-question smoke test …"
echo "  This takes approximately 60–120 minutes depending on API latency."
echo "  Progress is shown inline. Results are saved after each question."
echo "  You can safely Ctrl+C and resume later with --resume."
echo ""

START_TIME=$(date +%s)

cd "$REPO_ROOT"

# If --timestamp was passed, let the Python script pick a timestamped directory
if [[ -n "$TIMESTAMP_FLAG" ]]; then
    $PYTHON "$SCRIPT" \
        --workers "$WORKERS" \
        --log_level "$LOG_LEVEL" \
        $TIMESTAMP_FLAG \
        ${RESUME_FLAG:+"$RESUME_FLAG"}
else
    $PYTHON "$SCRIPT" \
        --output_dir "$OUTPUT_DIR" \
        --workers "$WORKERS" \
        --log_level "$LOG_LEVEL" \
        ${RESUME_FLAG:+"$RESUME_FLAG"}
fi

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

echo ""
if [[ $EXIT_CODE -ne 0 ]]; then
    echo "=================================================================="
    echo "  ERROR: Smoke test script exited with code $EXIT_CODE"
    echo "  Check the log at: $OUTPUT_DIR/smoke_test.log"
    echo "  To resume: bash scripts/smoke_test_66q.sh --resume"
    echo "=================================================================="
    exit $EXIT_CODE
fi

echo "=================================================================="
echo "  Smoke test completed in ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "=================================================================="
echo ""

# ---------------------------------------------------------------------------
# Verify output files exist
# ---------------------------------------------------------------------------
echo "[4/6] Verifying output files …"

EXPECTED_FILES=(
    "$OUTPUT_DIR/results.json"
    "$OUTPUT_DIR/detailed_traces.json"
    "$OUTPUT_DIR/component_summary.json"
    "$OUTPUT_DIR/failed_questions.json"
    "$OUTPUT_DIR/smoke_test.log"
)
ALL_OK=true
for f in "${EXPECTED_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        SIZE=$(wc -c < "$f")
        echo "  ✓ $(basename "$f")  (${SIZE} bytes)"
    else
        echo "  ✗ $(basename "$f")  MISSING"
        ALL_OK=false
    fi
done

if [[ "$ALL_OK" != "true" ]]; then
    echo ""
    echo "  WARNING: Some output files are missing. The run may have failed partially."
fi
echo ""

# ---------------------------------------------------------------------------
# Quick result count check
# ---------------------------------------------------------------------------
echo "[5/6] Quick sanity check on results.json …"
RESULTS_FILE="$OUTPUT_DIR/results.json"
if [[ -f "$RESULTS_FILE" ]]; then
    # Count entries using Python (handles edge cases better than jq)
    N_RESULTS=$($PYTHON -c "import json; d=json.load(open('$RESULTS_FILE')); print(len(d))" 2>/dev/null || echo "?")
    N_CORRECT=$($PYTHON -c "import json; d=json.load(open('$RESULTS_FILE')); print(sum(1 for r in d if r.get('correct')))" 2>/dev/null || echo "?")
    echo "  Questions evaluated: $N_RESULTS / 66"
    if [[ "$N_RESULTS" != "?" ]] && [[ "$N_CORRECT" != "?" ]]; then
        echo "  Correct answers    : $N_CORRECT / $N_RESULTS"
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# Run analyze_results.py for the full breakdown
# ---------------------------------------------------------------------------
echo "[6/6] Generating analysis report …"
if [[ -f "$RESULTS_FILE" ]]; then
    echo ""
    $PYTHON "$ANALYZE_SCRIPT" "$RESULTS_FILE" 2>/dev/null || {
        echo "  (analyze_results.py failed — check the log for details)"
    }
else
    echo "  Skipping analysis — results.json not found"
fi

# ---------------------------------------------------------------------------
# Final instructions
# ---------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "  Done! Output files:"
echo "    results.json           → Per-question accuracy + metadata"
echo "    detailed_traces.json   → Full op5–op9 traces per question"
echo "    component_summary.json → Aggregated component metrics"
echo "    failed_questions.json  → Incorrect answers with diagnostics"
echo "    smoke_test.log         → Full structured log"
echo ""
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "  Next steps:"
echo "    # Re-analyze results at any time:"
echo "    python scripts/analyze_results.py $OUTPUT_DIR/results.json"
echo ""
echo "    # Run the pytest e2e tests (loads cached results, very fast):"
echo "    pytest tests/e2e/test_bird_mini.py -v -m live"
echo ""
echo "    # Inspect detailed trace for a specific question (e.g. Q#42):"
echo "    python -c \""
echo "      import json"
echo "      traces = json.load(open('$OUTPUT_DIR/detailed_traces.json'))"
echo "      q = next((t for t in traces if t['question_id']==42), None)"
echo "      print(json.dumps(q, indent=2))"
echo "    \""
echo "=================================================================="
echo ""
