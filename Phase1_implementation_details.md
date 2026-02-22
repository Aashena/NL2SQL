# Phase 1 Implementation Details: API-Only NL2SQL Baseline

## Document Purpose

This document provides a step-by-step implementation guide for Phase 1 of the Adaptive Hybrid NL2SQL system. Phase 1 implements an API-only baseline using primarily the Claude API, covering Operations 0, 1, 5, 6, 7, 8, and 9 from the method plan. No local model training is required in this phase.

**Expected outcome:** ~70–72% Execution Accuracy (EX) on the BIRD dev set (1,534 questions), competitive with state-of-the-art methods that require multi-GPU training clusters.

---

## Design Adjustments for Phase 1

The following adjustments are made to the original method plan and implementation guide, reflecting the Claude-primary API constraint and opportunities for improvement:

### 1. Model Tier Assignment (Provider-Configurable)

Provider is selected via `LLM_PROVIDER` env var. Override tiers with `MODEL_FAST`, `MODEL_POWERFUL`, `MODEL_REASONING`.

| Task | Tier | Anthropic default | Gemini default | Rationale |
|------|------|-------------------|----------------|-----------|
| Keyword extraction | `model_fast` | `claude-haiku-4-5-20251001` | `gemini-2.5-flash` | Simple extraction, latency-sensitive |
| Field summarization (Op 0) | `model_fast` | `claude-haiku-4-5-20251001` | `gemini-2.5-flash` | Bulk offline task, cost-sensitive |
| Schema linking column selection | `model_powerful` | `claude-sonnet-4-6` | `gemini-2.5-pro` | Requires structured reasoning over metadata |
| Generator A (Reasoning) | `model_reasoning` + thinking | `claude-sonnet-4-6` | `gemini-2.5-flash` | Replicates RL-tuned reasoning generator |
| Generator B1 (Standard) | `model_fast` | `claude-haiku-4-5-20251001` | `gemini-2.5-flash` | Cost-effective, fast, diverse |
| Generator B2 (Complex SQL) | `model_powerful` | `claude-sonnet-4-6` | `gemini-2.5-pro` | Advanced SQL patterns require stronger model |
| Generator C (ICL) | `model_powerful` | `claude-sonnet-4-6` | `gemini-2.5-pro` | Few-shot reasoning benefits from strong model |
| Query fixer | `model_fast` | `claude-haiku-4-5-20251001` | `gemini-2.5-flash` | Error correction is targeted and bounded |
| Pairwise selection | `model_fast` | `claude-haiku-4-5-20251001` | `gemini-2.5-flash` | Comparative judgment at scale |

### 2. New Improvements Over Original Plans
- **Prompt caching**: Use Anthropic's prompt caching (`cache_control`) on large schema/system prompt blocks to cut schema-linking and generation costs by ~60–80% for repeated databases.
- **Structured tool-use output**: Use Claude tool definitions (JSON schema) rather than free-text parsing for all structured outputs (keyword lists, column selections, SQL candidates). This eliminates brittle regex parsing.
- **Evidence-aware keyword extraction**: Extract not just literals but also inferred database identifiers from the evidence field (e.g., if evidence says "use iso_code for country codes", extract the column reference too).
- **Adaptive candidate count**: If unanimous agreement is reached early during generation, skip remaining generator calls (saves 20–40% of generation cost on easy questions).
- **Confidence scoring**: Assign each candidate a confidence score based on: (a) successful execution, (b) non-empty result, (c) result plausibility (size vs. expected). Used as a tiebreaker in selection.
- **Multi-level caching**: Cache at both the operation level (full output) and sub-operation level (individual LLM responses) to support rapid iteration during development.
- **Fallback selection**: If fewer than 2 distinct clusters exist among executable candidates, apply majority voting as fallback instead of invoking the selection tournament.

### 3. Generator Configuration for Phase 1

```
Generator A  — Reasoning (model_reasoning + extended thinking)
  ├── S₁ schema, temp=0.0  → candidate A1
  ├── S₁ schema, temp=0.7  → candidate A2
  ├── S₂ schema, temp=0.0  → candidate A3
  └── S₂ schema, temp=0.7  → candidate A4

Generator B1 — Standard (model_fast, standard prompt)
  ├── S₁ schema            → candidate B1a
  └── S₂ schema            → candidate B1b

Generator B2 — Complex SQL (model_powerful, complex-emphasis prompt)
  ├── S₁ schema            → candidate B2a
  └── S₂ schema            → candidate B2b

Generator C  — ICL (model_powerful, few-shot)
  ├── Direct prompting     → candidate C1
  ├── Chain-of-thought     → candidate C2
  └── (optional) Step-back → candidate C3

Total: 10–11 candidates
```

---

## Project Structure

```
NL2SQL/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py            # Pydantic settings, model names, paths
│   ├── data/
│   │   ├── __init__.py
│   │   ├── bird_loader.py         # BIRD dataset loading and schema parsing
│   │   └── database.py            # SQLite query execution utilities
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── profiler.py            # Op 0a: Statistical database profiling
│   │   ├── summarizer.py          # Op 0b: LLM-based field summarization
│   │   └── schema_formatter.py    # Op 0c: DDL + Markdown schema formatting
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── lsh_index.py           # Op 1a: LSH cell value index
│   │   ├── faiss_index.py         # Op 1b: FAISS semantic field index
│   │   └── example_store.py       # Op 1c: Training example vector store
│   ├── grounding/
│   │   ├── __init__.py
│   │   └── context_grounder.py    # Op 5: Keyword extraction + retrieval
│   ├── schema_linking/
│   │   ├── __init__.py
│   │   └── schema_linker.py       # Op 6: Dual schema filtering
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── base_generator.py      # Shared generation utilities
│   │   ├── reasoning_generator.py # Op 7A: Extended thinking generator
│   │   ├── standard_generator.py  # Op 7B: Standard/complex generators
│   │   └── icl_generator.py       # Op 7C: Few-shot ICL generator
│   ├── fixing/
│   │   ├── __init__.py
│   │   └── query_fixer.py         # Op 8: Execution-guided query fixing
│   ├── selection/
│   │   ├── __init__.py
│   │   └── adaptive_selector.py   # Op 9: Clustering + tournament selection
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── offline_pipeline.py    # Orchestrates Ops 0 + 1
│   │   └── online_pipeline.py     # Orchestrates Ops 5→9 per question
│   ├── cache/
│   │   ├── __init__.py
│   │   └── cache_manager.py       # Disk-based LLM response caching
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluator.py           # EX accuracy computation
│       └── metrics.py             # Additional metrics (VES, error analysis)
├── tests/
│   ├── conftest.py                # Shared fixtures (mini DB, sample questions)
│   ├── unit/
│   │   ├── test_profiler.py
│   │   ├── test_summarizer.py
│   │   ├── test_schema_formatter.py
│   │   ├── test_lsh_index.py
│   │   ├── test_faiss_index.py
│   │   ├── test_example_store.py
│   │   ├── test_context_grounder.py
│   │   ├── test_schema_linker.py
│   │   ├── test_reasoning_generator.py
│   │   ├── test_standard_generator.py
│   │   ├── test_icl_generator.py
│   │   ├── test_query_fixer.py
│   │   ├── test_adaptive_selector.py
│   │   └── test_cache_manager.py
│   ├── integration/
│   │   ├── test_offline_pipeline.py
│   │   ├── test_online_pipeline.py
│   │   └── test_generation_diversity.py
│   └── e2e/
│       ├── test_bird_mini.py      # 50-question smoke test
│       └── test_bird_full.py      # Full 1534-question BIRD dev evaluation
├── scripts/
│   ├── download_bird.py           # Downloads BIRD dataset
│   ├── run_offline_preprocessing.py
│   ├── run_evaluation.py
│   └── analyze_results.py
├── data/
│   ├── bird/                      # BIRD dataset (downloaded)
│   │   ├── train/
│   │   ├── dev/
│   │   └── mini_dev/
│   └── preprocessed/              # Cached artifacts
│       ├── profiles/              # Per-DB statistical profiles
│       ├── summaries/             # Per-DB field summaries
│       ├── schemas/               # Formatted DDL + Markdown schemas
│       └── indices/               # LSH, FAISS, example store files
├── results/                       # Evaluation outputs
├── .env.example
├── pyproject.toml
└── requirements.txt
```

---

## Environment Setup

### Step 1.0 — Initialize Project

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install core dependencies
pip install anthropic>=0.40.0           # Claude API (extended thinking support)
pip install sentence-transformers>=2.7  # Embeddings
pip install faiss-cpu>=1.8             # Vector search
pip install datasketch>=1.6            # LSH (MinHash)
pip install spacy>=3.7                 # NER for skeleton masking
pip install pydantic>=2.0              # Data models and settings
pip install pydantic-settings>=2.0     # Environment-based config
pip install aiohttp>=3.9               # Async HTTP
pip install pytest>=8.0                # Testing
pip install pytest-asyncio>=0.23       # Async test support
pip install pytest-mock>=3.12          # Mocking
pip install python-dotenv>=1.0         # .env loading
pip install pandas>=2.0                # Data manipulation
pip install tqdm>=4.0                  # Progress bars
pip install tenacity>=8.0              # Retry logic for API calls
pip install tiktoken>=0.6              # Token counting

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Step 1.1 — Environment Configuration

Create `.env` (never commit this file):
```
ANTHROPIC_API_KEY=your_key_here
BIRD_DATA_DIR=./data/bird
PREPROCESSED_DIR=./data/preprocessed
CACHE_DIR=./data/cache
LOG_LEVEL=INFO

# LLM Provider: "anthropic" (default) or "gemini"
LLM_PROVIDER=anthropic
GEMINI_API_KEY=your_gemini_key_here

# Optional model tier overrides (leave blank to use provider defaults)
# Anthropic defaults: model_fast=claude-haiku-4-5-20251001, model_powerful=claude-sonnet-4-6
# Gemini defaults:    model_fast=gemini-2.5-flash,           model_powerful=gemini-2.5-pro
# MODEL_FAST=
# MODEL_POWERFUL=
# MODEL_REASONING=
```

Create `src/config/settings.py`:
```python
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Provider selection
    llm_provider: Literal["anthropic", "gemini"] = "anthropic"

    # API keys
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    # Model tiers — leave as "" to use provider defaults (filled by validator)
    # model_fast:      lightweight tasks (summarization, keyword extraction, fixing, selection, B1)
    # model_powerful:  complex reasoning (schema linking, B2, Generator C ICL)
    # model_reasoning: extended thinking (Generator A only)
    model_fast: str = ""
    model_powerful: str = ""
    model_reasoning: str = ""

    # Paths
    bird_data_dir: str = "./data/bird"
    preprocessed_dir: str = "./data/preprocessed"
    cache_dir: str = "./data/cache"

    # Generation
    max_candidates: int = 11
    query_fix_iterations: int = 2
    icl_examples_count: int = 8

    # Schema linking
    faiss_top_k: int = 30          # Fields retrieved before LLM filtering
    lsh_top_k: int = 5             # Cell values per keyword

    # Selection
    fast_path_threshold: int = 1   # Number of clusters for fast path

    @model_validator(mode="after")
    def _apply_model_defaults(self) -> "Settings":
        _DEFAULTS = {
            "anthropic": {"model_fast": "claude-haiku-4-5-20251001",
                          "model_powerful": "claude-sonnet-4-6",
                          "model_reasoning": "claude-sonnet-4-6"},
            "gemini":    {"model_fast": "gemini-2.5-flash",
                          "model_powerful": "gemini-2.5-pro",
                          "model_reasoning": "gemini-2.5-flash"},
        }
        for field, default in _DEFAULTS[self.llm_provider].items():
            if not getattr(self, field):
                setattr(self, field, default)
        return self

    model_config = {"env_file": ".env", "populate_by_name": True}

settings = Settings()
```

All LLM calls go through `src/llm/get_client().generate(...)` — see `src/llm/` for the
provider abstraction. Use `settings.model_fast`, `settings.model_powerful`, or
`settings.model_reasoning` as the `model` argument.

---

## Step 2 — Data Acquisition: BIRD Dataset

### Step 2.1 — Download BIRD

Create `scripts/download_bird.py`:

```python
"""
Download the BIRD dataset from Hugging Face.
- Mini-dev: 500 questions (for fast testing)
- Dev: 1,534 questions (primary evaluation)
- Train: ~9,428 questions (for few-shot example store)
"""
```

**Implementation steps:**

1. Navigate to https://huggingface.co/datasets/xlangai/bird and download:
   - `dev_20240627.zip` (dev split, 1534 questions + databases)
   - `train_20240627.zip` (train split, 9428 questions + databases)
   - `mini_dev_sqlite.zip` (500 question mini-dev for fast testing)
2. Extract to `data/bird/dev/`, `data/bird/train/`, `data/bird/mini_dev/`

**Expected directory structure after extraction:**
```
data/bird/dev/
  ├── dev.json                   # 1534 question-SQL pairs
  ├── dev_databases/             # 11 SQLite databases
  │   ├── california_schools/
  │   │   └── california_schools.sqlite
  │   ├── card_games/
  │   └── ...
  └── dev_tables.json            # Schema metadata

data/bird/train/
  ├── train.json                 # 9428 question-SQL pairs
  ├── train_databases/           # 84 SQLite databases
  └── train_tables.json

data/bird/mini_dev/
  ├── mini_dev_sqlite.json       # 500 questions
  └── databases/
```

**Entry format in dev.json** (each element):
```json
{
  "question_id": 0,
  "db_id": "california_schools",
  "question": "What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
  "evidence": "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`",
  "SQL": "SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE ...",
  "difficulty": "simple"
}
```

### Step 2.2 — BIRD Loader

Create `src/data/bird_loader.py` with:
- `load_bird_split(split: str) -> list[BirdEntry]` — loads train/dev/mini_dev
- `load_schema(db_id: str, databases_dir: str) -> DatabaseSchema` — parses table structure
- `BirdEntry` pydantic model with all fields
- `DatabaseSchema` pydantic model with tables, columns, primary keys, foreign keys

### Step 2.3 — Database Execution Utility

Create `src/data/database.py` with:
- `execute_sql(db_path: str, sql: str, timeout: float = 30.0) -> ExecutionResult`
- `ExecutionResult` dataclass: `success: bool`, `rows: list`, `error: str | None`, `execution_time: float`
- Handle: syntax errors, timeout, empty results, runtime errors

---

## Operation 0: Database Profiling + LLM Summarization

**File:** `src/preprocessing/profiler.py`, `src/preprocessing/summarizer.py`, `src/preprocessing/schema_formatter.py`

**Goal:** For each database, produce:
1. Per-field statistical profiles
2. Per-field short and long natural language summaries
3. Two schema representations: DDL (for reasoning generators) and Markdown (for standard/ICL generators)

---

### Sub-Operation 0a: Statistical Profiling

**File:** `src/preprocessing/profiler.py`

**Implementation steps:**

1. **Connect to SQLite database** and enumerate all tables and columns using `SELECT name FROM sqlite_master WHERE type='table'` and `PRAGMA table_info(table_name)`.

2. **For each column, compute:**
   - `total_count`: total number of rows
   - `null_count`: number of NULL values
   - `null_rate`: null_count / total_count
   - `distinct_count`: approximate distinct values (use `SELECT COUNT(DISTINCT col) FROM table`)
   - `data_type`: SQLite type affinity (TEXT, INTEGER, REAL, BLOB, NUMERIC)
   - `sample_values`: top-10 most frequent non-NULL values with frequency counts
   - `min_value`, `max_value`: for numeric/date columns
   - `avg_value`, `std_value`: for numeric columns
   - `avg_length`, `max_length`: for text columns
   - `minhash_sketch`: LSH MinHash sketch for approximate deduplication (using `datasketch.MinHash` with 128 permutations)

3. **Compute table-level metadata:**
   - Total row count
   - Primary keys (from `PRAGMA table_info`)
   - Foreign keys (from `PRAGMA foreign_key_list`)

4. **Output:** `DatabaseProfile` dataclass serializable to JSON, cached at `data/preprocessed/profiles/{db_id}.json`.

**Data structure:**
```python
@dataclass
class ColumnProfile:
    table_name: str
    column_name: str
    data_type: str
    total_count: int
    null_count: int
    null_rate: float
    distinct_count: int
    sample_values: list[tuple[Any, int]]  # (value, frequency)
    min_value: Any | None
    max_value: Any | None
    avg_value: float | None
    avg_length: float | None
    is_primary_key: bool
    foreign_key_ref: str | None  # "other_table.other_column"
    minhash_bands: list[int]     # serialized minhash bands

@dataclass
class DatabaseProfile:
    db_id: str
    tables: list[str]
    columns: list[ColumnProfile]
    foreign_keys: list[tuple[str, str, str, str]]  # (from_table, from_col, to_table, to_col)
    total_tables: int
    total_columns: int
```

#### Tests for Sub-Operation 0a

**File:** `tests/unit/test_profiler.py`

**Test setup:** Create a fixture with a small in-memory SQLite database:
```sql
CREATE TABLE students (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  age INTEGER,
  gpa REAL,
  country TEXT
);
INSERT INTO students VALUES (1, 'Alice', 20, 3.9, 'USA');
INSERT INTO students VALUES (2, 'Bob', 22, NULL, 'UK');
INSERT INTO students VALUES (3, 'Alice', 21, 3.5, 'USA');
INSERT INTO students VALUES (4, NULL, 25, 2.8, NULL);
```

**Tests to write:**

1. `test_column_count_matches_schema` — profiler returns one ColumnProfile per column defined in the table.
2. `test_null_rate_computation` — `age` has null_rate=0.0 (all filled), `name` has null_rate=0.25 (one NULL), `gpa` has null_rate=0.25.
3. `test_distinct_count` — `name` has distinct_count=2 (Alice appears twice), `id` has distinct_count=4.
4. `test_sample_values_ordering` — sample_values for `name` returns Alice first (frequency=2) then Bob (frequency=1).
5. `test_primary_key_detection` — `id` column has `is_primary_key=True`.
6. `test_numeric_stats` — `age` column has min_value=20, max_value=25, avg computable.
7. `test_foreign_key_detection` — Add a FK relationship; verify `foreign_key_ref` is populated correctly.
8. `test_minhash_generation` — MinHash sketch has the expected number of bands and is deterministic for the same column.
9. `test_empty_table_handling` — Table with 0 rows produces a profile with total_count=0 and empty sample_values.
10. `test_profile_json_serialization` — Round-trip serialize/deserialize a DatabaseProfile to/from JSON without data loss.

**How to run:**
```bash
pytest tests/unit/test_profiler.py -v
```

---

### Sub-Operation 0b: LLM Field Summarization

**File:** `src/preprocessing/summarizer.py`

**Implementation steps:**

1. **Batching strategy:** Group 5–8 column profiles per API call to reduce total calls from ~500 to ~60–100 per database. Each batch should be for the same table to maximize coherence.

2. **Prompt design (using Claude tool-use for structured output):**
   Define a tool `summarize_fields` that accepts a list of column profiles and returns a list of `{column_id, short_summary, long_summary}`.

   - **Short summary** (1–2 sentences): Describes what the field represents semantically. Example: `"Stores the ISO 3166-1 alpha-3 country code identifying the nation associated with each record."`
   - **Long summary** (3–5 sentences): Includes value format patterns, range, common values, what the field is used for in queries, and any domain-specific interpretation. Example: `"This field contains 3-letter uppercase country codes (e.g., 'USA', 'GBR', 'JPN') sourced from ISO 3166-1 alpha-3. Values are always 3 characters long. The most frequent values include 'USA' (1,234 records), 'CHN' (892 records), and 'GBR' (654 records). This column is typically used in WHERE clauses to filter by nationality or joined with a separate country reference table. NULL values represent records where country of origin is unknown."`

3. **System prompt** for the summarization call includes:
   - Database name and domain context (inferred from db_id)
   - Table name and all columns in the table (for context)
   - Column statistics (from ColumnProfile)
   - Top-10 sample values

4. **Model:** `claude-haiku-4-5-20251001` (cost-effective for bulk offline task)

5. **Retry logic:** Use `tenacity` with exponential backoff (3 retries, max 30s wait) for API errors.

6. **Output:** `FieldSummary` objects, cached at `data/preprocessed/summaries/{db_id}.json`.

**Improvement over original plan:** Use Anthropic's prompt caching (`cache_control: {"type": "ephemeral"}`) on the system prompt portion (database schema context) so that all batches for the same table share cached tokens. This reduces cost by ~70% for large databases.

**Data structure:**
```python
@dataclass
class FieldSummary:
    table_name: str
    column_name: str
    short_summary: str
    long_summary: str

@dataclass
class DatabaseSummary:
    db_id: str
    field_summaries: list[FieldSummary]
```

#### Tests for Sub-Operation 0b

**File:** `tests/unit/test_summarizer.py`

**Test setup:** Mock `get_client` in the summarizer module to return an `AsyncMock` LLM client. Use `pytest-mock`'s `mocker.patch("src.preprocessing.summarizer.get_client", return_value=mock_client)` where `mock_client.generate.return_value = LLMResponse(tool_inputs=[{...}])`. All test methods calling `summarize_database` must be `async def` (handled automatically by `asyncio_mode = "auto"`).

**Tests to write:**

1. `test_batching_groups_by_table` — Given columns from 2 tables (4 columns each), verifies the summarizer makes 2 API calls (one per table), not 8.
2. `test_summary_fields_present` — Each returned FieldSummary has non-empty `short_summary` and `long_summary`.
3. `test_summary_length_bounds` — Short summaries are ≤ 200 characters; long summaries are ≤ 1000 characters (enforced via prompt or post-processing).
4. `test_all_columns_covered` — For a 12-column database, all 12 columns have a corresponding FieldSummary (no missing columns).
5. `test_retry_on_api_error` — Mock the API to fail twice then succeed; verify summarizer retries and returns a valid result.
6. `test_cache_hit_skips_api` — If `{db_id}.json` already exists on disk, verify no API call is made.
7. `test_cache_write_on_success` — After successful summarization, verify the JSON file is written to disk.
8. `test_tool_use_format_parsing` — Verify the tool-use response from Claude is correctly parsed into FieldSummary objects even with varying JSON structures.
9. `test_empty_sample_values_handled` — Column with no sample values (empty table) still gets a valid summary.
10. `test_domain_inference_from_db_id` — Verify the system prompt includes a reasonable domain context when db_id is "california_schools" vs "card_games".

**How to run:**
```bash
pytest tests/unit/test_summarizer.py -v
# With live API (uses real credits):
pytest tests/unit/test_summarizer.py -v -m live
```

---

### Sub-Operation 0c: Schema Formatting

**File:** `src/preprocessing/schema_formatter.py`

**Implementation steps:**

1. **DDL Schema** (for reasoning generators, mimics code pretraining format):
   ```sql
   -- Table: frpm
   -- Description: Free or Reduced Price Meal data for California schools
   CREATE TABLE frpm (
     CDSCode TEXT PRIMARY KEY,  -- County-District-School code, unique identifier for each school. Format: 14-digit numeric string like '01100170109835'
     "Academic Year" TEXT,       -- School year in format 'YYYY-YYYY', e.g., '2014-2015'
     "County Name" TEXT,         -- Name of the county, e.g., 'Alameda', 'Los Angeles'
     "Free Meal Count (K-12)" REAL,  -- Number of students eligible for free meals in grades K-12
     "Enrollment (K-12)" REAL,   -- Total enrollment in grades K-12
     ...
   );
   -- Foreign keys: CDSCode REFERENCES schools(CDSCode)
   -- Example row: ('01100170109835', '2014-2015', 'Alameda', 245, 1000, ...)
   ```

   - Inject **long summaries** as inline comments after each column
   - Add a table-level comment with the table's inferred purpose
   - Include a sample row (first non-null row) at the end as a comment
   - Include all foreign key relationships as trailing comments

2. **Markdown Schema** (for standard/ICL generators):
   ```markdown
   ## Table: frpm
   *Free or Reduced Price Meal data for California schools*

   | Column | Type | Description | Sample Values |
   |--------|------|-------------|---------------|
   | CDSCode | TEXT (PK) | County-District-School 14-digit code identifying each school | 01100170109835, 01611190130401 |
   | Academic Year | TEXT | School year (YYYY-YYYY format) | 2014-2015, 2015-2016 |
   | Free Meal Count (K-12) | REAL | Students eligible for free meals in K-12 | 245, 1892, 0 |
   ...

   **Foreign keys:** CDSCode → schools.CDSCode
   ```

3. **Output:** Two files per database:
   - `data/preprocessed/schemas/{db_id}_ddl.sql`
   - `data/preprocessed/schemas/{db_id}_markdown.md`

#### Tests for Sub-Operation 0c

**File:** `tests/unit/test_schema_formatter.py`

**Tests to write:**

1. `test_ddl_contains_all_tables` — DDL output contains a CREATE TABLE statement for every table in the database.
2. `test_ddl_contains_all_columns` — Every column appears in the DDL with its name and type.
3. `test_ddl_injects_summaries` — Long summaries from FieldSummary objects appear as inline comments.
4. `test_ddl_primary_key_notation` — Primary key columns are marked with PRIMARY KEY constraint.
5. `test_ddl_foreign_key_comments` — Foreign key relationships appear as trailing comments.
6. `test_markdown_has_header_per_table` — Markdown output has one `## Table:` heading per table.
7. `test_markdown_table_row_count` — Markdown table has exactly one row per column.
8. `test_sample_values_truncation` — Sample values longer than 30 characters are truncated with `...` in Markdown format.
9. `test_special_character_escaping` — Column names with spaces or special chars are properly quoted in DDL (`"Free Meal Count (K-12)"`).
10. `test_deterministic_output` — Calling the formatter twice on the same input produces identical output (no randomness).

---

### Offline Preprocessing Script

**File:** `scripts/run_offline_preprocessing.py`

This script runs Operations 0a, 0b, 0c for all databases in the BIRD train and dev sets:

```bash
python scripts/run_offline_preprocessing.py --split dev
python scripts/run_offline_preprocessing.py --split train
```

It should:
- Skip databases already processed (cache hit)
- Show progress bars (tqdm)
- Log cost estimates (token counts × price per token)
- Handle partial failures gracefully (one DB fails → continue with others)

---

## Operation 1: Index Building

**Files:** `src/indexing/lsh_index.py`, `src/indexing/faiss_index.py`, `src/indexing/example_store.py`

**Goal:** Build three complementary indices:
1. LSH Index: approximate string matching for cell values
2. FAISS Semantic Index: vector search over field descriptions
3. Example Vector Store: structural similarity search over training questions

---

### Sub-Operation 1a: LSH Cell Value Index

**File:** `src/indexing/lsh_index.py`

**Implementation steps:**

1. **MinHash LSH** using `datasketch.MinHashLSH` and `datasketch.MinHash`:
   - For each (table, column) pair in the database, collect all distinct non-NULL string values.
   - For each value, create a MinHash object using k-shingling (character 3-grams).
   - Insert into the LSH index with key `"table.column::value"`.
   - Choose LSH parameters: `num_perm=128`, `threshold=0.5` (50% Jaccard similarity to match).

2. **Query interface:** `query(keyword: str, top_k: int = 5) -> list[CellMatch]`
   - Takes a keyword string, queries the LSH index.
   - Returns top-k matches with: `table`, `column`, `matched_value`, `similarity_score`.

3. **Persistence:** Serialize the LSH index to `data/preprocessed/indices/{db_id}_lsh.pkl`.

4. **Per-database:** Build a separate index per database (indices are not shared across databases).

**Data structure:**
```python
@dataclass
class CellMatch:
    table: str
    column: str
    matched_value: str
    similarity_score: float
    exact_match: bool
```

#### Tests for Sub-Operation 1a

**File:** `tests/unit/test_lsh_index.py`

**Test setup:** Use a small database with known cell values:
- Table `countries`: column `country_name` with values ["United States", "United Kingdom", "Germany", "Japan"]
- Table `orders`: column `status` with values ["pending", "completed", "cancelled"]

**Tests to write:**

1. `test_exact_match_retrieval` — Querying "United States" returns a match for `countries.country_name` with similarity=1.0.
2. `test_fuzzy_match_typo` — Querying "Untied States" (typo) still returns `countries.country_name` as top result with similarity > 0.5.
3. `test_no_match_for_unrelated_query` — Querying "xyz123" returns an empty list.
4. `test_cross_column_retrieval` — Querying "pending" returns `orders.status`, not anything from the `countries` table.
5. `test_top_k_limiting` — Querying a common prefix with top_k=2 returns at most 2 results.
6. `test_serialization_roundtrip` — Serialize index to disk, reload, verify same results as before.
7. `test_null_values_excluded` — NULL cell values are not indexed (no query matches NULL).
8. `test_numeric_values_as_strings` — Numeric cell values stored as text (e.g., "2015") are queryable as strings.
9. `test_index_build_speed` — Building index for a table with 10,000 rows completes in under 30 seconds.
10. `test_empty_table_index` — Building index for an empty table completes without errors.

---

### Sub-Operation 1b: FAISS Semantic Index

**File:** `src/indexing/faiss_index.py`

**Implementation steps:**

1. **Embedding model:** `sentence-transformers` with `all-MiniLM-L6-v2` (22M params, 384-dim embeddings, runs on CPU in <1s).

2. **Index content:** For each (table, column) pair, embed the **long summary** from Operation 0b. Store an index of (embedding → column identifier).

3. **FAISS index type:** `faiss.IndexFlatIP` (inner product, normalized embeddings = cosine similarity). For databases with >1000 fields, use `faiss.IndexIVFFlat` for faster queries.

4. **Query interface:** `query(question: str, top_k: int = 30) -> list[FieldMatch]`
   - Embed the query question.
   - Return top-k semantically similar fields.

5. **Persistence:** Save index to `data/preprocessed/indices/{db_id}_faiss.index` and field mapping to `{db_id}_faiss_fields.json`.

**Data structure:**
```python
@dataclass
class FieldMatch:
    table: str
    column: str
    similarity_score: float
    long_summary: str
    short_summary: str
```

#### Tests for Sub-Operation 1b

**File:** `tests/unit/test_faiss_index.py`

**Test setup:** Create 10 mock FieldSummary objects covering 2 tables:
- `sales`: `amount` (financial), `date` (temporal), `customer_id` (FK)
- `customers`: `name` (person name), `country` (geography), `age` (demographic)

**Tests to write:**

1. `test_semantic_retrieval_relevant` — Query "How many sales were made per country?" returns fields from both `sales` and `customers` (join needed).
2. `test_top_k_count` — Querying with top_k=3 returns exactly 3 results.
3. `test_similarity_ordering` — Results are returned in descending order of similarity score.
4. `test_financial_query_retrieves_amount` — Query "What is the total revenue?" returns `sales.amount` in top-3.
5. `test_geographic_query_retrieves_country` — Query "Which countries have customers?" returns `customers.country` as top result.
6. `test_serialization_roundtrip` — Save index, reload, verify query returns identical results.
7. `test_embedding_dimensionality` — Embeddings have the expected dimension (384 for all-MiniLM-L6-v2).
8. `test_index_build_with_single_field` — Building index with only 1 field completes without errors.
9. `test_query_returns_scores_between_0_and_1` — All similarity scores are in [0, 1] (cosine similarity).
10. `test_different_questions_different_rankings` — Two semantically different questions produce different field rankings.

---

### Sub-Operation 1c: Example Vector Store

**File:** `src/indexing/example_store.py`

**Implementation steps:**

1. **Load training data:** Load all entries from `data/bird/train/train.json`.

2. **Skeleton masking:** For each training question, replace entity names and specific values with generic placeholders:
   - Named entities (detected by spaCy NER): `[ENTITY]`
   - Numeric values: `[NUM]`
   - Quoted strings: `[STR]`
   - Example: `"How many students in Alameda County scored above 90?"` → `"How many [ENTITY] in [ENTITY] scored above [NUM]?"`

3. **Embedding:** Embed each skeleton using `all-MiniLM-L6-v2`.

4. **Index:** Build a FAISS index over skeleton embeddings. Store mapping: index_position → `{question, sql, db_id, evidence, skeleton}`.

5. **Query interface:** `query(question: str, top_k: int = 8) -> list[ExampleEntry]`
   - Mask the query question into a skeleton.
   - Find top-k structurally similar training examples.
   - Return original question-SQL pairs (not the skeletons) for use as few-shot examples.

6. **Deduplication:** Exclude training examples from the same `db_id` as the current query (to prevent schema leakage).

7. **Persistence:** Save to `data/preprocessed/indices/example_store.faiss` and `example_store_metadata.json`.

**Data structure:**
```python
@dataclass
class ExampleEntry:
    question_id: int
    db_id: str
    question: str
    evidence: str
    sql: str
    skeleton: str
    similarity_score: float
```

#### Tests for Sub-Operation 1c

**File:** `tests/unit/test_example_store.py`

**Test setup:** Create a mock training set of 50 question-SQL pairs with varied SQL patterns (aggregation, JOIN, subquery, GROUP BY).

**Tests to write:**

1. `test_skeleton_masking_removes_entities` — `"Find students in Alameda County"` → skeleton contains `[ENTITY]`, not "Alameda".
2. `test_skeleton_masking_removes_numbers` — `"Score above 90"` → skeleton contains `[NUM]`, not "90".
3. `test_retrieval_finds_structurally_similar` — A JOIN question retrieves other JOIN examples in top-3, not simple SELECT examples.
4. `test_top_k_count` — Query with top_k=8 returns at most 8 results.
5. `test_db_id_exclusion` — When db_id="california_schools", no returned examples come from california_schools.
6. `test_examples_include_sql` — All returned ExampleEntry objects have non-empty `sql` field.
7. `test_aggregation_query_retrieves_aggregation_examples` — `"Count the number of..."` retrieves examples with COUNT/SUM/AVG in SQL.
8. `test_serialization_roundtrip` — Save store, reload, same results returned.
9. `test_similarity_scores_ordered` — Results are in descending similarity order.
10. `test_empty_training_set_handled` — Building store with 0 training examples raises a clear error (not silent failure).

---

### Index Building Script

**File:** `scripts/run_offline_preprocessing.py` (extend with index building)

```bash
python scripts/run_offline_preprocessing.py --split dev --step indices
```

---

### Integration Test: Offline Pipeline

**File:** `tests/integration/test_offline_pipeline.py`

**Test:** Run the full offline pipeline (Ops 0+1) on one small BIRD database end-to-end:

1. `test_full_offline_pipeline_small_db` — Run profiling, summarization, formatting, and all 3 index builds on `california_schools` database. Assert all output files are created and non-empty.
2. `test_all_output_files_created` — Verify the exact set of files in `preprocessed/` after running the pipeline.
3. `test_pipeline_idempotent` — Running the pipeline twice on the same database produces identical outputs (cache prevents redundant API calls).
4. `test_pipeline_handles_missing_db_gracefully` — Passing a non-existent db_id raises a clear FileNotFoundError.

---

## Operation 5: Context Grounding

**File:** `src/grounding/context_grounder.py`

**Goal:** Given a question Q and evidence E, extract database-specific keywords and retrieve matching cell values + structurally similar few-shot examples.

### Implementation Steps

**Step 5.1 — Keyword Extraction via Claude**

Use `claude-haiku-4-5-20251001` with a structured tool-use call to extract two types of grounding information from (Q, E):

1. **Database literals:** Specific values that likely exist in the database (e.g., "Alameda County", "2014-2015", "USA", "pending").
2. **Schema references:** Column or table names explicitly mentioned in evidence (e.g., if evidence says "use CDSCode", extract "CDSCode").

**Tool definition:**
```python
extract_grounding_tool = {
    "name": "extract_grounding",
    "description": "Extract database literals and schema references from a question and evidence",
    "input_schema": {
        "type": "object",
        "properties": {
            "literals": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific values that might exist in database cells"
            },
            "schema_references": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Column or table names mentioned in the evidence"
            }
        },
        "required": ["literals", "schema_references"]
    }
}
```

**Step 5.2 — LSH Cell Value Retrieval**

For each extracted literal, query the pre-built LSH index to find matching cell values:
- Return top-k=5 matches per keyword
- Deduplicate across keywords (same table.column can appear for multiple keywords)
- Format output: `{"table": "frpm", "column": "County Name", "matched_value": "Alameda", "from_keyword": "Alameda County"}`

**Step 5.3 — Skeleton Masking + Example Retrieval**

- Apply skeleton masking to question Q
- Query the Example Vector Store with top_k=8
- Exclude examples from the same db_id as the current query

**Output:**
```python
@dataclass
class GroundingContext:
    matched_cells: list[CellMatch]     # From LSH retrieval
    schema_hints: list[str]            # Explicit column refs from evidence
    few_shot_examples: list[ExampleEntry]  # From example store
```

#### Tests for Context Grounding

**File:** `tests/unit/test_context_grounder.py`

**Test setup:** Mock the Claude API client and pre-built LSH index.

**Tests to write:**

1. `test_keyword_extraction_from_question` — Question "How many schools in Alameda County?" extracts literal "Alameda County".
2. `test_schema_reference_from_evidence` — Evidence "Eligible free rate = `Free Meal Count` / `Enrollment`" extracts schema refs ["Free Meal Count", "Enrollment"].
3. `test_cell_match_format` — Returned CellMatch objects have non-empty table, column, and matched_value fields.
4. `test_no_keywords_returns_empty_cells` — If extraction returns no literals, LSH query is skipped and matched_cells is empty.
5. `test_few_shot_examples_count` — Returns exactly top_k=8 examples (or fewer if training set is smaller).
6. `test_db_id_excluded_from_examples` — None of the returned examples come from the same database as the query.
7. `test_duplicate_cell_matches_deduplicated` — If two keywords match the same table.column, only one CellMatch is returned.
8. `test_haiku_model_used` — Verify (via mock inspection) that the keyword extraction call uses the haiku model.
9. `test_evidence_none_handled` — Question without evidence (evidence=None or "") still returns a valid GroundingContext.
10. `test_grounding_latency` — With mocked API and pre-loaded index, grounding completes in under 2 seconds.

---

## Operation 6: Adaptive Schema Linking

**File:** `src/schema_linking/schema_linker.py`

**Goal:** Produce two filtered schemas S₁ (precise) and S₂ (recall) from the full database schema, using FAISS pre-filtering + two LLM column selection iterations.

### Implementation Steps

**Step 6.1 — FAISS Pre-filtering**

Query the FAISS semantic index with the question text to retrieve the top-k=30 most semantically relevant fields. This narrows the LLM's attention without LLM cost.

**Step 6.2 — Iteration 1: Precise Selection (S₁)**

Call `claude-sonnet-4-6` with:
- The question and evidence
- The 30 candidate fields with their short summaries (not full schema — reduces tokens)
- Matched cell values from context grounding (helps the model anchor to relevant columns)
- Instruction to select only fields that are **definitely needed**

Use tool-use for structured output:
```python
select_columns_tool = {
    "name": "select_columns",
    "input_schema": {
        "type": "object",
        "properties": {
            "selected_fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string"},
                        "column": {"type": "string"},
                        "reason": {"type": "string"}
                    }
                }
            }
        }
    }
}
```

Automatically add primary keys for all selected tables. Automatically add foreign keys that connect selected tables.

**Step 6.3 — Iteration 2: Recall Expansion (S₂)**

Call `claude-sonnet-4-6` again with:
- The remaining fields (those not selected in S₁)
- Instruction to select fields that **might be needed** (cast wider net)

S₂ = S₁ ∪ newly selected fields ∪ their associated keys.

**Step 6.4 — Schema Rendering**

For each selected schema (S₁, S₂), render:
- A DDL version (subset of the full DDL schema from Operation 0c)
- A Markdown version (subset of the full Markdown schema)

Include only rows for the selected columns plus key columns.

**Step 6.5 — Prompt Caching**

Use `cache_control: {"type": "ephemeral"}` on the portion of the prompt containing the field summaries. Since multiple questions query the same database, the field summary tokens are cached after the first question for that database (~60% cost reduction on schema linking for BIRD dev).

**Output:**
```python
@dataclass
class LinkedSchemas:
    s1_ddl: str          # Precise DDL schema
    s1_markdown: str     # Precise Markdown schema
    s2_ddl: str          # Recall DDL schema
    s2_markdown: str     # Recall Markdown schema
    s1_fields: list[tuple[str, str]]  # (table, column) pairs in S₁
    s2_fields: list[tuple[str, str]]  # (table, column) pairs in S₂
    selection_reasoning: str  # Concatenated reasons from LLM
```

#### Tests for Schema Linking

**File:** `tests/unit/test_schema_linker.py`

**Test setup:** Mock the Claude API and FAISS index. Use a database with 40 columns across 5 tables.

**Tests to write:**

1. `test_s1_subset_of_s2` — Every field in S₁ appears in S₂ (S₁ ⊆ S₂).
2. `test_primary_keys_always_included` — All primary keys for tables referenced in S₁ are present in both schemas.
3. `test_foreign_keys_bridge_tables` — If S₁ includes columns from 2 tables, any foreign keys connecting them are included.
4. `test_s1_smaller_than_s2` — S₁ has fewer fields than S₂ (precision vs. recall trade-off).
5. `test_faiss_pre_filtering_top_k` — Verify FAISS is queried with top_k=30 (not the full field list).
6. `test_two_api_calls_made` — Verify exactly 2 Claude API calls are made for a standard schema linking operation.
7. `test_schema_rendered_as_ddl` — S₁ DDL output is valid SQL DDL syntax (starts with CREATE TABLE).
8. `test_schema_rendered_as_markdown` — S₂ Markdown output contains `##` table headers and `|` column rows.
9. `test_cell_matches_influence_selection` — When a CellMatch points to `table_x.column_y`, that field appears in S₁.
10. `test_prompt_caching_applied` — Verify that the API call includes `cache_control` markers on the field summary block.
11. `test_schema_linking_with_single_table_db` — Works correctly for databases with only 1 table.
12. `test_all_selected_fields_in_original_schema` — No hallucinated table/column names appear in selected fields.

---

## Operation 7: Diverse SQL Generation

**File:** `src/generation/base_generator.py`, `reasoning_generator.py`, `standard_generator.py`, `icl_generator.py`

**Goal:** Produce 10–11 diverse SQL candidate queries using three generator types with different models, prompts, and schema views.

### Shared Base Generator

**File:** `src/generation/base_generator.py`

Implement shared utilities:
- `SQLCandidate` dataclass: `sql`, `generator_id`, `schema_used`, `temperature`, `reasoning_trace`
- `build_generation_prompt(question, evidence, schema, cell_matches, examples)` — shared prompt construction
- `clean_sql(raw_output: str) -> str` — extract SQL from markdown code blocks, strip semicolons, normalize whitespace
- `validate_sql_syntax(sql: str) -> bool` — lightweight syntax check without DB execution (regex-based)

---

### Sub-Operation 7A: Reasoning Generator

**File:** `src/generation/reasoning_generator.py`

**Model:** `claude-sonnet-4-6` with extended thinking (`"thinking": {"type": "enabled", "budget_tokens": 8000}`)

**Implementation steps:**

1. **System prompt (DDL-focused):**
   ```
   You are an expert SQL analyst. Analyze the database schema carefully, think through join paths
   and value matching step by step, then write a precise SQL query.

   Database schema (DDL format):
   {ddl_schema}

   Matched cell values from database:
   {cell_matches}
   ```

2. **User prompt:**
   ```
   Question: {question}
   Evidence: {evidence}

   Think carefully about:
   1. Which tables and columns are needed
   2. How tables should be joined
   3. Any WHERE conditions based on matched values
   4. Aggregations or subqueries needed

   Write the SQL query.
   ```

3. **Generation calls (4 candidates):**
   - Call A1: S₁ DDL schema, extended thinking enabled, temperature=0 (greedy)
   - Call A2: S₁ DDL schema, extended thinking enabled, temperature=0.7
   - Call A3: S₂ DDL schema, extended thinking enabled, temperature=0
   - Call A4: S₂ DDL schema, extended thinking enabled, temperature=0.7

4. **Post-processing:** Extract SQL from the thinking trace's final answer section. Use `clean_sql()` to normalize.

5. **Note on extended thinking:** Extended thinking forces `temperature=1` in the API, so temperature variation is achieved through prompt variation (with/without CoT instruction) rather than the temperature parameter directly. Adjust calls A1-A4 accordingly: A1 and A3 use minimal prompts, A2 and A4 use more elaborate step-by-step guidance.

#### Tests for Reasoning Generator

**File:** `tests/unit/test_reasoning_generator.py`

**Tests to write:**

1. `test_generates_4_candidates` — Returns exactly 4 SQLCandidate objects.
2. `test_candidates_have_different_schemas` — 2 candidates use S₁ DDL, 2 use S₂ DDL.
3. `test_sql_extracted_from_response` — Raw Claude response with markdown code blocks is correctly stripped to plain SQL.
4. `test_extended_thinking_enabled_in_api_call` — Verify API call includes `"thinking": {"type": "enabled"}` parameter.
5. `test_generator_id_label` — All returned candidates have `generator_id` starting with "reasoning_".
6. `test_sonnet_model_used` — API calls use `claude-sonnet-4-6` (not haiku).
7. `test_cell_matches_in_prompt` — If cell_matches is non-empty, matched values appear in the prompt.
8. `test_empty_sql_handling` — If Claude returns an empty response, the candidate is marked with an error flag (not silently empty).
9. `test_reasoning_trace_captured` — The `reasoning_trace` field in SQLCandidate is populated from the thinking block.
10. `test_concurrent_calls_possible` — All 4 API calls can be initiated concurrently (using asyncio) without errors.

---

### Sub-Operation 7B: Standard + Complex Generators

**File:** `src/generation/standard_generator.py`

**Models:**
- Generator B1 (Standard): `claude-haiku-4-5-20251001`
- Generator B2 (Complex): `claude-sonnet-4-6`

**Implementation steps:**

1. **Generator B1 — Standard SQL (Haiku):**

   System prompt (Markdown-focused):
   ```
   You are an expert SQL writer. Given a database schema and a question, write a correct SQL query.
   Focus on accuracy. Use the schema metadata to understand column meanings.
   ```

   Generate 2 candidates:
   - B1a: S₁ Markdown schema
   - B1b: S₂ Markdown schema

2. **Generator B2 — Complex SQL (Sonnet):**

   System prompt:
   ```
   You are an expert SQL writer specializing in advanced query patterns. For complex questions,
   prefer using CTEs, window functions, or subqueries to express logic clearly.
   Avoid unnested JOINs when CTEs improve readability and correctness.
   ```

   Generate 2 candidates:
   - B2a: S₁ Markdown schema
   - B2b: S₂ Markdown schema

3. **Both generators share** the same user prompt structure as the reasoning generator but without extended thinking.

#### Tests for Standard Generator

**File:** `tests/unit/test_standard_generator.py`

**Tests to write:**

1. `test_generates_4_candidates_total` — B1 + B2 together return exactly 4 SQLCandidate objects.
2. `test_b1_uses_haiku_model` — B1 API calls use `claude-haiku-4-5-20251001`.
3. `test_b2_uses_sonnet_model` — B2 API calls use `claude-sonnet-4-6`.
4. `test_generator_ids_labeled` — B1 candidates have `generator_id="standard_b1_*"`, B2 have `"complex_b2_*"`.
5. `test_markdown_schema_used` — Prompts contain Markdown table syntax (|column|type|), not DDL.
6. `test_schema_scope_varies` — One S₁ and one S₂ candidate per generator.
7. `test_no_extended_thinking` — API calls do NOT include `"thinking"` parameter.
8. `test_sql_clean_extracted` — SQL returned is free of markdown code fences.
9. `test_b2_prompt_mentions_advanced_patterns` — B2 system prompt contains language about CTEs/window functions.
10. `test_parallel_generation_both_generators` — B1 and B2 calls can run concurrently without race conditions.

---

### Sub-Operation 7C: ICL Generator

**File:** `src/generation/icl_generator.py`

**Model:** `claude-sonnet-4-6`

**Implementation steps:**

1. **Few-shot example formatting:** Format the 8 retrieved examples as:
   ```
   ## Example {n}
   Question: {question}
   Evidence: {evidence}
   SQL: {sql}
   ```
   Include all 8 examples in the system prompt (enables prompt caching).

2. **Generate 2–3 candidates:**
   - C1 (Direct): S₂ Markdown schema + few-shot examples. Prompt: "Write the SQL query for this question."
   - C2 (Chain-of-thought): S₂ Markdown schema + few-shot examples. Prompt: "First, identify which tables and joins are needed. Then write the SQL query."
   - C3 (Step-back, optional): S₂ Markdown schema + few-shot examples. Prompt: "What is the general SQL pattern for answering this type of question? Then apply it."

3. **Prompt caching:** Apply `cache_control` to the few-shot examples block (largest part of the prompt). Since examples are retrieved by structural similarity, the same examples appear for many questions, making caching highly effective.

4. **Cost guard:** If the few-shot examples block exceeds 6000 tokens, use only top-6 examples (trim from the bottom, keep most similar).

#### Tests for ICL Generator

**File:** `tests/unit/test_icl_generator.py`

**Tests to write:**

1. `test_generates_2_or_3_candidates` — Returns 2 or 3 SQLCandidate objects.
2. `test_few_shot_examples_in_prompt` — All retrieved few-shot examples appear in the system prompt.
3. `test_s2_schema_used_always` — ICL generator always uses S₂ (recall schema), never S₁.
4. `test_sonnet_model_used` — API calls use `claude-sonnet-4-6`.
5. `test_prompt_caching_applied_to_examples` — `cache_control` marker on the examples block.
6. `test_token_limit_guard` — With 8 long examples exceeding 6000 tokens, only top-6 are used.
7. `test_generator_id_label` — All candidates have `generator_id` starting with "icl_c".
8. `test_c2_prompt_includes_cot_instruction` — C2 candidate's prompt contains step-by-step analysis instruction.
9. `test_empty_examples_fallback` — If no few-shot examples are available, generates with 0-shot prompt (no crash).
10. `test_different_prompts_produce_different_candidates` — C1 and C2 prompts are verifiably different (checked via mock call capture).

---

### Integration Test: Generation Diversity

**File:** `tests/integration/test_generation_diversity.py`

**Tests to write:**

1. `test_all_generators_run_on_real_question` — Run all 3 generators on a single BIRD question using mocked Claude API. Verify 10–11 total candidates returned.
2. `test_generator_ids_are_unique` — All 10–11 candidates have unique `generator_id` labels.
3. `test_candidates_use_different_schemas` — At least 4 candidates use S₁ and at least 4 use S₂.
4. `test_generators_run_concurrently` — Using asyncio, all 3 generator types complete in parallel (wall time ≈ max of individual generators, not sum).
5. `test_candidate_pool_upper_bound_estimation` — On 20 BIRD mini-dev questions (live API), at least one correct SQL is in the candidate pool for >70% of questions (oracle upper bound test).

---

## Operation 8: Query Fixer

**File:** `src/fixing/query_fixer.py`

**Goal:** For each candidate SQL that fails execution, attempt up to β=2 fix iterations using a lightweight LLM.

### Implementation Steps

**Step 8.1 — Execute All Candidates**

Execute each candidate against the database:
- Collect: `success` (bool), `error_message` (str|None), `result_rows` (list), `is_empty` (bool)
- Flag candidates that need fixing: `not success` OR `is_empty`

**Step 8.2 — Error Categorization**

Categorize errors for targeted fix prompts:
- `syntax_error`: SQLite syntax error (e.g., "near 'FORM': syntax error")
- `schema_error`: Column/table not found (e.g., "no such column: frpm.Score")
- `type_error`: Type mismatch in comparison
- `empty_result`: Query is valid but returns 0 rows

Different error types get different fix instructions:
- Syntax: "Fix the SQL syntax error: {error}"
- Schema: "The column '{col}' doesn't exist. Check the schema and use the correct column name."
- Empty: "The query returned no rows. Review the WHERE conditions — they may be too restrictive."

**Step 8.3 — Fix Loop**

For each candidate needing a fix:
```
for iteration in range(β=2):
    if not (candidate.error or candidate.is_empty):
        break

    fixed_sql = call_fixer_llm(candidate, error_info, schema, question)
    candidate = execute_and_update(fixed_sql)
```

**Model:** `claude-haiku-4-5-20251001` (fast, sufficient for targeted error correction).

**Fix prompt:**
```
You are an expert SQL debugger. Fix the SQL query below.

Database Schema:
{schema_ddl}  # Use S₂ for maximum context

Question: {question}
Evidence: {evidence}
Matched values in database: {cell_matches}

Broken SQL:
{sql}

Error: {error_message}

Write only the corrected SQL query. No explanation.
```

**Step 8.4 — Output**

```python
@dataclass
class FixedCandidate:
    original_sql: str
    final_sql: str
    generator_id: str
    fix_iterations: int  # 0 if no fix needed
    execution_result: ExecutionResult
    confidence_score: float  # computed in Step 8.5
```

**Step 8.5 — Confidence Scoring (Improvement)**

Assign a confidence score to each candidate:
- `+1.0` if execution succeeds and result is non-empty
- `+0.5` bonus if result size is "plausible" (1–100 rows for non-aggregation, 1 row for aggregation)
- `-0.5` if required 1 or 2 fix iterations to execute
- `0.0` if still failing after β=2 fixes (discard from pool)

Normalize scores to [0, 1] across the candidate pool.

#### Tests for Query Fixer

**File:** `tests/unit/test_query_fixer.py`

**Test setup:** Use a real SQLite in-memory database; mock the Claude API for fix calls.

**Tests to write:**

1. `test_valid_sql_passes_through_unchanged` — A syntactically correct SQL that returns rows gets fix_iterations=0 and is unchanged.
2. `test_syntax_error_triggers_fix` — SQL with syntax error ("SELECT form students") triggers a fix call.
3. `test_empty_result_triggers_fix` — Valid SQL returning 0 rows triggers a fix call.
4. `test_max_2_iterations` — Even if both fix iterations fail, the loop stops at 2 (no more API calls).
5. `test_fix_uses_haiku_model` — Fix API calls use `claude-haiku-4-5-20251001`.
6. `test_still_failing_candidate_discarded` — Candidates that fail all β=2 fix attempts have confidence_score=0.0.
7. `test_error_categorization_syntax` — "near 'FORM'" error is categorized as `syntax_error`.
8. `test_error_categorization_schema` — "no such column" error is categorized as `schema_error`.
9. `test_error_message_in_fix_prompt` — The actual error message is included verbatim in the fix prompt.
10. `test_confidence_score_higher_for_clean_success` — A candidate that executes immediately has higher confidence than one that needed 1 fix.
11. `test_confidence_score_plausibility_bonus` — Aggregation query returning exactly 1 row gets plausibility bonus.
12. `test_parallel_fixing_independent_candidates` — Fix calls for multiple candidates can run concurrently.

---

## Operation 9: Adaptive SQL Selection

**File:** `src/selection/adaptive_selector.py`

**Goal:** Select the best SQL from the fixed candidate pool using fast-path consensus or pairwise tournament.

### Implementation Steps

**Step 9.1 — Execute and Cluster**

Execute all surviving candidates (those with confidence_score > 0) on the database.
Cluster by execution result equivalence:
- Two candidates are in the same cluster if their result sets are identical (same rows, same order after sorting).
- For clustering: sort each result set's rows, stringify, hash to a cluster key.

**Step 9.2 — Fast Path (unanimous, ~45% of questions)**

If all candidates fall into 1 cluster (unanimous agreement):
- Select the shortest SQL (fewest characters) from the cluster.
- Return immediately. No API call needed.

**Step 9.3 — Representative Selection**

If 2+ clusters:
1. Select 1 representative per cluster: the candidate with the highest confidence score (ties broken by shortest SQL).
2. Reorganize representatives:
   - **Inter-group:** Sort clusters by size descending (largest cluster first).
   - **Intra-group:** Already have 1 per cluster; apply generator performance ranking as secondary sort.

**Generator performance ranking** (pre-defined based on expected accuracy):
1. Reasoning generator (A1–A4) — highest
2. Complex generator (B2a, B2b)
3. ICL generator (C1–C3)
4. Standard generator (B1a, B1b) — lowest per-candidate but fast

**Step 9.4 — Pairwise Tournament**

For m representatives, run C(m,2) pairwise comparisons using `claude-haiku-4-5-20251001`:

**Selection prompt:**
```
You are evaluating two SQL queries for a natural language question. Choose which query
better answers the question based on the schema and execution results.

Question: {question}
Evidence: {evidence}
Database Schema (filtered): {s1_markdown}

Candidate A (generated by {generator_a}):
{sql_a}
Execution result: {result_a}

Candidate B (generated by {generator_b}):
{sql_b}
Execution result: {result_b}

Which candidate better answers the question? Reply with just "A" or "B".
```

Use a tool-use definition to force structured output:
```python
select_winner_tool = {
    "name": "select_winner",
    "input_schema": {
        "type": "object",
        "properties": {
            "winner": {"type": "string", "enum": ["A", "B"]},
            "reason": {"type": "string"}
        }
    }
}
```

**Step 9.5 — Final Selection**

- Winner = argmax(wins).
- Tiebreaker 1: Cluster size (more candidates agreeing).
- Tiebreaker 2: Confidence score.
- Tiebreaker 3: Generator performance ranking.

**Step 9.6 — Fallback (Improvement over original plan)**

If fewer than 2 executable candidates survive after fixing:
- Apply majority voting across all candidates (including non-executable) using the cluster with most members.
- Log a warning that selection quality may be reduced.

**Output:**
```python
@dataclass
class SelectionResult:
    final_sql: str
    selection_method: str    # "fast_path" | "tournament" | "fallback"
    tournament_wins: dict[str, int]  # generator_id → win count
    confidence: float
    cluster_count: int
    candidates_evaluated: int
```

#### Tests for Adaptive Selector

**File:** `tests/unit/test_adaptive_selector.py`

**Test setup:** Create mock candidates with pre-defined execution results. Mock Claude API for pairwise calls.

**Tests to write:**

1. `test_unanimous_takes_fast_path` — 5 candidates all returning the same result → fast path selected, `selection_method="fast_path"`, 0 API calls.
2. `test_fast_path_selects_shortest_sql` — Fast path chooses the candidate with fewest characters among the unanimous cluster.
3. `test_two_clusters_triggers_tournament` — 2 distinct result clusters → tournament triggered, `selection_method="tournament"`.
4. `test_tournament_winner_has_most_wins` — In a 3-way tournament, winner has won 2 of 2 matches.
5. `test_candidate_reorganization_order` — Verify representatives are ordered by cluster size descending before tournament.
6. `test_generator_ranking_as_tiebreaker` — When cluster sizes are equal, reasoning generator representative comes before standard generator.
7. `test_haiku_used_for_pairwise` — Pairwise comparison calls use `claude-haiku-4-5-20251001`.
8. `test_pairwise_comparison_count` — With 4 clusters (4 representatives), C(4,2)=6 API calls are made.
9. `test_fallback_on_no_executable_candidates` — When all candidates fail execution, fallback to majority voting.
10. `test_structured_winner_output` — Selection prompt uses tool-use to force "A" or "B" response (not free text).
11. `test_result_equivalence_clustering` — Two candidates with same rows in different order are in the same cluster.
12. `test_empty_result_cluster_deprioritized` — Candidates returning empty results are placed in lower-priority clusters.

---

## Pipeline Integration

**Files:** `src/pipeline/offline_pipeline.py`, `src/pipeline/online_pipeline.py`

### Offline Pipeline

**File:** `src/pipeline/offline_pipeline.py`

Orchestrates Operations 0 and 1 for a given database:
```python
async def run_offline_pipeline(db_id: str, db_path: str, train_data: list[BirdEntry]) -> OfflineArtifacts:
    profile = await run_profiling(db_id, db_path)
    summary = await run_summarization(profile)
    schemas = format_schemas(profile, summary)
    lsh_index = build_lsh_index(db_id, db_path)
    faiss_index = build_faiss_index(summary)
    example_store = build_example_store(train_data)
    return OfflineArtifacts(profile, summary, schemas, lsh_index, faiss_index, example_store)
```

### Online Pipeline

**File:** `src/pipeline/online_pipeline.py`

Orchestrates Operations 5→9 for a single question:
```python
async def answer_question(
    entry: BirdEntry,
    artifacts: OfflineArtifacts
) -> PipelineResult:
    # Op 5: Context Grounding
    grounding = await ground_context(entry.question, entry.evidence, artifacts)

    # Op 6: Schema Linking
    schemas = await link_schema(entry.question, entry.evidence, grounding, artifacts)

    # Op 7: SQL Generation (all generators in parallel)
    candidates = await asyncio.gather(
        generate_reasoning(entry, schemas, grounding, artifacts),
        generate_standard(entry, schemas, grounding),
        generate_icl(entry, schemas, grounding),
    )
    candidates = flatten(candidates)

    # Op 8: Query Fixer
    fixed_candidates = await fix_candidates(candidates, entry, schemas, artifacts)

    # Op 9: Selection
    result = await select_best(fixed_candidates, entry, schemas)

    return PipelineResult(final_sql=result.final_sql, metadata=result)
```

### Integration Tests for Online Pipeline

**File:** `tests/integration/test_online_pipeline.py`

**Tests to write:**

1. `test_full_pipeline_single_question` — Run the complete online pipeline on one BIRD question (mocked Claude API). Verify a non-empty SQL string is returned.
2. `test_pipeline_produces_valid_sql` — The returned SQL is parseable (no obvious syntax errors).
3. `test_pipeline_result_has_metadata` — PipelineResult includes cluster_count, selection_method, candidates_evaluated.
4. `test_pipeline_handles_all_candidates_failing` — If all 10 candidates fail execution and fixing, pipeline returns the best-effort SQL (not a crash).
5. `test_pipeline_caches_offline_artifacts` — Second call with same db_id skips API calls for offline operations.
6. `test_async_generators_run_in_parallel` — Using asyncio event loop tracing, verify generator calls overlap in time.

---

## BIRD Evaluation

**Files:** `src/evaluation/evaluator.py`, `scripts/run_evaluation.py`

### Evaluation Implementation

**File:** `src/evaluation/evaluator.py`

**Execution Accuracy (EX):**
- Execute both the predicted SQL and the ground-truth SQL on the database.
- Compare result sets (sorted, type-normalized).
- `EX = 1` if result sets are identical, else `0`.
- Report mean EX across all questions.

**Stratified reporting:**
- Overall EX
- EX by difficulty: "simple", "moderate", "challenging"
- EX by database
- EX by selection method (fast_path vs. tournament)
- EX by generator (which generator produced the winning candidate)

**Error analysis:**
- Count of questions with no executable candidate (full failure)
- Count fixed by query fixer (counted as rescued)
- Tournament win distribution per generator

**File:** `scripts/run_evaluation.py`

```bash
# Run on mini-dev (500 questions, fast)
python scripts/run_evaluation.py --split mini_dev --output results/mini_dev_results.json

# Run on full dev (1534 questions)
python scripts/run_evaluation.py --split dev --output results/dev_results.json

# Resume interrupted run
python scripts/run_evaluation.py --split dev --resume results/dev_results_partial.json
```

Features:
- Progress bar with live EX estimate
- Save intermediate results to disk after each question (resumable)
- Cost tracking (token usage per operation)
- Configurable parallelism (`--workers N` for N concurrent questions)

---

## End-to-End Tests

### E2E Smoke Test (50 Questions)

**File:** `tests/e2e/test_bird_mini.py`

**Purpose:** Fast validation that the full pipeline runs without crashes on representative questions. Uses live API calls (marks with `@pytest.mark.live`).

**Tests to write:**

1. `test_pipeline_runs_50_questions` — Run the full pipeline on the first 50 questions of BIRD mini-dev. Assert:
   - No unhandled exceptions
   - All 50 questions produce a non-empty SQL string
   - EX accuracy >= 50% (basic sanity bar)
   - No question takes > 60 seconds end-to-end

2. `test_fast_path_rate_reasonable` — Among 50 questions, at least 30% trigger the fast path (unanimous agreement).

3. `test_query_fixer_rescue_rate` — Among all failing candidates across 50 questions, at least 30% are successfully fixed.

4. `test_cost_estimate_reasonable` — Total API cost for 50 questions < $10 (based on token counting).

5. `test_results_saved_to_disk` — After running, a JSON results file exists with 50 entries.

**How to run:**
```bash
pytest tests/e2e/test_bird_mini.py -v -m live --timeout=300
```

### E2E Full BIRD Dev Evaluation

**File:** `tests/e2e/test_bird_full.py`

**Purpose:** Full evaluation on the BIRD dev set (1,534 questions). This is the primary benchmark.

**Tests to write:**

1. `test_bird_dev_ex_accuracy` — Run full evaluation and assert EX >= 68% (conservative Phase 1 target; SOTA is ~73–75%).
2. `test_simple_questions_accuracy` — EX on "simple" difficulty questions >= 78%.
3. `test_moderate_questions_accuracy` — EX on "moderate" difficulty >= 65%.
4. `test_challenging_questions_accuracy` — EX on "challenging" difficulty >= 45%.
5. `test_no_crashes_across_all_questions` — Pipeline completes all 1,534 questions without unhandled exceptions.
6. `test_evaluation_resumable` — Interrupt after 500 questions, resume; final results identical to uninterrupted run.
7. `test_fast_path_rate` — Fast path invoked for 40–55% of questions (validates the ~45% estimate from XiYan).
8. `test_generator_contribution` — Each generator type contributes at least 5% of winning candidates (no generator is never selected).
9. `test_cost_within_budget` — Total API cost < $200 for full dev set (at ~$0.13/question average).
10. `test_error_analysis_report_generated` — Results JSON contains per-difficulty, per-database, and per-generator breakdowns.

**How to run:**
```bash
pytest tests/e2e/test_bird_full.py -v -m live --timeout=7200  # 2 hour timeout
```

---

## Implementation Order and Milestones

### Week 1 — Foundation (Ops 0, 1)

| Day | Task |
|-----|------|
| 1–2 | Project setup, BIRD dataset download, `settings.py`, `bird_loader.py`, `database.py` |
| 3 | `profiler.py` + `test_profiler.py` (all 10 tests pass) |
| 4 | `summarizer.py` + `test_summarizer.py` (all 10 tests pass) |
| 5 | `schema_formatter.py` + `test_schema_formatter.py` (all 10 tests pass) |
| 6 | `lsh_index.py` + `test_lsh_index.py` + `faiss_index.py` + `test_faiss_index.py` |
| 7 | `example_store.py` + `test_example_store.py` + integration test `test_offline_pipeline.py` |

**Milestone 1:** Run `scripts/run_offline_preprocessing.py --split dev` on all BIRD dev databases. Verify preprocessed artifacts exist for all 11 databases.

### Week 2 — Online Pipeline (Ops 5, 6, 7)

| Day | Task |
|-----|------|
| 8 | `context_grounder.py` + `test_context_grounder.py` |
| 9 | `schema_linker.py` + `test_schema_linker.py` (first 8 tests) |
| 10 | `schema_linker.py` (tests 9–12) + prompt caching validation |
| 11 | `reasoning_generator.py` + `test_reasoning_generator.py` |
| 12 | `standard_generator.py` + `icl_generator.py` + tests |
| 13 | `online_pipeline.py` (partial: Ops 5+6+7) + integration tests |
| 14 | `test_generation_diversity.py` integration tests |

**Milestone 2:** Run the pipeline through Op 7 on 10 BIRD questions. Verify 10–11 candidates are generated per question.

### Week 3 — Completion + E2E (Ops 8, 9)

| Day | Task |
|-----|------|
| 15 | `query_fixer.py` + `test_query_fixer.py` |
| 16 | `adaptive_selector.py` + `test_adaptive_selector.py` (first 8 tests) |
| 17 | `adaptive_selector.py` (tests 9–12) + edge cases |
| 18 | `online_pipeline.py` (complete: all ops) + `test_online_pipeline.py` |
| 19 | `evaluator.py` + `scripts/run_evaluation.py` |
| 20 | `tests/e2e/test_bird_mini.py` — smoke test on 50 questions |
| 21 | Full BIRD dev evaluation + `tests/e2e/test_bird_full.py` |

**Milestone 3:** Full BIRD dev evaluation complete with results report. EX ≥ 68% achieved.

---

## Testing Quick Reference

```bash
# Run all unit tests (no API calls, fast)
pytest tests/unit/ -v

# Run integration tests (mocked API, fast)
pytest tests/integration/ -v

# Run e2e smoke test (live API, ~$2–5, ~10 minutes)
pytest tests/e2e/test_bird_mini.py -v -m live

# Run full BIRD evaluation (live API, ~$150–200, ~4–8 hours)
pytest tests/e2e/test_bird_full.py -v -m live

# Run a specific test file
pytest tests/unit/test_schema_linker.py -v

# Run with coverage report
pytest tests/unit/ --cov=src --cov-report=html

# Run only tests that don't need live API
pytest tests/ -m "not live" -v
```

---

## Cache Manager

**File:** `src/cache/cache_manager.py`

**Purpose:** Avoid duplicate API calls during development. Cache LLM responses keyed by a hash of the input prompt + model.

**Implementation:**
- Cache directory: `data/cache/`
- Cache key: `SHA256(model + prompt_text)`
- Cache format: JSON files `{cache_key[:16]}.json` with fields: `key`, `model`, `prompt_hash`, `response`, `timestamp`, `token_count`
- Cache invalidation: Manual (delete cache directory) or TTL (default: never expire for offline ops, 24h for online ops)
- Cache toggle: `CACHE_LLM_RESPONSES=true` in `.env` (default false for production)

**Usage in all generators:**
```python
@cache_manager.cached(model="claude-haiku-4-5-20251001")
async def extract_keywords(question: str, evidence: str) -> GroundingContext:
    ...
```

#### Tests for Cache Manager

**File:** `tests/unit/test_cache_manager.py`

**Tests to write:**

1. `test_cache_hit_returns_cached_response` — Second call with same inputs returns cached value without API call.
2. `test_cache_miss_calls_api` — First call (no cache) makes an API call and writes to cache.
3. `test_cache_key_is_deterministic` — Same inputs always produce the same cache key.
4. `test_different_models_different_cache_keys` — Same prompt but different model → different cache entry.
5. `test_cache_disabled_by_config` — With `CACHE_LLM_RESPONSES=false`, no caching occurs.
6. `test_cache_file_created_on_write` — After a cache miss, the JSON cache file exists on disk.
7. `test_cache_roundtrip_preserves_response` — Cached response is byte-for-byte identical to original API response.

---

## Cost Estimation

| Operation | Model | Est. Tokens/Question | Est. Cost/Question |
|-----------|-------|---------------------|-------------------|
| Keyword extraction | Haiku | ~500 in / ~50 out | ~$0.001 |
| Schema linking (2 calls) | Sonnet | ~4,000 in / ~200 out | ~$0.013 |
| Reasoning gen (4 calls) | Sonnet + thinking | ~6,000 in / ~500 out | ~$0.030 |
| Standard gen B1 (2 calls) | Haiku | ~3,000 in / ~100 out | ~$0.004 |
| Complex gen B2 (2 calls) | Sonnet | ~3,500 in / ~100 out | ~$0.012 |
| ICL gen (2-3 calls) | Sonnet | ~5,000 in / ~200 out | ~$0.020 |
| Query fixer (avg 2 calls) | Haiku | ~2,000 in / ~100 out | ~$0.003 |
| Selection tournament | Haiku | ~2,000 in / ~20 out | ~$0.003 |
| **Total** | | | **~$0.086** |

With prompt caching on schema/examples (est. 70% cache hit on schema tokens):
- **Effective cost: ~$0.06–0.09 per question**
- **Full BIRD dev (1,534 questions): ~$90–140**

---

## Key Technical Notes

### Handling BIRD Evaluation Edge Cases

1. **Multiple valid SQL formulations:** BIRD uses execution-result comparison, so syntactically different SQL queries are both correct if they produce the same result. This benefits our approach.

2. **Evidence utilization:** BIRD often includes evidence that defines formulas or clarifies column meanings. Always include evidence in both grounding and generation prompts.

3. **Database timeout:** Set a 30-second timeout on all SQL execution to handle queries that accidentally produce Cartesian products.

4. **Type normalization in EX:** When comparing result sets, normalize types: float 1.0 == int 1, None == NULL. Use string comparison after sorting.

5. **Column order in results:** BIRD's official evaluator sorts result rows but does NOT require specific column ordering. Match this behavior.

### Extended Thinking Configuration for claude-sonnet-4-6

The extended thinking feature uses `budget_tokens` to control thinking depth:
- Simple questions: `budget_tokens=4000`
- Complex (multiple JOINs, subqueries): `budget_tokens=8000`

**Adaptive budget:** Estimate complexity from number of tables in S₂ schema:
- 1–2 tables: 4,000 tokens
- 3–4 tables: 6,000 tokens
- 5+ tables: 8,000 tokens

### Async Architecture

All online pipeline operations use `asyncio`:
- Generators run concurrently using `asyncio.gather()`
- Fix loops run concurrently per candidate using `asyncio.gather()`
- Pairwise selection comparisons run concurrently using `asyncio.gather()`
- Global semaphore: `asyncio.Semaphore(10)` limits concurrent API calls to avoid rate limits

### Anthropic Rate Limits

Handle rate limits with `tenacity`:
```python
@retry(
    retry=retry_if_exception_type(anthropic.RateLimitError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
async def call_claude(messages, ...):
    ...
```

---

## Expected Results

After completing Phase 1:

| Metric | Target |
|--------|--------|
| BIRD Dev EX Accuracy | ≥ 68% (stretch: ≥ 72%) |
| Simple difficulty EX | ≥ 78% |
| Moderate difficulty EX | ≥ 65% |
| Challenging difficulty EX | ≥ 45% |
| Fast-path rate | 40–55% of questions |
| Average API cost per question | < $0.13 |
| Total BIRD dev evaluation cost | < $200 |
| Average latency per question | < 30 seconds |
| Unit test coverage | ≥ 90% on src/ |

These targets position Phase 1 within 2–5% of methods requiring 32-GPU training clusters, at a fraction of the infrastructure cost.
