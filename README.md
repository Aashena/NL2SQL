# Adaptive Hybrid NL2SQL

An API-only, research-grade Text-to-SQL system built on the BIRD benchmark.
It combines the highest-impact components from four state-of-the-art methods —
**Agentar-Scale-SQL**, **Automatic Metadata Extraction**, **CHASE-SQL**, and **XiYan-SQL** —
into a single cohesive pipeline that targets ≥ 68 % Execution Accuracy (EX) on the BIRD dev set
using only LLM API calls (no local fine-tuning required in Phase 1).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Pipeline Stages](#pipeline-stages)
   - [Offline Phase](#offline-phase)
   - [Online Phase](#online-phase)
3. [Project Structure](#project-structure)
4. [Component Reference](#component-reference)
   - [LLM Abstraction Layer](#llm-abstraction-layer)
   - [Data Layer](#data-layer)
   - [Preprocessing](#preprocessing)
   - [Indexing](#indexing)
   - [Grounding](#grounding)
   - [Schema Linking](#schema-linking)
   - [SQL Generation](#sql-generation)
   - [Query Fixer](#query-fixer)
   - [Adaptive Selector](#adaptive-selector)
   - [Cache Manager](#cache-manager)
   - [Evaluation](#evaluation)
5. [Configuration](#configuration)
6. [Setup & Installation](#setup--installation)
7. [Dataset Setup](#dataset-setup)
8. [Running the System](#running-the-system)
9. [Testing](#testing)
10. [Performance & Design Decisions](#performance--design-decisions)

---

## Architecture Overview

The system is split into two phases:

```
┌──────────────────────────────────────────────────────────────────────┐
│  OFFLINE PHASE  (one-time per database, results cached to disk)       │
│                                                                       │
│  SQLite DB ──► Op 0a: Statistical Profiling                           │
│                ──► Op 0b: LLM Field Summarization                     │
│                ──► Op 0c: Schema Formatting (DDL + Markdown)          │
│                ──► Op 1a: LSH Index  (cell values)                    │
│                ──► Op 1b: FAISS Index (field semantics)               │
│  Training Set  ──► Op 1c: Example Store (few-shot retrieval)          │
└──────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ONLINE PHASE  (per question at inference time)                       │
│                                                                       │
│  (Question, Evidence) ──► Op 5: Context Grounding                    │
│                       ──► Op 6: Adaptive Schema Linking  (S₁, S₂)   │
│                       ──► Op 7: Diverse SQL Generation   (10-11 cands)│
│                       ──► Op 8: Query Fixer + Semantic Verifier      │
│                       ──► Op 9: Adaptive Selection ──► Final SQL     │
└──────────────────────────────────────────────────────────────────────┘
```

**Expected inference cost per question:** 14–68 LLM calls
(vs. Agentar: 20–630 | CHASE-SQL: ~480 | XiYan: ~12–15)

---

## Pipeline Stages

### Offline Phase

The offline phase runs once per database and caches all results to disk. Subsequent online runs load from cache without re-computing anything.

#### Op 0a — Statistical Profiling (`src/preprocessing/profiler.py`)

Performs a single-pass scan of every column in every table:

- Counts total rows and NULL values (null rate)
- Computes distinct value counts (HyperLogLog approximation)
- Extracts top-k most frequent cell values
- Computes min / max / avg for numeric columns
- Computes average string length for text columns
- Builds MinHash sketches (3-grams) for fuzzy-matching support

Output is a `DatabaseProfile` saved as JSON under `data/preprocessed/profiles/{db_id}.json`.

#### Op 0b — LLM Field Summarization (`src/preprocessing/summarizer.py`)

Calls a **fast LLM** (Haiku / Gemini Flash) once per batch of columns to generate two kinds of descriptions:

| Description | Length | Purpose |
|-------------|--------|---------|
| **Short summary** | ≤ 200 chars | Used in schema-linking prompts (space-efficient) |
| **Long summary** | ≤ 1 000 chars | Injected into generation schemas (rich semantic context) |

Features:
- Batches columns by table (up to 6 per LLM call) to minimize API costs
- Falls back to individual column calls if a batch fails
- Fills with deterministic defaults if all retries fail (never crashes the pipeline)
- Prompt caching via `CacheableText(cache=True)` on the instruction block

Output: `DatabaseSummary` saved as JSON under `data/preprocessed/summaries/{db_id}.json`.

#### Op 0c — Schema Formatting (`src/preprocessing/schema_formatter.py`)

Generates two schema representations from the profile + summaries:

| Format | Style | Consumed by |
|--------|-------|-------------|
| **DDL** (`{db_id}_ddl.sql`) | `CREATE TABLE` statements with inline summary comments and sample values | Generator A (Reasoning) |
| **Markdown** (`{db_id}_markdown.md`) | Tabular format with long summaries, types, and FK annotations | Generators B1, B2, C (ICL) |

Both formats include primary/foreign key annotations and are stored under `data/preprocessed/schemas/`.

#### Op 1a — LSH Index (`src/indexing/lsh_index.py`)

Builds a Locality-Sensitive Hashing index over all distinct cell values in the database:

- Uses 3-character shingling + MinHash signatures
- Stores up to 50 000 distinct values per column (cap prevents index bloat on large string columns)
- Supports fuzzy lookup at Jaccard threshold ≈ 0.3 (retrieves single-transposition typos)
- Post-retrieval re-ranks results by true Jaccard similarity, exact matches always first
- Serialized via `pickle` to `data/preprocessed/indices/{db_id}_lsh.pkl`

#### Op 1b — FAISS Semantic Index (`src/indexing/faiss_index.py`)

Builds a vector index over **long field summaries** using `sentence-transformers/all-MiniLM-L6-v2`:

- Uses `IndexFlatIP` with L2-normalized embeddings (cosine similarity) for databases ≤ 1 000 fields
- Uses `IndexIVFFlat` (nlist = min(32, N//10)) for larger databases
- Enables semantic field retrieval: find schema columns related to a question even without keyword overlap
- Saved as `{db_id}_faiss.index` + `{db_id}_faiss_fields.json`

#### Op 1c — Example Store (`src/indexing/example_store.py`)

Builds a shared vector store over all BIRD **training** question–SQL pairs:

- Applies skeleton masking: entity names, numbers, and strings are replaced with `[ENTITY]`, `[NUM]`, `[STR]` placeholders using spaCy NER + regex
- Embeds the masked skeleton with `all-MiniLM-L6-v2`
- Stores metadata: original question, SQL, db_id, difficulty
- At query time, excludes examples from the same `db_id` to prevent schema leakage
- Saved as a shared `indices/example_store.faiss` + `indices/example_store_metadata.json`

---

### Online Phase

The online phase processes one question at a time and orchestrates Ops 5–9 through `src/pipeline/online_pipeline.py`.

Each `answer_question()` call has:
- **Per-generator timeout:** 120 s — timed-out generators are silently dropped
- **Per-question soft timeout:** 300 s — returns empty fallback rather than hanging

#### Op 5 — Context Grounding (`src/grounding/context_grounder.py`)

Two parallel retrieval operations:

1. **Keyword extraction** — a fast LLM extracts database literals from the question and evidence
   (e.g., `"students in Alameda County"` → keywords `["Alameda County", "Alameda"]`)
2. **LSH lookup** — each keyword queries the LSH index; returns `CellMatch` objects with `(table, column, value, similarity_score)`
3. **Schema hints** — keywords with high-confidence matches are recorded as column hints for the schema linker
4. **Skeleton retrieval** — the question is embedded (with skeleton masking) and the top-8 structurally similar training examples are retrieved from the Example Store

Multi-word keywords are also queried word-by-word (words ≥ 3 chars) to improve recall.

Output: `GroundingContext` with `matched_cells`, `schema_hints`, `few_shot_examples`.

#### Op 6 — Adaptive Schema Linking (`src/schema_linking/schema_linker.py`)

Reduces the full database schema to two focused subsets:

```
Full schema (all fields)
  │
  ▼
FAISS top-50 field candidates (semantic similarity to question)
  + Schema hints from grounding context
  │
  ├──► LLM Iteration 1 → S₁ (precise: only the most needed fields)
  │         + auto-include PKs/FKs for selected tables
  │
  └──► LLM Iteration 2 → S₂ = S₁ ∪ additional candidates (higher recall)
```

Implementation details:
- Hallucination filter: LLM-returned `(table, column)` pairs are validated against the actual schema before inclusion
- `S₁ ⊆ S₂` invariant is guaranteed at output time
- Both S₁ and S₂ are returned in DDL and Markdown formats
- The system prompt block (candidate field list with summaries) is prompt-cached via `CacheableText(cache=True)`, saving ~60 % of schema-linking costs for repeated databases
- Exactly 2 LLM calls are made, even if no candidates remain after the first iteration

Output: `LinkedSchemas` with `s1_ddl`, `s1_markdown`, `s2_ddl`, `s2_markdown`, `s1_fields`, `s2_fields`.

#### Op 7 — Diverse SQL Generation

Three generators run **concurrently** via `asyncio.gather()`:

##### Generator A — Reasoning Generator (`src/generation/reasoning_generator.py`)

Uses `model_reasoning` with **extended thinking** (Anthropic) or **thinking mode** (Gemini):

| Candidate | Schema | Prompt variant |
|-----------|--------|----------------|
| A1 | S₁ DDL | Minimal — "Write a SQL query" |
| A2 | S₁ DDL | Step-by-step CoT |
| A3 | S₂ DDL | Minimal |
| A4 | S₂ DDL | Step-by-step CoT |

- Thinking budget scales with schema complexity: 4 000 (≤ 2 tables) → 6 000 (≤ 4) → 8 000 (5+) tokens
- Reasoning trace is stored in `SQLCandidate.reasoning_trace`
- SQL is extracted from free-text response (no tool-use), cleaned of markdown fences

##### Generator B — Standard & Complex (`src/generation/standard_generator.py`)

Two sub-generators, all 4 candidates launched concurrently:

| Candidate | Generator | Model | Schema | Focus |
|-----------|-----------|-------|--------|-------|
| B1a | B1 Standard | `model_fast` | S₁ Markdown | Broad coverage |
| B1b | B1 Standard | `model_fast` | S₂ Markdown | Broad coverage |
| B2a | B2 Complex | `model_powerful` | S₁ Markdown | CTEs, window functions |
| B2b | B2 Complex | `model_powerful` | S₂ Markdown | CTEs, window functions |

- Uses tool-use JSON schema for structured SQL output
- B2 prompt explicitly references advanced SQL patterns (CTEs, subqueries, window functions)

##### Generator C — ICL Generator (`src/generation/icl_generator.py`)

Uses `model_powerful` with few-shot examples from the Example Store:

| Candidate | Prompt style |
|-----------|-------------|
| C1 | Direct: "Write the SQL query" |
| C2 | Chain-of-thought: "First identify which tables and joins..." |
| C3 | Step-back: "What is the general SQL pattern for this type of question?" |

- Uses S₂ Markdown schema (maximum recall)
- Includes up to 8 few-shot examples; trims to 6 if estimated token count exceeds 6 000
- The few-shot examples block is prompt-cached across C1/C2/C3 calls

**Total: 10–11 SQL candidates** (C3 is optional based on question complexity)

**Diversity axes:**
1. Model tier (fast / powerful / reasoning with extended thinking)
2. Schema format (DDL vs. Markdown)
3. Schema scope (S₁ precise vs. S₂ recall)
4. Prompting strategy (direct / CoT / step-back / few-shot ICL)

#### Op 8 — Query Fixer + Semantic Verifier (`src/fixing/query_fixer.py`, `src/verification/query_verifier.py`)

Two-stage verification integrated into the fix loop:

1. **Verification plan generation** (once per question, 1 LLM call): generates a list of semantic
   tests applicable to the question — grain, null, duplicate, ordering, scale, completeness (cheap),
   plus column_alignment, boundary, symmetry (expensive, LLM-judged)
2. **Per-candidate fix loop** (β + 1 = 3 iterations total: 2 fix attempts + 1 final assessment):
   - **Stage A:** Execute SQL → check executability (syntax/schema/empty errors)
   - **Stage B:** If execution succeeded, evaluate verification tests (cheap every iteration;
     expensive only on final pass)
   - If both stages pass → break early; else fix with **combined feedback** from both stages
3. Fix LLM (`model_fast`) receives execution errors AND verification failure hints in one prompt

**Confidence scoring** (applied after fixing):

| Signal | Score |
|--------|-------|
| Successful execution | +1.0 |
| Plausible result size (aggregation → 1 row, non-agg → 1–100 rows) | +0.5 |
| Per fix iteration required | −0.5 |
| Critical verification failure (grain, duplicate, column_alignment) | −0.3 each |
| Minor verification failure (null, ordering, scale, boundary, symmetry) | −0.1 each |
| All applicable verification tests pass | +0.2 bonus |
| Still failing after all iterations | 0.0 |

Scores are normalized across the candidate pool (all-zero → all 1.0).

Output: list of `FixedCandidate` objects with `sql`, `success`, `confidence`, `fix_count`,
`verification_results` (`VerificationEvaluation`).

#### Op 9 — Adaptive Selection (`src/selection/adaptive_selector.py`)

Multi-phase selection algorithm:

**Phase 1 — Cluster by execution result**

All candidates are executed; results are sorted and normalized (`None` → `"NULL_SENTINEL"`, `float` → 6 d.p. string) and used as cluster keys. Column order within each row is ignored.

**Phase 2 — Decision branch**

```
m = number of distinct clusters

if m == 1 (unanimous, ~45% of questions):
    ── FAST PATH ──
    return shortest SQL from the single cluster
    (0 LLM calls)

else:
    ── TOURNAMENT PATH ──
    1. Pick 1 representative per cluster
       (highest confidence; tiebreak: shortest SQL)
    2. Sort representatives:
       - Empty-result clusters last
       - Then by cluster size descending (majority signal)
       - Then by generator rank (reasoning=0 > complex=1 > icl=2 > standard=3)
    3. Run pairwise round-robin tournament: C(m,2) comparisons
       - model_fast evaluates each pair via tool-use
       - Higher-ranked representative presented as option "A"
    4. Winner = argmax(tournament wins)
       - Tiebreakers: cluster size → confidence → generator rank

if < 2 executable candidates:
    ── FALLBACK PATH ──
    return highest-confidence candidate (0 LLM calls)
```

**Worst-case tournament comparisons:** C(8,2) = 28 (vs. Agentar's 561)

Output: `SelectionResult` with `final_sql`, `selection_method`, `cluster_count`, `candidates_evaluated`, `tournament_wins`.

---

## Project Structure

```
NL2SQL/
├── src/
│   ├── config/
│   │   └── settings.py            # Pydantic settings (reads .env)
│   ├── data/
│   │   ├── bird_loader.py         # BIRD dataset loading + schema parsing
│   │   └── database.py            # SQLite execution utilities
│   ├── preprocessing/
│   │   ├── profiler.py            # Op 0a: Statistical profiling
│   │   ├── summarizer.py          # Op 0b: LLM field summarization
│   │   └── schema_formatter.py    # Op 0c: DDL + Markdown schema generation
│   ├── indexing/
│   │   ├── lsh_index.py           # Op 1a: LSH cell-value index
│   │   ├── faiss_index.py         # Op 1b: FAISS semantic field index
│   │   └── example_store.py       # Op 1c: Few-shot example vector store
│   ├── llm/
│   │   ├── base.py                # Abstract LLMClient, LLMResponse, CacheableText, etc.
│   │   ├── anthropic_client.py    # Anthropic provider (Claude Haiku / Sonnet)
│   │   ├── gemini_client.py       # Google Gemini provider
│   │   └── mlx_client.py          # MLX local inference (Apple Silicon)
│   ├── grounding/
│   │   └── context_grounder.py    # Op 5: Keyword extraction + LSH/example retrieval
│   ├── schema_linking/
│   │   └── schema_linker.py       # Op 6: Dual-schema FAISS + LLM filtering
│   ├── generation/
│   │   ├── base_generator.py      # Shared SQL utilities (clean_sql, SQL writing rules)
│   │   ├── reasoning_generator.py # Op 7A: Extended thinking generator (4 candidates)
│   │   ├── standard_generator.py  # Op 7B: Standard + complex generators (4 candidates)
│   │   └── icl_generator.py       # Op 7C: ICL few-shot generator (2–3 candidates)
│   ├── fixing/
│   │   └── query_fixer.py         # Op 8: Execute → fix errors/verify semantics
│   ├── verification/
│   │   ├── __init__.py
│   │   └── query_verifier.py      # Op 8 (integrated): semantic verification plan
│   ├── selection/
│   │   └── adaptive_selector.py   # Op 9: Cluster → fast-path / tournament selection
│   ├── cache/
│   │   └── cache_manager.py       # Disk-backed LLM response cache (SHA-256 keyed)
│   ├── monitoring/
│   │   └── fallback_tracker.py    # Records model fallback events
│   ├── pipeline/
│   │   ├── offline_pipeline.py    # Orchestrates Ops 0 + 1 per database
│   │   └── online_pipeline.py     # Orchestrates Ops 5–9 per question
│   └── evaluation/
│       ├── evaluator.py           # compute_ex(), EvaluationResult, aggregate_metrics()
│       └── metrics.py             # Result normalization + comparison
├── scripts/
│   ├── run_offline_preprocessing.py   # Run/resume offline phase for all databases
│   ├── run_evaluation.py              # Evaluate against a BIRD split
│   ├── run_smoke_test.py              # Instrumented 66-question smoke test
│   ├── smoke_test_66q.sh              # Bash wrapper with pre-flight checks
│   └── analyze_results.py             # Print EX breakdown from a results JSON file
├── tests/
│   ├── test_bird_loader.py
│   ├── test_database.py
│   ├── test_profiler.py
│   ├── test_summarizer.py
│   ├── test_schema_formatter.py
│   ├── test_lsh_index.py
│   ├── test_faiss_index.py
│   ├── test_example_store.py
│   ├── test_offline_pipeline.py
│   ├── test_cache_manager.py
│   ├── test_context_grounder.py
│   ├── test_schema_linker.py
│   ├── test_reasoning_generator.py
│   ├── test_standard_generator.py
│   ├── test_icl_generator.py
│   ├── test_generation_diversity.py
│   ├── test_query_fixer.py
│   ├── test_adaptive_selector.py
│   ├── test_online_pipeline.py
│   ├── test_evaluator.py
│   └── e2e/
│       ├── test_bird_mini.py      # 66-question live smoke test (@pytest.mark.live)
│       └── test_bird_full.py      # 1 534-question full evaluation (@pytest.mark.live)
├── data/
│   ├── bird/
│   │   ├── dev/                   # BIRD dev set (1 534 questions, 11 databases)
│   │   ├── train/                 # BIRD train set (~9 428 questions, 84 databases)
│   │   └── mini_dev/              # BIRD mini-dev (500 questions)
│   ├── preprocessed/              # Offline artifacts (auto-generated)
│   │   ├── profiles/
│   │   ├── summaries/
│   │   ├── schemas/
│   │   └── indices/
│   └── cache/                     # LLM response cache (auto-generated)
├── New_NL2SQL_method_plan.md      # Full method design with ablation analysis
├── Phase1_implementation_details.md   # Phase 1 API-only implementation spec
├── Implementation_Guide_Constrained_Hardware.md
├── implementation_progress.json   # Step-by-step progress tracker
├── requirements.txt
└── .env                           # API keys and settings (not committed)
```

---

## Component Reference

### LLM Abstraction Layer

**Location:** `src/llm/`

All LLM calls go through a single async interface defined in `src/llm/base.py`:

```python
response = await get_client().generate(
    model="claude-haiku-4-5-20251001",   # or a list for fallback
    system=[CacheableText("...", cache=True)],
    messages=[{"role": "user", "content": "..."}],
    tools=[ToolParam(name="...", description="...", input_schema={...})],
    tool_choice_name="tool_name",        # force specific tool call
    thinking=ThinkingConfig(enabled=True, budget_tokens=8000),
    max_tokens=2000,
    temperature=0.7,
)
```

Key abstractions:

| Class | Purpose |
|-------|---------|
| `LLMClient` | Abstract base — `generate()` handles model fallback |
| `LLMResponse` | Normalized response: `tool_inputs`, `text`, `thinking`, token counts |
| `CacheableText` | System prompt block with optional `cache=True` hint |
| `ThinkingConfig` | Extended thinking / reasoning mode configuration |
| `ToolParam` | Provider-agnostic JSON Schema tool definition |
| `LLMError` | Base exception for all LLM failures |
| `LLMRateLimitError` | Triggers model fallback when raised |

**Model fallback:** Pass `model` as a `list[str]` to automatically try the next model on rate-limit errors.

**Retry:** Each provider implements 3 attempts with exponential backoff (2–30 s). Callers do not retry.

**Prompt caching:** `CacheableText(cache=True)` on large system blocks enables Anthropic's `cache_control: ephemeral`. The Gemini client ignores this hint without error.

Three providers are implemented:

| Provider | Class | Activated by |
|----------|-------|-------------|
| Anthropic (Claude) | `AnthropicClient` | `LLM_PROVIDER=anthropic` |
| Google Gemini | `GeminiClient` | `LLM_PROVIDER=gemini` |
| MLX local server | `MLXClient` | `LLM_PROVIDER=mlx` |

---

### Data Layer

**`src/data/bird_loader.py`** — Loads and parses BIRD dataset splits.

- `load_bird_split(split)` → `list[BirdEntry]`
- `BirdEntry`: `question_id`, `db_id`, `question`, `evidence`, `SQL`, `difficulty`
- Parses `tables.json` for schema metadata per database
- Tolerates alternative SQL field names across BIRD dataset versions

**`src/data/database.py`** — SQLite execution utilities.

- `execute_sql(db_path, sql, timeout_s=30)` → `ExecutionResult`
- `ExecutionResult`: `rows`, `columns`, `error`, `success`, `elapsed_ms`
- Timeout via `threading.Thread` + `conn.interrupt()` (macOS-compatible; avoids `SIGALRM`)

---

### Preprocessing

**`src/preprocessing/profiler.py`**

- `profile_database(db_path, db_id, output_dir, force)` → `DatabaseProfile`
- `DatabaseProfile` → dict of `TableProfile` → dict of `ColumnProfile`
- `ColumnProfile` fields: `total_count`, `null_count`, `null_rate`, `distinct_count`, `top_values`, `min_value`, `max_value`, `avg_value`, `avg_length`, `sql_type`, `minhash_bands`
- Numeric aggregates only for `INTEGER/REAL/NUMERIC`; text length only for `TEXT`
- Empty tables: guarded with `total_count=0, null_rate=0.0`
- Serialized via `dataclasses.asdict()` + `json.dump(default=str)`

**`src/preprocessing/summarizer.py`**

- `summarize_database(profile, output_dir)` → `DatabaseSummary`
- `DatabaseSummary` → dict of `(table, column)` → `FieldSummary(short, long)`
- Batches columns (up to 6/call) to reduce API costs
- Individual retry on batch failure; deterministic default on total failure
- Length enforcement: short summaries sliced to 200 chars, long to 1 000 chars

**`src/preprocessing/schema_formatter.py`**

- `format_and_save_schemas(profile, summary, output_dir)` → `FormattedSchemas`
- `FormattedSchemas`: `ddl` (str), `markdown` (str)
- DDL: long summary truncated to 120 chars with `...`; sample values to 50 chars
- Markdown: first 3 sample values, each truncated to 30 chars
- FK annotations injected as `-- Foreign keys: table.col → ref_table.ref_col`
- `needs_quoting()` uses `^[A-Za-z0-9_]+$` to detect identifiers requiring backtick quoting

---

### Indexing

**`src/indexing/lsh_index.py`**

- `LSHIndex.build(db_path, db_id)` — scans all text columns, caps at 50 000 distinct values/column
- `LSHIndex.query(keyword, top_k)` → `list[CellMatch]`
- `CellMatch`: `table`, `column`, `value`, `similarity_score`, `exact_match`
- 3-gram MinHash + `datasketch.MinHashLSH` (threshold=0.3)
- Post-retrieval Jaccard re-ranking; exact matches sort first
- Serialized via `pickle`

**`src/indexing/faiss_index.py`**

- `FAISSIndex.build(field_summaries)` — embeds long summaries with `all-MiniLM-L6-v2`
- `FAISSIndex.query(question, top_k)` → `list[FieldMatch]`
- `FieldMatch`: `table`, `column`, `score`, `short_summary`, `long_summary`
- Scores clipped to `[0, 1]` with `np.clip`
- Saved as `.index` (FAISS binary) + `_fields.json` (metadata)

**`src/indexing/example_store.py`**

- `ExampleStore.build(train_entries)` — builds skeleton-masked embeddings
- `ExampleStore.query(question, db_id, top_k)` → `list[ExampleEntry]` (excludes same db_id)
- Skeleton masking pipeline: spaCy NER (`en_core_web_sm`) + regex for `[NUM]`, `[STR]`, `[ENTITY]`; overlapping spans resolved greedily; applied in reverse order
- `ExampleEntry`: `question`, `sql`, `db_id`, `difficulty`, `masked_question`

---

### Grounding

**`src/grounding/context_grounder.py`**

- `ground_context(question, evidence, db_id, lsh_index, example_store)` → `GroundingContext`
- `GroundingContext`: `matched_cells`, `schema_hints`, `few_shot_examples`
- LLM call uses `model_fast` with `temperature=0.0` (deterministic extraction)
- Deduplication: when two literals map to the same `(table, column)`, keeps highest similarity
- Multi-word literals are also queried word-by-word (words ≥ 3 chars)
- Empty/None evidence normalized to `"None"` before passing to LLM

---

### Schema Linking

**`src/schema_linking/schema_linker.py`**

- `link_schema(question, evidence, grounding_context, faiss_index, full_ddl, full_markdown, available_fields)` → `LinkedSchemas`
- `LinkedSchemas`: `s1_ddl`, `s1_markdown`, `s2_ddl`, `s2_markdown`, `s1_fields`, `s2_fields`
- FAISS top-50 + grounding schema hints form the initial candidate set
- Two LLM calls to `model_powerful`; second call with remaining candidates (never skipped)
- PK/FK auto-inclusion: parses `-- Table:` / `-- Foreign keys:` DDL comments
- Hallucination guard: validates every `(table, column)` against `available_field_set`
- S₁ ⊆ S₂ invariant enforced at output
- S₁ and S₂ field lists are sorted tuples for stable, deterministic output

---

### SQL Generation

**`src/generation/base_generator.py`**

Shared utilities available to all generators:

- `clean_sql(text)` — strips ` ```sql ``` ` fences, trailing semicolons, collapses whitespace
- `validate_sql_syntax(sql)` — checks for `SELECT` and `FROM` (case-insensitive)
- `build_base_prompt(question, evidence, cell_matches)` — formats question + evidence + matched values as `table.column = 'value'`
- **SQL Writing Rules** injected into all generator system prompts (anti-hallucination guardrails)

**`src/generation/reasoning_generator.py`**

- `ReasoningGenerator.generate(...)` → `list[SQLCandidate]` (4 candidates)
- All 4 calls run concurrently via `asyncio.gather()`
- Adaptive thinking budget based on schema table count
- SQL extracted from `response.text` (not tool-use)
- `generator_id` format: `"reasoning_A1"` through `"reasoning_A4"`

**`src/generation/standard_generator.py`**

- `StandardAndComplexGenerator.generate(...)` → `list[SQLCandidate]` (4 candidates)
- All 4 calls (B1a, B1b, B2a, B2b) run concurrently
- Both B1 and B2 use Markdown schemas; thinking disabled
- B2 prompt emphasizes CTEs, window functions, subqueries
- `generator_id` format: `"standard_B1_s1"`, `"complex_B2_s2"`, etc.

**`src/generation/icl_generator.py`**

- `ICLGenerator.generate(...)` → `list[SQLCandidate]` (2–3 candidates)
- Few-shot examples formatted as `## Example N` blocks
- Examples block uses `CacheableText(cache=True)` (shared across C1/C2/C3)
- Token guard: `len(formatted_examples) // 4 > 6000` → trims to first 6 examples
- `generator_id` format: `"icl_C1"`, `"icl_C2"`, `"icl_C3"`

**`SQLCandidate`** dataclass fields: `sql`, `generator_id`, `schema_used` (`"s1"`/`"s2"`), `schema_format` (`"ddl"`/`"markdown"`), `reasoning_trace`, `error_flag`.

---

### Query Fixer + Semantic Verifier

**`src/fixing/query_fixer.py`**, **`src/verification/query_verifier.py`**

- `QueryFixer.fix_candidates(candidates, question, evidence, schemas, db_path, cell_matches)` → `list[FixedCandidate]`
- `QueryFixer(verifier=...)` — optional verifier injection for testing
- All candidates fixed concurrently via `asyncio.gather()`
- Fix loop: β + 1 = 3 iterations (2 fix attempts + 1 final assessment)
- Stage A: executability check; Stage B: semantic verification (9 test types)
- Fix prompt includes combined execution + verification feedback
- `QueryVerifier.generate_plan(question, evidence, schema)` → 1 LLM call (question-level)
- `QueryVerifier.evaluate_candidate(specs, candidate_id, sql, exec_result, db_path, run_expensive)` → `VerificationEvaluation`
- Cheap tests (grain, null, duplicate, ordering, scale, completeness): SQL/structural, zero LLM cost
- Expensive tests (column_alignment, boundary, symmetry): `model_fast` LLM judgment, final pass only
- `FixedCandidate` fields: `original_sql`, `final_sql`, `generator_id`, `fix_iterations`, `confidence_score`, `execution_result`, `verification_results`

---

### Adaptive Selector

**`src/selection/adaptive_selector.py`**

- `AdaptiveSelector.select(candidates, question, evidence, schemas, db_path)` → `SelectionResult`
- Clustering: `str(sorted_rows)` as cluster key (order-independent equivalence)
- Row normalization: `None` → `"NULL_SENTINEL"`, `bool` → `str(int(v))` (before `int` check), `float` → `f"{v:.6f}"`
- `syncio.get_event_loop().run_in_executor` wraps synchronous `execute_sql` to avoid blocking the event loop
- Tournament via tool-use `select_winner` tool; `tool_choice_name="select_winner"` forces structured output
- `SelectionResult` fields: `final_sql`, `selection_method`, `tournament_wins`, `confidence`, `cluster_count`, `candidates_evaluated`

---

### Cache Manager

**`src/cache/cache_manager.py`**

- `CacheManager` — disk-backed LLM response cache
- Cache key: `SHA256(model + json.dumps(messages, sort_keys=True))` — deterministic regardless of dict key order
- Cache files: `{key[:16]}.json`; full key stored inside for collision detection
- TTL checked at read time; no proactive eviction
- `enabled=False` short-circuits with no disk I/O
- `@cached()` decorator for wrapping LLM call coroutines
- Toggled via `CACHE_LLM_RESPONSES=true` in `.env`

---

### Evaluation

**`src/evaluation/evaluator.py`**

- `compute_ex(predicted_sql, truth_sql, db_path)` → `bool`
- Guards for empty/whitespace predicted SQL before any DB calls
- Truth SQL failure logs a warning and returns `False`
- `EvaluationResult` dataclass: `question_id`, `db_id`, `difficulty`, `predicted_sql`, `correct`, `fix_count`, `cost_estimate`
- `aggregate_metrics(results)` → dict with overall EX, per-difficulty EX, per-database EX, selection method distribution

**`src/evaluation/metrics.py`**

- `compare_results(rows_a, rows_b)` → `bool`
- Column-order independence: values within each row are sorted, then the list of row tuples is sorted
- `_normalize_cell()`: `None` → `"NULL_SENTINEL"`, `bool` → `str(int(v))`, `float` → `Decimal` 6 d.p., everything else → `str(v)`

---

## Configuration

All settings are managed by `src/config/settings.py` (Pydantic v2 + `pydantic-settings`).
Create a `.env` file in the project root:

```dotenv
# Provider selection
LLM_PROVIDER=anthropic          # anthropic | gemini | mlx

# API keys
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AI...

# Model tier overrides (leave blank for provider defaults)
# Supports comma-separated fallback lists: MODEL_POWERFUL=gemini-2.5-pro,gemini-2.5-flash
MODEL_FAST=
MODEL_POWERFUL=
MODEL_REASONING=

# MLX local server (only when LLM_PROVIDER=mlx)
MLX_SERVER_URL=http://127.0.0.1:8080
MLX_MODEL_NAME=mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit

# Paths
BIRD_DATA_DIR=./data/bird
PREPROCESSED_DIR=./data/preprocessed
CACHE_DIR=./data/cache

# Generation
MAX_CANDIDATES=11
QUERY_FIX_ITERATIONS=2
ICL_EXAMPLES_COUNT=8

# Schema linking
SCHEMA_LINKER_FAISS_TOP_K=50

# Caching
CACHE_LLM_RESPONSES=false       # set to true to cache LLM responses to disk

# Logging
LOG_LEVEL=INFO
```

### Provider defaults

| Task | Anthropic default | Gemini default |
|------|-------------------|----------------|
| `model_fast` (keyword extraction, field summarization, query fixing, B1, pairwise selection) | `claude-haiku-4-5-20251001` | `gemini-2.5-flash` |
| `model_powerful` (schema linking, B2, ICL generator) | `claude-sonnet-4-6` | `gemini-2.5-pro` |
| `model_reasoning` (Generator A with extended thinking) | `claude-sonnet-4-6` | `gemini-2.5-flash` |

---

## Setup & Installation

**Requirements:** Python 3.11+, Apple Silicon (for MLX) or Linux (for Gemini/Anthropic API use)

```bash
# 1. Create conda environment (recommended)
conda create -n NL2SQL python=3.11
conda activate NL2SQL

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install spaCy language model (required by example_store.py)
python -m spacy download en_core_web_sm

# 4. Create your .env file
cp .env.example .env   # then fill in your API keys
```

---

## Dataset Setup

Download the BIRD dataset and place it under `data/bird/`:

```
data/bird/
├── dev/
│   ├── dev.json                   # 1 534 questions
│   ├── dev_tables.json
│   └── dev_databases/
│       ├── california_schools/
│       │   └── california_schools.sqlite
│       ├── card_games/
│       │   └── card_games.sqlite
│       └── ...  (11 databases total)
├── train/
│   ├── train.json                 # ~9 428 questions
│   ├── train_tables.json
│   └── train_databases/
│       └── ...  (84 databases)
└── mini_dev/
    ├── mini_dev.json              # 500 questions
    └── ...
```

---

## Running the System

### Step 1 — Run the offline pipeline (one-time)

```bash
# Process all 11 dev databases + build example store from training data
python scripts/run_offline_preprocessing.py --split dev

# Force re-run (ignore cached artifacts)
python scripts/run_offline_preprocessing.py --split dev --force

# Process a single database only
python scripts/run_offline_preprocessing.py --split dev --db california_schools
```

Artifacts are saved under `data/preprocessed/` and loaded automatically on subsequent runs.

### Step 2 — Run evaluation

```bash
# Full dev evaluation (1 534 questions)
python scripts/run_evaluation.py --split dev

# With 8 concurrent workers
python scripts/run_evaluation.py --split dev --workers 8

# Resume a partial run
python scripts/run_evaluation.py --split dev --resume results/dev_results.json

# Single database
python scripts/run_evaluation.py --split dev --db_filter california_schools

# Mini-dev (500 questions, faster)
python scripts/run_evaluation.py --split mini_dev --workers 8
```

Results are saved incrementally to a JSON file and a summary table is printed at the end
(broken down by difficulty and by database).

### Step 3 — Analyze results

```bash
python scripts/analyze_results.py results/dev_results.json
```

Prints:
- Overall EX accuracy
- EX by difficulty (simple / moderate / challenging)
- EX by database (sorted ascending to highlight worst performers)
- Worst 3 database details with incorrect question snippets

### Step 4 — Smoke test (66 stratified questions)

```bash
# Full smoke test with pre-flight checks
bash scripts/smoke_test_66q.sh

# Or directly (66 questions: 6 per database × 11 databases, stratified, seed=42)
python scripts/run_smoke_test.py --workers 4
```

The smoke test writes:
- `results.json` — per-question results
- `detailed_traces.json` — full per-op traces (cell matches, schema recall, candidates, fix iterations, tournament wins)
- `component_summary.json` — aggregate stats per component
- `failed_questions.json` — details of incorrect answers

### Single-question API usage

```python
import asyncio
from src.data.bird_loader import BirdEntry
from src.pipeline.offline_pipeline import run_offline_pipeline
from src.pipeline.online_pipeline import answer_question

async def main():
    # Load offline artifacts (from cache if available)
    artifacts = await run_offline_pipeline(
        db_id="california_schools",
        db_path="data/bird/dev/dev_databases/california_schools/california_schools.sqlite",
        train_data=[],  # pass training entries for example store
    )

    entry = BirdEntry(
        question_id=1,
        db_id="california_schools",
        question="How many schools are in Alameda County?",
        evidence="",
        SQL="",
        difficulty="simple",
    )

    result = await answer_question(entry, artifacts, db_path=artifacts.db_path)
    print(result.final_sql)
    print(f"Selection method: {result.selection_method}")

asyncio.run(main())
```

---

## Testing

The test suite uses **pytest + pytest-asyncio + pytest-mock**.
All 296 unit tests mock LLM calls via `patch.object(get_client(), "_generate_single")`.

```bash
# Run all unit tests
/path/to/conda/envs/NL2SQL/bin/python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_schema_linker.py -v

# Run only fast (non-live) tests
python -m pytest tests/ -v -m "not live"

# Run live API tests (requires .env with real API keys)
python -m pytest tests/e2e/test_bird_mini.py -v -m live
```

**Sampling rule for all tests:**
Never use "first N questions" — BIRD dev questions are ordered by database, so the first N would cover only one database. Always use stratified random sampling across all 11 databases with `random.seed(42)`:

- Tests originally ≤ 10 questions → **33 total** (3 per DB: 1 simple, 1 moderate, 1 challenging)
- Tests originally at 50 questions → **66 total** (6 per DB: 2 simple, 2 moderate, 2 challenging)

---

## Performance & Design Decisions

### Inference cost profile

| Stage | LLM Calls | Notes |
|-------|-----------|-------|
| Context Grounding (Op 5) | 1 | Keyword extraction with fast model |
| Schema Linking (Op 6) | 2–3 | FAISS (free) + 2 LLM column selection calls |
| SQL Generation (Op 7) | 10–11 | 4 reasoning + 4 standard/complex + 2–3 ICL |
| Query Fixing (Op 8) | 1–23 | 1 plan generation + conditional fixes; β=2 max; ~3–6 on average |
| Selection (Op 9) | 0–28 | Fast path: 0; worst case C(8,2)=28 |
| **Total** | **14–68** | Average ~21–36 |

### Key design decisions

| Decision | Rationale |
|----------|-----------|
| Extended thinking via **prompt variation** (not temperature) for Generator A | Anthropic API forces temperature=1 when thinking is enabled; prompt variants (minimal vs. CoT) provide diversity instead |
| **Prompt caching** on schema/field-summary blocks | Saves ~60% of schema-linking and generation costs for repeated databases |
| **Confidence scoring** formula: +1.0 success +0.5 plausibility −0.5/fix | Composite score that rewards correctness, penalizes uncertainty, and rewards clean generation |
| **Fast path** for unanimous candidates | ~45% of questions handled with 0 selection LLM calls (XiYan analysis) |
| **Representatives only** in tournament (1 per cluster, shortest SQL) | Reduces worst-case comparisons from 561 (Agentar) to 28 while preserving tournament quality |
| **Two schemas S₁ (precise) and S₂ (recall)** | Structured diversity axis — generators can exploit both focused and broad context simultaneously |
| **Example store excludes same-db_id examples** | Prevents schema leakage from training databases into few-shot demonstrations |
| **LSH threshold = 0.3** (not 0.5) | True 3-gram Jaccard for single-transposition typos is ~0.47; 0.5 would miss them. Post-retrieval re-ranking restores precision. |
| **β = 2 fix iterations** (not 3) | Diminishing returns: the vast majority of fixable errors are resolved on the first iteration |

### Phase 1 vs. full method

Phase 1 replaces the locally fine-tuned RL generators with API calls:

| Full method component | Phase 1 replacement |
|----------------------|---------------------|
| RL-GRPO Reasoning Generator (32B) | Claude Sonnet with extended thinking |
| Multi-task FT Generators (2 × 32B) | Claude Haiku (B1) + Claude Sonnet (B2) with different prompts |
| RL Selection Model (7B) | Claude Haiku via pairwise tool-use |

Expected Phase 1 accuracy: **~70–72% EX** on BIRD dev (1 534 questions).

### Source methods comparison

| Method | Final EX | Avg API Calls | Our usage |
|--------|:--------:|:-------------:|-----------|
| Agentar-Scale-SQL | 74.90% | ~170 | RL reasoning paradigm |
| Automatic Metadata Extraction | 63.2%* | ~25 | Profiling + LLM summarization |
| CHASE-SQL | 73.01% | ~480 | Query fixer, pairwise selection design |
| XiYan-SQL | 73.34% | ~12–15 | Multi-task training, candidate reorganization, adaptive fast path |
| **This system (Phase 1)** | **~70–72%** | **~20–35** | Combines all of the above |

*Auto Metadata on MiniDev (500 q) with GPT-4o; others on BIRD dev (1 534 q).
