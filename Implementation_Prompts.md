# Implementation Prompts — Phase 1 NL2SQL

## How to Use This File

Copy each prompt verbatim into Claude Code. After each **Implementation Prompt**, review the code before moving to the next one. Use the **Refinement Checkpoints** to run real tests and discuss changes before continuing.

**Rules:**
- Never skip to the next prompt if tests aren't passing for the current one.
- At every checkpoint, ask Claude: "What would you change about what we just built?"
- If Claude flags a design issue during implementation, discuss it before continuing.
- Run `/compact focus on NL2SQL Phase 1 implementation` before prompts 8, 12, and 15 if the session is long.

---

## Prompt 1 of 15 — Project Foundation

```
Read CLAUDE.md and Phase1_implementation_details.md to understand the project.

Then implement the project foundation — every subsequent component depends on this:

Files to create:
1. `pyproject.toml` — project metadata + all dependencies from the "Environment Setup" section of Phase1_implementation_details.md. Use pyproject.toml format with [project] and [project.optional-dependencies] for test deps.
2. `src/config/settings.py` — Pydantic settings class exactly as specified in Step 1.1 of the implementation guide.
3. `src/data/bird_loader.py` — BirdEntry pydantic model + load_bird_split() + DatabaseSchema model + load_schema(). BirdEntry fields: question_id, db_id, question, evidence, SQL, difficulty.
4. `src/data/database.py` — ExecutionResult dataclass + execute_sql() with 30-second timeout, handling syntax errors, runtime errors, empty results, and Cartesian product protection.
5. `tests/conftest.py` — shared pytest fixtures: a small in-memory SQLite database (students table as specified in the profiler tests section), a sample BirdEntry, and a temporary directory for preprocessed artifacts.
6. `tests/unit/test_bird_loader.py` — 5 tests: load returns list of BirdEntry, all fields populated, db_id is string, difficulty in valid set, evidence can be empty string.
7. `tests/unit/test_database.py` — 6 tests: valid SELECT returns rows, syntax error returns success=False with error message, empty result returns is_empty=True, timeout kills hanging query, execution_time is measured, Cartesian product exceeding row limit returns error.

All __init__.py files for every src/ subdirectory.

After implementing, run: pytest tests/unit/test_bird_loader.py tests/unit/test_database.py -v

Report: which tests pass, and flag any design decisions you made that deviate from the spec or that you think should be discussed before we continue.
```

---

## Prompt 2 of 15 — Database Profiler (Op 0a)

```
Read CLAUDE.md. The project foundation (Prompt 1) is already implemented.

Implement the database statistical profiler — Sub-Operation 0a from Phase1_implementation_details.md.

Files to create:
1. `src/preprocessing/profiler.py` — implements ColumnProfile and DatabaseProfile dataclasses + profile_database(db_path, db_id) function. Must compute: total_count, null_count, null_rate, distinct_count, data_type, sample_values (top-10 by frequency), min/max/avg for numerics, avg/max length for text, is_primary_key, foreign_key_ref, and minhash_bands using datasketch.MinHash with num_perm=128 and character 3-grams. Output is cached to data/preprocessed/profiles/{db_id}.json.
2. `tests/unit/test_profiler.py` — all 10 tests exactly as specified in the "Tests for Sub-Operation 0a" section of Phase1_implementation_details.md.

Use the conftest.py students table fixture for tests. The profiler must handle: NULL values in any column, empty tables, tables with only 1 row, and columns with all identical values.

After implementing, run: pytest tests/unit/test_profiler.py -v

All 10 tests must pass. Report: test results + any design decisions or concerns to discuss.
```

---

## Prompt 3 of 15 — Field Summarizer (Op 0b)

```
Read CLAUDE.md. Ops 0a is implemented and tested.

Implement the LLM-based field summarizer — Sub-Operation 0b from Phase1_implementation_details.md.

Files to create:
1. `src/preprocessing/summarizer.py` — implements FieldSummary and DatabaseSummary dataclasses + summarize_database(profile: DatabaseProfile) -> DatabaseSummary. Must:
   - Use claude-haiku-4-5-20251001 with tool-use (not free text) to get structured short + long summaries
   - Batch 5-8 columns per API call, grouped by table
   - Apply prompt caching (cache_control: ephemeral) on the system prompt's schema context block
   - Retry with tenacity (3 attempts, exponential backoff) on API errors
   - Cache output to data/preprocessed/summaries/{db_id}.json and skip if file exists
   - Define the summarize_fields tool schema exactly as described in the implementation guide

2. `tests/unit/test_summarizer.py` — all 10 tests from the "Tests for Sub-Operation 0b" section. Use pytest-mock to mock the Anthropic client. The mock should return a realistic tool-use response object that matches the actual Anthropic API response structure.

The mock must correctly simulate: (a) successful tool_use response, (b) API error on first 2 attempts then success, (c) cache hit (file already exists on disk).

After implementing, run: pytest tests/unit/test_summarizer.py -v

All 10 tests must pass. Report: results + any design decisions.

Note: Do NOT make real API calls during tests. All 10 tests must be mockable without a key.
```

---

## Prompt 4 of 15 — Schema Formatter + Offline Script (Op 0c)

```
Read CLAUDE.md. Ops 0a + 0b are implemented and tested.

Implement the schema formatter and offline preprocessing script — Sub-Operation 0c from Phase1_implementation_details.md.

Files to create:
1. `src/preprocessing/schema_formatter.py` — format_schemas(profile, summary) -> FormattedSchemas. FormattedSchemas has: ddl (str), markdown (str). DDL format: CREATE TABLE with column names, types, long summaries as inline comments, PK notation, sample row comment, FK trailing comments. Markdown format: ## Table headers, pipe-table with Column/Type/Description/Sample Values columns. Both formats must handle: column names with spaces (quoted in DDL), special chars, truncate sample values >30 chars with "..." in Markdown. Save to data/preprocessed/schemas/{db_id}_ddl.sql and {db_id}_markdown.md.

2. `tests/unit/test_schema_formatter.py` — all 10 tests from the "Tests for Sub-Operation 0c" section.

3. `scripts/run_offline_preprocessing.py` — CLI script with --split (dev/train/mini_dev) and --step (all/profile/summarize/format/indices) flags. For now implement the profile+summarize+format steps. Must: show tqdm progress bars, skip already-cached databases, handle partial failures gracefully (log error, continue with next DB), print cost estimate at end (token counts from Anthropic response headers).

After implementing, run: pytest tests/unit/test_schema_formatter.py -v

All 10 tests must pass. Report: results + anything in the schema output format that could be improved before we build the generators that consume it.
```

---

## ★ Refinement Checkpoint A — Review Ops 0a, 0b, 0c

```
We've finished Ops 0a, 0b, 0c. Before building the indices, I want to do a real test.

1. Run `scripts/run_offline_preprocessing.py --split dev --step profile` on one real BIRD dev database. If the BIRD data isn't downloaded yet, show me the exact commands to download and extract it from Hugging Face.

2. Show me the first 50 lines of the generated profile JSON for one database.

3. Show me the DDL and Markdown schema output for one table from that database.

4. Looking at the actual output, what would you change about:
   - The profiler's statistics (anything missing or misleading)?
   - The summarizer's prompt design (can we get better summaries)?
   - The schema format (will generators understand this well)?

Propose specific changes and I'll tell you which ones to implement before we move to Prompt 5.
```

---

## Prompt 5 of 15 — LSH Cell Value Index (Op 1a)

```
Read CLAUDE.md. Ops 0a, 0b, 0c are implemented and tested. Any refinements from Checkpoint A have been applied.

Implement the LSH cell value index — Sub-Operation 1a from Phase1_implementation_details.md.

Files to create:
1. `src/indexing/lsh_index.py` — CellMatch dataclass + LSHIndex class with:
   - build(db_path, db_id) — collects all distinct non-NULL string values per (table, column), creates MinHash with 128 perms and 3-gram shingling, inserts into MinHashLSH with threshold=0.5
   - query(keyword, top_k=5) -> list[CellMatch] — queries LSH, returns ranked matches
   - save(path) / load(path) — pickle serialization
   - Index is per-database (not shared)
   - Numeric values are included as their string representation
   - NULL values are excluded from indexing

2. `tests/unit/test_lsh_index.py` — all 10 tests from the "Tests for Sub-Operation 1a" section. Use an in-memory SQLite test database (not BIRD data) for tests. The 10,000-row speed test should use a programmatically generated SQLite table.

After implementing, run: pytest tests/unit/test_lsh_index.py -v

All 10 tests must pass. Report: results + observed matching quality (e.g., does "Untied States" actually match "United States"? what threshold works best in practice?).
```

---

## Prompt 6 of 15 — FAISS Index + Example Store + Offline Pipeline (Ops 1b, 1c)

```
Read CLAUDE.md. Ops 0 and 1a are implemented and tested.

Implement the FAISS semantic index, example vector store, and offline pipeline orchestrator — Sub-Operations 1b and 1c from Phase1_implementation_details.md.

Files to create:
1. `src/indexing/faiss_index.py` — FieldMatch dataclass + FAISSIndex class:
   - build(field_summaries: list[FieldSummary]) — embed long_summary for each field using all-MiniLM-L6-v2, build faiss.IndexFlatIP (normalize embeddings for cosine sim)
   - query(question, top_k=30) -> list[FieldMatch]
   - save(index_path, fields_path) / load(index_path, fields_path)

2. `src/indexing/example_store.py` — ExampleEntry dataclass + ExampleStore class:
   - build(train_entries: list[BirdEntry]) — apply skeleton masking (spaCy NER for [ENTITY], regex for [NUM] and [STR]), embed skeletons with all-MiniLM-L6-v2, build FAISS index
   - query(question, db_id, top_k=8) -> list[ExampleEntry] — mask query, search, exclude same db_id examples
   - save(faiss_path, meta_path) / load(faiss_path, meta_path)

3. `src/pipeline/offline_pipeline.py` — OfflineArtifacts dataclass + run_offline_pipeline(db_id, db_path, train_data) async function that orchestrates Ops 0+1 for one database. Loads from cache if all artifacts exist.

4. `tests/unit/test_faiss_index.py` — all 10 tests from the "Tests for Sub-Operation 1b" section.
5. `tests/unit/test_example_store.py` — all 10 tests from the "Tests for Sub-Operation 1c" section.
6. `tests/integration/test_offline_pipeline.py` — 4 integration tests from the "Integration Test: Offline Pipeline" section. Mock the summarizer API calls.

Extend `scripts/run_offline_preprocessing.py` to also run the index-building step (--step indices).

After implementing, run: pytest tests/unit/test_faiss_index.py tests/unit/test_example_store.py tests/integration/test_offline_pipeline.py -v

All tests must pass. Report: results + whether the skeleton masking is working as expected (show 3 example question→skeleton transformations).
```

---

## ★ Refinement Checkpoint B — Run Full Offline Pipeline on Real Data

```
Now let's run the complete offline pipeline on real BIRD data.

1. Run the full offline pipeline on all BIRD dev databases:
   python scripts/run_offline_preprocessing.py --split dev --step all

2. Report: how many databases processed, total API cost (from token counts), any failures.

3. Pick the california_schools database and show me:
   - How many fields are in the LSH index
   - A FAISS query result for "county name" (top 5 fields)
   - 3 example questions from the train set that would be retrieved for a question about "free meal counts in schools"

4. Looking at these real results, what would you change? Consider:
   - Are the field summaries good enough for schema linking?
   - Is the skeleton masking too aggressive or not aggressive enough?
   - Is 30 fields from FAISS the right pre-filtering amount?

Propose changes and I'll tell you which ones to implement before Prompt 7.
```

---

## Prompt 7 of 15 — Cache Manager + Context Grounder (Op 5)

```
Read CLAUDE.md. Offline pipeline (Ops 0+1) is complete and tested on real data. Any refinements from Checkpoint B have been applied.

Implement the cache manager and context grounder — from Phase1_implementation_details.md.

Files to create:
1. `src/cache/cache_manager.py` — CacheManager class:
   - Disk cache keyed by SHA256(model_name + prompt_content)
   - Stores/retrieves raw Anthropic API response objects as JSON
   - Configurable via CACHE_LLM_RESPONSES env var (default: False)
   - Optional TTL (None = never expire, int = seconds)
   - @cache_manager.cached(model, ttl) decorator for async functions

2. `src/grounding/context_grounder.py` — GroundingContext dataclass + ground_context(question, evidence, db_id, lsh_index, example_store) async function:
   - Uses claude-haiku-4-5-20251001 with extract_grounding tool (literals + schema_references)
   - Queries LSH index for each literal (top_k=5), deduplicates by table.column
   - Masks question to skeleton, queries example store (top_k=8, excludes db_id)
   - Returns GroundingContext(matched_cells, schema_hints, few_shot_examples)
   - Uses CacheManager if enabled

3. `tests/unit/test_cache_manager.py` — all 7 tests from the "Tests for Cache Manager" section.
4. `tests/unit/test_context_grounder.py` — all 10 tests from the "Tests for Context Grounding" section. Mock the Anthropic client and LSH index.

After implementing, run: pytest tests/unit/test_cache_manager.py tests/unit/test_context_grounder.py -v

All tests must pass. Report: results + anything about the keyword extraction prompt that should be improved (what might Haiku miss vs. what Sonnet would catch?).
```

---

## Prompt 8 of 15 — Schema Linker (Op 6)

```
Read CLAUDE.md. Context grounder is implemented and tested.

Implement the adaptive schema linker — Operation 6 from Phase1_implementation_details.md.

Files to create:
1. `src/schema_linking/schema_linker.py` — LinkedSchemas dataclass + link_schema(question, evidence, grounding_context, faiss_index, full_ddl, full_markdown) async function:
   - Step 6.1: Query FAISS for top_k=30 candidate fields
   - Step 6.2: Call claude-sonnet-4-6 with select_columns tool (precise selection = S₁). Auto-add PKs for selected tables. Auto-add FKs that bridge selected tables.
   - Step 6.3: Call claude-sonnet-4-6 again with remaining fields (recall expansion = S₂). S₂ = S₁ ∪ new fields ∪ their keys.
   - Step 6.4: Render S₁ and S₂ as filtered DDL and Markdown subsets of the full schemas.
   - Step 6.5: Apply prompt caching on the candidate field summaries block (the same fields list appears across questions for the same database).
   - Validate: all selected fields must exist in the original schema (no hallucinations). Log a warning + filter out any hallucinated field names.

2. `tests/unit/test_schema_linker.py` — all 12 tests from the "Tests for Schema Linking" section. Mock the Claude API and FAISS index. Construct a realistic 40-column, 5-table mock database schema for the tests.

After implementing, run: pytest tests/unit/test_schema_linker.py -v

All 12 tests must pass. Report: results + specifically comment on: (1) Does the S₁ ⊆ S₂ invariant hold under all test scenarios? (2) Is there a case where prompt caching won't apply and what should we do about it?
```

---

## ★ Refinement Checkpoint C — Test Grounding + Schema Linking End-to-End

```
Let's test Ops 5+6 together on real BIRD questions before building the generators.

Enable the cache (CACHE_LLM_RESPONSES=true) and run the pipeline through Op 6 on these 5 BIRD dev questions (first 5 from dev.json). Show me for each question:

1. The extracted keywords and schema hints (Op 5 output)
2. Which fields were selected for S₁ and S₂ (Op 6 output)
3. The actual S₁ Markdown schema rendered (truncated to first 20 lines)

Then evaluate:
- Are the right fields being selected? (Compare against ground truth SQL to see if needed columns are in S₂)
- Is FAISS pre-filtering missing anything that ends up in the correct SQL?
- Are there hallucinated column names getting through the validation step?

Propose specific improvements. For any improvement that could affect later components (generators consume S₁ and S₂), flag it so I can decide before we build generators.
```

---

## Prompt 9 of 15 — Base Generator + Reasoning Generator (Op 7A)

```
Read CLAUDE.md. Ops 0, 1, 5, 6 are implemented and tested. Any refinements from Checkpoint C have been applied.

Implement the base generator utilities and reasoning generator — shared base + Sub-Operation 7A from Phase1_implementation_details.md.

Files to create:
1. `src/generation/base_generator.py`:
   - SQLCandidate dataclass: sql (str), generator_id (str), schema_used (str: "s1"/"s2"), schema_format (str: "ddl"/"markdown"), reasoning_trace (str|None), error_flag (bool)
   - clean_sql(raw: str) -> str: strip markdown code fences (```sql ... ```), strip trailing semicolons, normalize whitespace
   - validate_sql_syntax(sql: str) -> bool: regex check for basic SELECT structure (not execution-based)
   - build_base_prompt(question, evidence, cell_matches) -> str: shared user prompt portion

2. `src/generation/reasoning_generator.py` — ReasoningGenerator class:
   - generate(question, evidence, schemas: LinkedSchemas, grounding: GroundingContext) -> list[SQLCandidate]
   - Uses claude-sonnet-4-6 with extended thinking (budget_tokens adaptive: 4000 for 1-2 tables, 6000 for 3-4, 8000 for 5+)
   - 4 candidates: A1 (S₁ DDL, minimal prompt), A2 (S₁ DDL, step-by-step prompt), A3 (S₂ DDL, minimal), A4 (S₂ DDL, step-by-step). This achieves diversity via prompt variation since extended thinking forces temperature=1.
   - Extracts SQL from the response text after the thinking block
   - Populates reasoning_trace from the thinking block content
   - All 4 calls run concurrently via asyncio.gather()
   - generator_id format: "reasoning_A1", "reasoning_A2", etc.

3. `tests/unit/test_reasoning_generator.py` — all 10 tests from the spec. Mock must realistically simulate the extended thinking API response format (response has a thinking block followed by a text block).

After implementing, run: pytest tests/unit/test_reasoning_generator.py -v

All 10 tests must pass. Report: results + show me what the mock extended thinking response structure looks like so I can verify it matches the real Anthropic API format.
```

---

## Prompt 10 of 15 — Standard, Complex + ICL Generators (Ops 7B, 7C)

```
Read CLAUDE.md. Reasoning generator (7A) is implemented and tested.

Implement the standard generator, complex generator, and ICL generator — Sub-Operations 7B and 7C from Phase1_implementation_details.md.

Files to create:
1. `src/generation/standard_generator.py` — StandardAndComplexGenerator class:
   - Generator B1 (Haiku, standard prompt): 2 candidates (B1a=S₁ markdown, B1b=S₂ markdown)
   - Generator B2 (Sonnet, complex SQL prompt emphasizing CTEs/window functions): 2 candidates (B2a=S₁ markdown, B2b=S₂ markdown)
   - Both run concurrently. generator_ids: "standard_B1_s1", "standard_B1_s2", "complex_B2_s1", "complex_B2_s2"

2. `src/generation/icl_generator.py` — ICLGenerator class:
   - Formats 8 few-shot examples as "## Example N\nQuestion: ...\nEvidence: ...\nSQL: ..." in system prompt
   - Applies prompt caching on the examples block
   - Cost guard: if examples block > 6000 tokens (use tiktoken to estimate), trim to top-6
   - 3 candidates: C1 (direct), C2 (CoT: "First identify tables and joins, then write SQL"), C3 (step-back: "What SQL pattern applies? Then write it")
   - Always uses S₂ Markdown schema
   - All 3 run concurrently. generator_ids: "icl_C1", "icl_C2", "icl_C3"

3. `tests/unit/test_standard_generator.py` — all 10 tests from the spec.
4. `tests/unit/test_icl_generator.py` — all 10 tests from the spec.
5. `tests/integration/test_generation_diversity.py` — first 4 tests from the spec (mock API). Save test 5 (live upper bound test) for Checkpoint D.

After implementing, run:
pytest tests/unit/test_standard_generator.py tests/unit/test_icl_generator.py tests/integration/test_generation_diversity.py -v

All tests must pass. Report: results + flag anything about the ICL prompt format or the complex SQL prompt that you think could be improved.
```

---

## ★ Refinement Checkpoint D — Test Full Generation on 10 Questions

```
All generators are built. Let's test them on real BIRD questions with the cache enabled.

Run the pipeline through Op 7 (grounding → schema linking → all 3 generators) on the first 10 BIRD dev questions. For each question report:
1. Which of the 10-11 candidates execute successfully (success rate per generator type)
2. Do any candidates produce identical SQL? (measures real diversity)
3. What is the "oracle upper bound" — does at least 1 candidate match the ground truth? (run this as the test_candidate_pool_upper_bound_estimation test with the 10 questions)

Then:
4. Look at 2 questions where NO candidate matches ground truth. What went wrong? (Schema linking missed a column? Generators hallucinated values? Wrong join?)
5. Look at 2 questions where ALL candidates match. Are those candidates genuinely diverse or just superficially different?

Based on this analysis, propose improvements to:
- Prompts (can we guide generators better?)
- Schema linking (are we giving generators the right fields?)
- Candidate diversity strategy

I'll decide which improvements to implement before moving to the query fixer.
```

---

## Prompt 11 of 15 — Query Fixer (Op 8)

```
Read CLAUDE.md. All generators (7A, 7B, 7C) are implemented and tested. Any refinements from Checkpoint D have been applied.

Implement the query fixer — Operation 8 from Phase1_implementation_details.md.

Files to create:
1. `src/fixing/query_fixer.py` — FixedCandidate dataclass + QueryFixer class:
   - fix_candidates(candidates: list[SQLCandidate], question, evidence, schemas: LinkedSchemas, db_path, cell_matches) async -> list[FixedCandidate]
   - Step 8.1: Execute all candidates concurrently against db_path
   - Step 8.2: Categorize errors: syntax_error ("syntax error" in message), schema_error ("no such column"/"no such table"), type_error ("datatype mismatch"), empty_result (success but 0 rows)
   - Step 8.3: Fix loop with β=2 using claude-haiku-4-5-20251001. Use S₂ DDL schema for fix context (maximum context). Each fix call is targeted: include specific error type instruction in prompt. Fix calls for independent candidates run concurrently.
   - Step 8.4+8.5: Compute confidence score: base=1.0 if success+non-empty, +0.5 if result plausible (1-100 rows for non-aggregation detected via SQL keywords, 1 row for aggregation), -0.5 per fix iteration. Candidates still failing after β=2: confidence=0.0.
   - Normalize confidence scores to [0, 1] across the pool.

2. `tests/unit/test_query_fixer.py` — all 12 tests from the spec. Use a real in-memory SQLite database for execution tests. Mock Claude API for fix calls only.

After implementing, run: pytest tests/unit/test_query_fixer.py -v

All 12 tests must pass. Report: results + what fraction of typical candidate errors are syntax vs. schema vs. empty? (Estimate from the 10-question test in Checkpoint D if you have that data.)
```

---

## Prompt 12 of 15 — Adaptive Selector (Op 9)

```
Read CLAUDE.md. Query fixer (Op 8) is implemented and tested.

Implement the adaptive SQL selector — Operation 9 from Phase1_implementation_details.md.

Files to create:
1. `src/selection/adaptive_selector.py` — SelectionResult dataclass + AdaptiveSelector class:
   - select(candidates: list[FixedCandidate], question, evidence, schemas: LinkedSchemas, db_path) async -> SelectionResult
   - Step 9.1: Execute survivors (confidence>0) concurrently. Cluster by result set equivalence: sort rows, stringify as tuple, SHA256 hash as cluster key. Type normalize: float→str with consistent precision, None→"NULL".
   - Step 9.2: Fast path if cluster_count==1 → return shortest SQL, selection_method="fast_path", 0 API calls.
   - Step 9.3: Pick 1 representative per cluster (highest confidence, tiebreak: shortest SQL). Sort representatives: clusters by size desc (inter-group), then by generator performance ranking (intra-group). Ranking: reasoning_A* > complex_B2_* > icl_C* > standard_B1_*
   - Step 9.4: Pairwise round-robin tournament using claude-haiku-4-5-20251001 with select_winner tool (returns "A" or "B"). Present higher-ranked candidate as "A" in prompt (exploits positional bias). All pairs run concurrently.
   - Step 9.5: Winner = argmax(wins). Tiebreakers: cluster size, confidence score, generator ranking.
   - Step 9.6: Fallback if <2 executable candidates → return candidate with highest confidence, selection_method="fallback".

2. `tests/unit/test_adaptive_selector.py` — all 12 tests from the spec. Mock Claude API for pairwise calls. Use pre-defined ExecutionResult objects (don't need a real DB for unit tests).

After implementing, run: pytest tests/unit/test_adaptive_selector.py -v

All 12 tests must pass. Report: results + specifically: does the result equivalence clustering correctly handle: (a) float vs int results (1.0 vs 1), (b) NULL values, (c) multi-column results in different column orders?
```

---

## ★ Refinement Checkpoint E — End-to-End Test Before Integration

```
All individual components are implemented. Before wiring them into the full pipeline, let's validate them together on real data.

Run a manual end-to-end test on 5 BIRD dev questions by calling each component in sequence (not through the pipeline yet):

For each question, show me a table:
| Stage | Output Summary |
|-------|---------------|
| Grounding | N keywords, N cell matches, N few-shot examples |
| Schema linking | N fields in S₁, N fields in S₂ |
| Generation | 10-11 candidates (N per generator type) |
| Fixing | N candidates needed fixing, N succeeded, N failed |
| Selection | Method used (fast/tournament), winner generator_id |
| Correct? | Yes/No (compare to ground truth) |

Based on this:
1. What is the accuracy on these 5 questions?
2. For any wrong answers: at what stage did it go wrong?
3. Are there any component interfaces that feel awkward to wire together?
4. Is there anything you'd change about the data flow before we write the pipeline?

Discuss and then I'll approve moving to Prompt 13.
```

---

## Prompt 13 of 15 — Full Pipeline Integration

```
Read CLAUDE.md. All components (Ops 5-9) are individually implemented and tested. Interfaces have been reviewed in Checkpoint E.

Implement the complete pipeline integration — from Phase1_implementation_details.md "Pipeline Integration" section.

Files to create:
1. `src/pipeline/offline_pipeline.py` (update if already exists) — ensure OfflineArtifacts dataclass holds: profile, summary, schemas (FormattedSchemas), lsh_index (LSHIndex), faiss_index (FAISSIndex), example_store (ExampleStore). run_offline_pipeline() loads all from cache if artifacts exist on disk.

2. `src/pipeline/online_pipeline.py` — PipelineResult dataclass + answer_question(entry: BirdEntry, artifacts: OfflineArtifacts, db_path: str) async function:
   - Orchestrates Ops 5→9 exactly as in the online pipeline code block in the implementation guide
   - Uses asyncio.gather() to run all 3 generator types concurrently
   - Global asyncio.Semaphore(10) to limit concurrent API calls across all components
   - Returns PipelineResult(final_sql, selection_method, cluster_count, candidates_evaluated, generator_wins, total_cost_estimate)

3. `tests/integration/test_online_pipeline.py` — all 6 tests from the spec. Mock all Claude API calls. Use a real SQLite in-memory database for SQL execution.

4. `scripts/run_evaluation.py` — CLI with --split, --output, --resume, --workers flags. For each question: run answer_question(), compare to ground truth, save result to JSON after each question (resumable). Show live EX estimate in progress bar. Print cost summary at end.

After implementing, run: pytest tests/integration/test_online_pipeline.py -v

All 6 tests must pass. Report: results + is the pipeline's async semaphore correctly preventing rate limit errors while still enabling parallelism?
```

---

## Prompt 14 of 15 — Evaluation + Results Infrastructure

```
Read CLAUDE.md. Full pipeline is integrated and tested.

Implement the evaluation system and analysis tools.

Files to create:
1. `src/evaluation/evaluator.py`:
   - compare_results(predicted_rows, truth_rows) -> bool: sort both, normalize types (float→Decimal with fixed precision, None→"NULL_SENTINEL", bool→int), compare as sets of tuples. Column order does NOT matter.
   - compute_ex(predicted_sql, truth_sql, db_path) -> bool: execute both, compare results. If predicted_sql errors: return False. If truth_sql errors: log warning and skip.
   - EvaluationResult dataclass: question_id, db_id, difficulty, predicted_sql, truth_sql, correct, selection_method, winner_generator, cluster_count, fix_count, latency_seconds, cost_estimate

2. `src/evaluation/metrics.py`:
   - aggregate_metrics(results: list[EvaluationResult]) -> dict with: overall_ex, ex_by_difficulty, ex_by_db, ex_by_selection_method, ex_by_winner_generator, fast_path_rate, avg_fix_count, full_failure_rate, avg_latency, total_cost

3. `scripts/analyze_results.py` — reads results JSON, prints formatted summary table + highlights worst databases and most-failed question types.

4. Add a conftest.py fixture for a 10-entry results list (varied correct/incorrect, varied selection methods) for testing metrics.

5. `tests/unit/test_evaluator.py` — 8 tests: type normalization handles float/int equality, NULL equality, multi-column results, column order independence, predicted error returns False, empty result vs non-empty result, execution timeout returns False, both empty results returns True.

After implementing, run: pytest tests/unit/test_evaluator.py -v

All 8 tests must pass. Report: results + is the type normalization robust enough for BIRD's diverse data types (dates stored as text, booleans as 0/1)?
```

---

## Prompt 15 of 15 — E2E Tests + BIRD Evaluation

```
Read CLAUDE.md. All components, pipeline, and evaluation are implemented and tested.

Implement the end-to-end test suite and run the BIRD evaluation.

Files to create:
1. `tests/e2e/test_bird_mini.py` — 5 tests from the E2E Smoke Test section of the implementation guide. Mark all with @pytest.mark.live. The test_pipeline_runs_50_questions test should run the first 50 questions from BIRD mini-dev and assert: no exceptions, all return non-empty SQL, EX >= 50%, no question > 60s.

2. `tests/e2e/test_bird_full.py` — 10 tests from the E2E Full BIRD Dev Evaluation section. Mark all with @pytest.mark.live.

Then actually run the smoke test:
   pytest tests/e2e/test_bird_mini.py -v -m live --timeout=600

Report the results including:
- EX accuracy on 50 questions
- Fast path rate
- Which generator produced the most winning candidates
- Average latency and cost per question
- Any questions that caused errors or unexpected behavior

Based on these 50-question results, propose what should be tuned/fixed before running the full 1534-question BIRD dev evaluation. I'll review and approve the final run.
```

---

## ★ Final Checkpoint — Full BIRD Evaluation + Phase 1 Summary

```
The 50-question smoke test passed and we've reviewed the results. Now run the full BIRD dev evaluation:

python scripts/run_evaluation.py --split dev --output results/phase1_dev_results.json --workers 5

While it runs (expect 4-8 hours), monitor with:
python scripts/analyze_results.py results/phase1_dev_results.json

When complete, provide:
1. Overall EX accuracy and comparison to targets from CLAUDE.md
2. Breakdown by difficulty (simple/moderate/challenging)
3. Top 3 databases where we perform worst and likely reasons
4. Generator win distribution (which generator's candidates won selection most)
5. Fast path rate and average tournament size for non-fast-path questions
6. Total cost

Then answer: Based on Phase 1 results, what are the top 3 improvements that would have the highest impact on accuracy for Phase 2?
```

---

## Notes on Running These Prompts

**Before starting:**
- BIRD dataset must be downloaded before Prompt 2 (though you can start coding — the offline pipeline tests use mocks)
- Set `ANTHROPIC_API_KEY` in `.env` before any checkpoint that runs real API calls
- Enable `CACHE_LLM_RESPONSES=true` during all checkpoint tests to avoid redundant API costs

**Context management:**
- Run `/compact focus on NL2SQL Phase 1 implementation, current component interfaces, and test results` before prompts 8, 12, and 15 if the session has been long
- The `CLAUDE.md` file anchors key context — update it after each checkpoint if new decisions were made

**If a prompt produces failing tests:**
- Do not proceed to the next prompt
- Ask Claude to debug and fix within the same conversation turn
- If the issue reveals a design problem, discuss before patching

**Refinement philosophy:**
- Checkpoints A–E are the primary refinement opportunities
- During implementation prompts, Claude will flag issues — always discuss them before saying "continue"
- Better to spend an extra turn discussing a design flaw than to propagate it across 5 components
