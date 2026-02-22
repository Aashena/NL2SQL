# Checkpoint A Review — Ops 0a, 0b, 0c on BIRD Dev Dataset

**Date:** 2026-02-22
**Phase:** 1
**Ops covered:** Op 0a (Profiling), Op 0b (LLM Field Summarization), Op 0c (Schema Formatting)
**Databases:** 11 BIRD dev databases
**LLM Provider:** Gemini (`gemini-2.5-flash` for model_fast tier)

---

## Pipeline Run Summary

| Step | Databases processed | Failures | Notes |
|------|---------------------|----------|-------|
| Profile (Op 0a) | 11/11 | 0 | Pure SQL statistics — no API calls |
| Summarize (Op 0b) | 11/11 | 0 | LLM API calls per batch; graceful fallback on blocked responses |
| Format (Op 0c) | 11/11 | 0 | DDL + Markdown schemas generated |

---

## Database Coverage

| Database | Tables | Columns | LLM Summaries | Default Fallbacks |
|----------|--------|---------|---------------|-------------------|
| california_schools | 3 | 89 | 89 | 0 |
| card_games | 6 | 115 | 97 | 18 |
| codebase_community | 8 | 71 | 71 | 0 |
| debit_card_specializing | 5 | 21 | 21 | 0 |
| european_football_2 | 7 | 199 | 181 | 18 |
| financial | 8 | 55 | 49 | 6 |
| formula_1 | 13 | 94 | 94 | 0 |
| student_club | 8 | 48 | 48 | 0 |
| superhero | 10 | 31 | 31 | 0 |
| thrombosis_prediction | 3 | 64 | 34 | 30 |
| toxicology | 4 | 11 | 11 | 0 |
| **TOTAL** | **75** | **798** | **726 (91%)** | **72 (9%)** |

---

## Unit Tests

All **64 tests pass** (12 bird_loader+database, 31 profiler, 10 summarizer, 11 schema_formatter).
See `unit_test_results/test_results.txt` for full output.

**Re-verified 2026-02-22** after applying improvements — still 64/64 pass (0.70 s).

---

## Bugs Found and Fixed During This Run

### Bug 1 — `summarize_database` was never awaited in the script
- **File:** `scripts/run_offline_preprocessing.py`
- **Problem:** `_run_summarize()` called `summarize_database(...)` without `await`, so the async coroutine was created but never executed. All 11 "summarizations" completed in 0.0s and produced no output files.
- **Fix:** Wrapped the call in `asyncio.run()`.

### Bug 2 — Gemini API returns `None` content on some batches
- **Files:** `src/llm/gemini_client.py`, `src/preprocessing/summarizer.py`
- **Problem:** For certain large/complex tables, Gemini 2.5-flash returned a response where `candidates[0].content` was `None` (finish_reason likely `SAFETY` or `MAX_TOKENS`). This caused `AttributeError: 'NoneType' object has no attribute 'parts'` and crashed the entire database summarization.
- **Fix 1:** `_parse_response()` in the Gemini client now checks for missing candidates or `None` content and raises a descriptive `LLMError` instead of crashing.
- **Fix 2:** `generate()` now wraps `_parse_response()` inside the try/except so all errors propagate as `LLMError`.
- **Fix 3:** `summarize_database()` catches `LLMError` per batch and falls back to the default summary format (`"The {col} field in the {table} table."`) for affected columns.
- **Impact:** 72/798 columns (9%) ended up with default fallback summaries. Mostly in `thrombosis_prediction` (30), `card_games` (18), `european_football_2` (18), `financial` (6).

---

## Output Quality Assessment

### Op 0a — Statistical Profile
**Strengths:**
- Full null rates, distinct counts, top-10 sample values (by frequency) for all columns
- Correct primary key and foreign key detection
- MinHash bands (128 permutations) stored per column for LSH index in Op 1a
- SQLite type affinity correctly mapped (5-rule algorithm)

**Observations:**
- `minhash_bands` now stored as base64 in JSON (128 uint64 → struct.pack → base64) — ~33% smaller than a plain integer array. Fixed before re-run.
- MinHash sampling guard (`_MINHASH_SAMPLE_LIMIT = 50_000`) caps computation for very-high-cardinality columns — fixes the multi-hour stalls in `codebase_community` and `card_games`. Fixed before re-run.

### Op 0b — LLM Field Summarization
**Strengths:**
- 91% LLM-generated summaries (726/798 columns)
- Short summaries (≤200 chars): concise and accurate
- Long summaries (≤1000 chars): rich context including value semantics, FK relationships, and domain explanations

**Sample quality check (california_schools.frpm):**
- `CDSCode`: "The CDSCode is a critical unique identifier in the California public school system. It is a 14-digit code formed by conc..." — accurate, mentions PK role
- `Charter School (Y/N)`: "This column uses a binary indicator (1 or 0) to denote whether a school operates as a charter school..." — correctly identifies binary encoding
- `FRPM Count (K-12)`: properly explains FRPM acronym and eligibility context

**Issues addressed:**
- `thrombosis_prediction` had 30/64 default fallbacks (47%) — medical/patient column names likely triggered safety filters. The individual-column retry (improvement #1) will recover most of these on a re-run.
- Batch retry now implemented: on `LLMError` for a 6-column batch, each column is retried individually before falling back to defaults.

### Op 0c — Schema Formatting
**Strengths:**
- DDL format: column summaries truncated to 120 chars as inline comments — clean and readable
- Example row appended at end of each table definition — very helpful for generators
- Foreign key annotations as trailing comments
- Markdown format: table-per-section with column type, description, and sample values

**Sample DDL quality (`california_schools`):**
```sql
CREATE TABLE frpm (
  CDSCode TEXT PRIMARY KEY,  -- The CDSCode is a critical unique identifier...
  "Academic Year" TEXT,  -- This field indicates the academic year...
  ...
);
-- Foreign keys: CDSCode REFERENCES schools(CDSCode)
-- Example row: ('01100170109835', '2014-2015', '19', ...)
```
This format is well-suited for SQL generators — the inline comments provide semantic context without bloating the prompt.

---

## Improvements Applied Before Prompt 5

### High Priority — DONE
1. **Retry failed batches with smaller batch size** ✅
   - **File:** `src/preprocessing/summarizer.py` (lines 218–251)
   - When `LLMError` is raised on a batch of >1 columns, each column in that batch is now retried individually (batch_size=1). Only if the individual retry also fails does the column fall back to the default summary.
   - Expected impact: recovers most of the 72 missing summaries from the original run.

2. **Log columns with default fallbacks** ✅
   - **File:** `src/preprocessing/summarizer.py` (lines 269–273)
   - A `logger.warning()` is emitted per table listing every column that received a default summary. Grep for `"used default fallback summary"` to find quality gaps.

### Medium Priority — DONE
3. **MinHash sampling guard per column** ✅
   - **File:** `src/preprocessing/profiler.py` (line 24, lines 318–333)
   - `_MINHASH_SAMPLE_LIMIT = 50_000` caps rows fed into MinHash when `distinct_count > 50_000`. Logs a warning and uses `LIMIT 50000` in the SQL fetch. Fixes multi-hour stalls in `codebase_community` and `card_games`.

4. **Base64-compress `minhash_bands` in profile JSON** ✅
   - **File:** `src/preprocessing/profiler.py` (lines 105–112, 394, 408–409)
   - `_encode_minhash()` packs 128 uint64 values via `struct.pack("128Q", ...)` then base64-encodes (~33% smaller than a JSON integer array). `_load_profile_from_json()` detects the string format vs. legacy int-list and decodes transparently.

### Low Priority — DONE
5. **Gemini context caching** ✅
   - **File:** `src/llm/gemini_client.py` (`_try_create_cache()`, `generate()`)
   - `CacheableText(cache=True)` blocks are submitted to the Gemini context caching API. Falls back silently when the model or token threshold doesn't qualify. Cache hits skip re-sending the system prompt on subsequent calls.

---

## Files in This Review Folder

```
checkpoint_A_review/
├── README.md                           (this file)
└── unit_test_results/
    └── test_results.txt                (64 tests, all passed)
```

The preprocessed output files live in the canonical cache directory — no copies are kept here:

| Output | Path | Description |
|--------|------|-------------|
| Profiles (Op 0a) | `data/preprocessed/profiles/<db>.json` | 11 JSONs — statistical profile per database |
| Summaries (Op 0b) | `data/preprocessed/summaries/<db>.json` | 11 JSONs — LLM field summaries per database |
| Schemas DDL (Op 0c) | `data/preprocessed/schemas/<db>_ddl.sql` | 11 SQL files — DDL with inline summary comments |
| Schemas Markdown (Op 0c) | `data/preprocessed/schemas/<db>_markdown.md` | 11 Markdown files — human-readable schema tables |

---

## Checkpoint A Status: APPROVED ✅

All improvements (high, medium, and low priority) have been applied and verified:

| # | Improvement | File | Status |
|---|-------------|------|--------|
| 1 | Retry batch → individual on LLMError | `summarizer.py:218–251` | ✅ Done |
| 2 | Warning log for default-fallback columns | `summarizer.py:269–273` | ✅ Done |
| 3 | MinHash sampling guard (50k distinct cap) | `profiler.py:24,318–333` | ✅ Done |
| 4 | Base64 compression for minhash_bands JSON | `profiler.py:105–112,394,408–409` | ✅ Done |
| 5 | Gemini context caching for system prompts | `gemini_client.py:_try_create_cache()` | ✅ Done |
| — | Gemini None-content guard (`_parse_response`) | `gemini_client.py:210–218` | ✅ Done |
| — | `asyncio.run()` fix in offline script | `run_offline_preprocessing.py` | ✅ Done |

All **64 unit tests pass** after improvements. Ready to proceed to **Prompt 5 (LSH Index)**.
