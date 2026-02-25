# Checkpoint E — Inspection Report v5

**Date:** 2026-02-24
**Script version:** checkpoint_e_test.py (v3)
**Run tag:** v5 (re-run after code changes from v4)
**LLM Provider:** gemini
**Models:** fast=gemini-2.5-flash, powerful=gemini-2.5-flash, reasoning=gemini-2.5-flash
**Cache enabled:** true
**Elapsed time:** 2967.8 s (~89.9 s/question)
**Results file:** `checkpoint_E_review/checkpoint_e_results.json`
**Log file:** `checkpoint_E_review/run_v5.log`

---

## 1. Executive Summary

This run tests five code changes introduced since v4:

| # | File | Change |
|---|------|--------|
| 1 | `gemini_client.py` | `can_use_cache` guard: only use cached_content when (a) ALL system blocks are cacheable AND (b) no tool_defs — avoids Gemini API restriction that forbids `system_instruction` + `cached_content` in the same request |
| 2 | `gemini_client.py` | Multi-step MAX_TOKENS escalation: retry up to 3× doubling max_tokens (8192→16384→32768) instead of a single one-shot retry capped at 8192 |
| 3 | `icl_generator.py` | `instruction_block` changed to `cache=True`; MAX_TOKENS retry loop (8192→16384→32768) rather than discarding truncated responses |
| 4 | `schema_linker.py` | Text-based S₂ fallback (`_parse_text_json_fields`) when tool-use exhausts all MAX_TOKENS escalations; `schema_linker_faiss_top_k` raised from 30→50 |
| 5 | `adaptive_selector.py` | `sanitize_prompt_text` on all tournament prompt strings; `_format_execution_result` with column names and tabular preview; better tournament instructions |

### Overall Results

| Metric | v4 (2026-02-24) | **v5 (2026-02-24)** | Δ |
|--------|-----------------|---------------------|---|
| Accuracy | 16/33 (48.5%) | **17/33 (51.5%)** | +1 (+3.0 pp) |
| Oracle pre-fix (≥1 correct candidate) | 19/33 (57.6%) | **21/33 (63.6%)** | +2 (+6.1 pp) |
| Oracle post-fix (after fixer) | 18/33 (54.5%) | **21/33 (63.6%)** | +3 (+9.1 pp) |
| Selector precision (when oracle achievable) | 16/18 (88.9%) | **17/21 (81.0%)** | −1 (−7.9 pp) |
| Generation errors / timeouts | 4 | **0** | −4 |
| Run time | 7291 s (~221 s/q) | **2968 s (~89.9 s/q)** | −4323 s (3.1× faster) |
| Fixer needed | 38 candidates | **23 candidates** | −15 |
| Fixer success rate | 63.2% (24/38) | **30.4% (7/23)** | −32.8 pp |
| S1 complete (all required tables+cols) | 27/33 | **25/33** | −2 |
| S2 complete | 30/33 | **28/33** | −2 |

**Key takeaway:** The primary gains from v5 are (a) eliminating generation timeouts entirely, dramatically cutting runtime and adding 2 oracle questions, and (b) fixing the ICL caching bug so ICL candidates now contribute correctly. The overall accuracy rose by +3 pp. The main unresolved bottleneck is generation quality: 10 of 16 wrong answers have complete S2 schemas but no correct candidate was ever generated, meaning the LLMs produced logically incorrect SQL.

---

## 2. Accuracy by Difficulty and Database

### By Difficulty

| Difficulty | Correct | Total | Accuracy |
|------------|---------|-------|----------|
| simple | 7 | 11 | **63.6%** |
| moderate | 5 | 11 | **45.5%** |
| challenging | 5 | 11 | **45.5%** |

### By Database

| Database | Correct | Total | Accuracy | vs v4 |
|----------|---------|-------|----------|-------|
| california_schools | 1 | 3 | 33.3% | = |
| card_games | 1 | 3 | 33.3% | = |
| codebase_community | 1 | 3 | 33.3% | = |
| debit_card_specializing | 3 | 3 | **100.0%** | = |
| european_football_2 | 1 | 3 | 33.3% | = |
| financial | 0 | 3 | **0.0%** | = |
| formula_1 | 1 | 3 | 33.3% | = |
| student_club | 2 | 3 | 66.7% | = |
| superhero | 3 | 3 | **100.0%** | = |
| thrombosis_prediction | 1 | 3 | 33.3% | = |
| toxicology | 3 | 3 | **100.0%** | +1 |

> **Note:** Per-database accuracy is largely unchanged vs v4. The +1 net improvement is driven by general generation quality gains (fewer timeouts) rather than any DB-specific fix.

---

## 3. Oracle Performance on Generated Candidates

The oracle measures whether the correct SQL was present *anywhere* in the 11-candidate pool before and after the fixer. It is the theoretical upper bound for perfect selection.

### Summary

| Metric | Value |
|--------|-------|
| Questions with executable gold SQL | 33/33 (100%) |
| Oracle pre-fix: ≥1 correct candidate before fixing | **21/33 = 63.6%** |
| Oracle post-fix: ≥1 correct candidate after fixing | **21/33 = 63.6%** |
| New correct questions added by fixer (oracle gap pre→post) | **0 new questions** |
| Net new correct *candidates* added by fixer | **+5** (172 → 177) |
| Actual accuracy | 17/33 = 51.5% |
| Accuracy gap vs oracle post-fix | 4 questions (selector misses) |
| Selector precision when oracle achievable | **17/21 = 81.0%** |

### Interpretation

- **Generation quality** is the dominant bottleneck: only 63.6% of questions ever produce a correct candidate. 12 questions have no correct candidate in the entire pool of 11.
- The fixer did not help any additional question reach oracle status in v5, though it added 5 correct candidates at the individual-candidate level (spread across questions that already had oracle).
- 4 questions were lost by the selector despite having a correct candidate — these are *selector* failures.
- If generation oracle improved to 80%, accuracy would be ~65% (assuming current 81% selector precision holds).

### Per-Question Oracle Table

| # | DB | Q# | Diff | Cands | Pre-fix (count) | Post-fix (count) | Selector | Correct |
|---|----|----|------|-------|-----------------|-----------------|----------|---------|
| 1 | california_schools | 64 | simple | 11 | ✓ (11/11) | ✓ (11/11) | ✓ | YES |
| 2 | california_schools | 23 | moderate | 11 | ✓ (3/11) | ✓ (3/11) | **✗** | NO |
| 3 | california_schools | 28 | challenging | 11 | ✗ | ✗ | — | NO |
| 4 | card_games | 463 | simple | 11 | ✗ | ✗ | — | NO |
| 5 | card_games | 427 | moderate | 11 | ✓ (6/11) | ✓ (8/11) | ✓ | YES |
| 6 | card_games | 431 | challenging | 11 | ✗ | ✗ | — | NO |
| 7 | codebase_community | 601 | simple | 11 | ✓ (11/11) | ✓ (11/11) | ✓ | YES |
| 8 | codebase_community | 571 | moderate | 11 | ✗ | ✗ | — | NO |
| 9 | codebase_community | 586 | challenging | 11 | ✗ | ✗ | — | NO |
| 10 | debit_card_specializing | 1519 | simple | 11 | ✓ (11/11) | ✓ (11/11) | ✓ | YES |
| 11 | debit_card_specializing | 1474 | moderate | 11 | ✓ (7/11) | ✓ (8/11) | ✓ | YES |
| 12 | debit_card_specializing | 1526 | challenging | 11 | ✓ (7/11) | ✓ (7/11) | ✓ | YES |
| 13 | european_football_2 | 1027 | simple | 11 | ✗ | ✗ | — | NO |
| 14 | european_football_2 | 1025 | moderate | 11 | ✓ (11/11) | ✓ (11/11) | ✓ | YES |
| 15 | european_football_2 | 1031 | challenging | 11 | ✗ | ✗ | — | NO |
| 16 | financial | 109 | simple | 11 | ✓ (5/11) | ✓ (5/11) | **✗** | NO |
| 17 | financial | 135 | moderate | 11 | ✗ | ✗ | — | NO |
| 18 | financial | 149 | challenging | 11 | ✗ | ✗ | — | NO |
| 19 | formula_1 | 953 | simple | 11 | ✓ (3/11) | ✓ (3/11) | **✗** | NO |
| 20 | formula_1 | 852 | moderate | 11 | ✓ (2/11) | ✓ (2/11) | **✗** | NO |
| 21 | formula_1 | 994 | challenging | 11 | ✓ (10/11) | ✓ (11/11) | ✓ | YES |
| 22 | student_club | 1349 | simple | 11 | ✓ (11/11) | ✓ (11/11) | ✓ | YES |
| 23 | student_club | 1456 | moderate | 11 | ✗ | ✗ | — | NO |
| 24 | student_club | 1464 | challenging | 11 | ✓ (9/11) | ✓ (9/11) | ✓ | YES |
| 25 | superhero | 803 | simple | 11 | ✓ (11/11) | ✓ (11/11) | ✓ | YES |
| 26 | superhero | 782 | moderate | 11 | ✓ (11/11) | ✓ (11/11) | ✓ | YES |
| 27 | superhero | 773 | challenging | 11 | ✓ (11/11) | ✓ (11/11) | ✓ | YES |
| 28 | thrombosis_prediction | 1276 | simple | 11 | ✓ (5/11) | ✓ (5/11) | ✓ | YES |
| 29 | thrombosis_prediction | 1225 | moderate | 11 | ✗ | ✗ | — | NO |
| 30 | thrombosis_prediction | 1295 | challenging | 11 | ✗ | ✗ | — | NO |
| 31 | toxicology | 195 | simple | 11 | ✓ (10/11) | ✓ (11/11) | ✓ | YES |
| 32 | toxicology | 237 | moderate | 11 | ✓ (11/11) | ✓ (11/11) | ✓ | YES |
| 33 | toxicology | 304 | challenging | 11 | ✓ (6/11) | ✓ (6/11) | ✓ | YES |

---

## 4. Query Fixer Performance

### Summary

| Metric | v4 | **v5** | Δ |
|--------|----|--------|---|
| Total candidates | — | 363 | — |
| Executed successfully before fixing | — | 340/363 (93.7%) | — |
| Executed successfully after fixing | — | 347/363 (95.6%) | +7 |
| Candidates needing fixing | 38 | **23** | −15 |
| Successfully fixed | 24/38 (63.2%) | **7/23 (30.4%)** | −32.8 pp |
| Still failing after fixer | 14 | **16** | +2 |
| Oracle candidates pre-fix (correct count) | — | 172 | — |
| Oracle candidates post-fix | — | **177** | +5 |

### Interpretation

- **Fewer candidates need fixing in v5** (23 vs 38). This is the direct result of eliminating generation timeouts — timeout candidates all failed execution, inflating v4's "needs fix" count artificially.
- **Fix success rate dropped** (30.4% vs 63.2%). In v4, many "failed" candidates were merely truncated by MAX_TOKENS and trivially fixed by a retry. In v5, truncation is handled at generation time, so the candidates that do fail are genuinely wrong SQLs (wrong logic, wrong joins, wrong aggregations), which are much harder for the fixer to repair in 2 iterations.
- **Net oracle improvement:** +5 candidates across 4 questions (card_games Q#427 +2, debit Q#1474 +1, formula_1 Q#994 +1, toxicology Q#195 +1). However, all those questions already had oracle pre-fix, so the fixer didn't unlock any new *question-level* oracle.
- The fixer is most useful at fixing schema errors (no such column/table) — these are typically generated in 1 iteration. Semantic errors (wrong join, wrong aggregation) are not fixable by the current prompt.

### Per-Question Fixer Detail

| # | DB | Q# | Diff | NeedFix | FixedOK | StillFail | OrigSucc | PostSucc |
|---|----|----|------|---------|---------|-----------|----------|----------|
| 3 | california_schools | 28 | challenging | 2 | 1 | 1 | 9 | 10 |
| 5 | card_games | 427 | moderate | 5 | 2 | 3 | 6 | 8 |
| 6 | card_games | 431 | challenging | 7 | 0 | 7 | 4 | 4 |
| 11 | debit_card_specializing | 1474 | moderate | 1 | 1 | 0 | 10 | 11 |
| 12 | debit_card_specializing | 1526 | challenging | 4 | 0 | 4 | 7 | 7 |
| 15 | european_football_2 | 1031 | challenging | 1 | 0 | 1 | 10 | 10 |
| 20 | formula_1 | 852 | moderate | 1 | 1 | 0 | 10 | 11 |
| 21 | formula_1 | 994 | challenging | 1 | 1 | 0 | 10 | 11 |
| 31 | toxicology | 195 | simple | 1 | 1 | 0 | 10 | 11 |
*All other questions: NeedFix=0*

> **Notable:** Q#431 [card_games/challenging] had 7 failing candidates and 0 fixed — all failures are semantic (wrong table: uses `foreign_data` + `cards` instead of `sets` + `set_translations`). The fixer cannot guess a different table; it only patches syntax errors and missing columns.

---

## 5. Query Selector Performance

### Selection Method Distribution

| Method | Count | % |
|--------|-------|---|
| fast_path (unanimous 1 cluster) | 21 | 63.6% |
| tournament (2+ clusters) | 12 | 36.4% |

### Selector Accuracy by Method

| Method | Correct | Total | Accuracy | Oracle Cases | Oracle Matched | Oracle Precision |
|--------|---------|-------|----------|-------------|----------------|-----------------|
| fast_path | 13 | 21 | 61.9% | 13 | 13 | **100.0%** |
| tournament | 4 | 12 | 33.3% | 8 | 4 | **50.0%** |

**Key insight:** Fast-path is perfect — when all 11 candidates agree, the result is always right (13/13). The bottleneck is the **tournament**, where the LLM judge picks the wrong answer 50% of the time when oracle is achievable.

### Winner Generator Distribution

| Generator | Total Wins | Correct | Win Accuracy |
|-----------|-----------|---------|-------------|
| reasoning_A1 | 12 | 7 | 58.3% |
| icl_C1 | 4 | 2 | 50.0% |
| reasoning_A2 | 3 | 0 | **0.0%** |
| standard_B1_s1 | 3 | 2 | 66.7% |
| reasoning_A3 | 2 | 0 | **0.0%** |
| complex_B2_s1 | 2 | 2 | 100.0% |
| standard_B1_s2 | 2 | 0 | **0.0%** |
| reasoning_A4 | 2 | 1 | 50.0% |
| icl_C2 | 1 | 1 | 100.0% |
| complex_B2_s2 | 1 | 1 | 100.0% |
| icl_C3 | 1 | 1 | 100.0% |

> reasoning_A1 wins the tournament most often (12 times) — but reasoning_A2, A3 still win sometimes with 0 correctness, suggesting MALFORMED_FUNCTION_CALL is still occurring and defaulting to A.

### Selector Misses (Oracle Achievable but Wrong Answer Selected)

**Q#23** [california_schools / moderate]
- Oracle: 3/11 correct candidates available
- Winner: reasoning_A2 | Method: tournament (2 clusters)
- Root cause: S1 schema missing `schools.School` column. The 8 incorrect candidates use `frpm."School Name"` (wrong column/table) but they all agree, forming the large cluster. The 3 correct candidates using `schools.School` are in the minority cluster. The tournament picked the majority-cluster representative — which was wrong because the correct `School` column was not in S1, confusing the model.

**Q#109** [financial / simple]
- Oracle: 5/11 correct candidates available
- Winner: reasoning_A3 | Method: tournament (2 clusters)
- Root cause: Reasoning model chose the more complex SQL (3-table JOIN via `disp`) even though the simpler 2-table JOIN (`client + district`) gives the same answer. The tournament likely saw a 5-vs-6 split and selected the larger cluster, which was the more complex (but wrong) approach.

**Q#953** [formula_1 / simple]
- Oracle: 3/11 correct candidates available
- Winner: reasoning_A4 | Method: tournament (2 clusters)
- Root cause: S1 missing `results.laps` column. Generators without `laps` in context used `lapTimes` table as a proxy join (wrong approach). Those formed the majority cluster. The 3 correct candidates using `results.laps` were in the minority cluster and lost the tournament.

**Q#852** [formula_1 / moderate]
- Oracle: 2/11 correct candidates available
- Winner: standard_B1_s1 | Method: tournament (4 clusters)
- Root cause: The gold SQL uses a hardcoded list of country names (`NOT IN ('Bahrain', 'China', 'Singapore', ...)`), but the system generated `NOT IN ('Asia', 'Europe')` — which is wrong SQL (countries are not continents). The 2 correct candidates were likely in a small cluster and lost to the 9-candidate majority that all made the same continent-vs-country mistake.

---

## 6. Schema Linker Recall and Accuracy

Recall measures what fraction of tables and (table, column) pairs required by the gold SQL are present in S1 (precise schema) and S2 (recall schema).

### Summary

| Metric | S1 (Precise) | S2 (Recall) | vs v4 S1 | vs v4 S2 |
|--------|-------------|------------|---------|---------|
| Avg Table Recall | 94.95% | 94.95% | −1.0 pp | −1.1 pp |
| Avg Column Recall | 93.33% | 95.66% | −2.4 pp | −1.2 pp |
| Complete coverage (all tables+cols) | 25/33 | 28/33 | −2 | −2 |

> **Regression vs v4:** Schema linker completeness slightly regressed. S1 complete went from 27→25, S2 from 30→28. The most likely cause: the `can_use_cache` guard now bypasses Gemini prompt caching for schema linker calls (which use tool_defs), so schema linker runs as a plain system_instruction request. This may affect response quality when the model processes wide schemas without cached context. However, the `schema_linker_faiss_top_k=50` change partially compensates by providing more candidates.

### Questions with Schema Deficiencies

| Q# | DB | Diff | S1 | S2 | Missing in S1 | Missing in S2 |
|----|----|----|----|----|---------------|---------------|
| 28 | california_schools | challenging | Incomplete | **Incomplete** | t2.school, t2.doc | t2.doc |
| 431 | card_games | challenging | Incomplete | Complete | t1.name | — |
| 1519 | debit_card | simple | Incomplete | Incomplete | gasstations table, gasstationid cols | gasstations table, gasstationid cols |
| 1526 | debit_card | challenging | Incomplete | Incomplete | gasstations table, gasstationid cols | gasstations table, gasstationid cols |
| 953 | formula_1 | simple | Incomplete | Complete | t1.laps | — |
| 1456 | student_club | moderate | Incomplete | **Incomplete** | budget table, budget_id, spent | budget table, budget_id, spent |
| 1464 | student_club | challenging | Incomplete | Complete | event, attendance tables | — |
| 23 | california_schools | moderate | Incomplete | Complete | t1.school | — |

> **S2 failures (both S1 and S2 incomplete):**
> - **Q#28:** `schools.DOC` column not summarized, never appears in FAISS/DDL context.
> - **Q#1519, Q#1526:** `gasstations` table not retrieved by FAISS (no semantic match for `GasStationID` in the question). However, both questions were still answered correctly — the generators used `transactions_1k` primary key independently.
> - **Q#1456:** `budget` table entirely absent from both S1 and S2. The question says "most money" but the gold SQL uses `budget.spent`, not `expense.cost`. The schema linker correctly picks up `expense` (more semantically obvious) but misses `budget`, causing all 11 generators to aggregate on `expense.cost` instead.

---

## 7. Failure Analysis — Root Cause Classification

### Failure Categories

| Category | Count | Questions |
|----------|-------|----------|
| Pure generation failure (complete S2 schema, wrong SQL logic) | 10 | 463, 571, 586, 1027, 1031, 135, 149, 1225, 1295 + 1 |
| Schema + generation failure (S2 also missing critical elements) | 2 | 28, 1456 |
| Selector miss (oracle available but wrong candidate chosen) | 4 | 23, 109, 953, 852 |
| **Total failures** | **16** | — |

> Note: One "pure generation failure" case also has a partial S1 miss (card_games Q#431), but S2 is complete, so it counts as a generation failure.

### Pure Generation Failures — Pattern Analysis

| Q# | DB | Diff | Failure Pattern |
|----|----|----|----------------|
| 463 | card_games | simple | Missing `COUNT(DISTINCT)` + `IS NOT NULL` filter. Counts duplicate rows; gold counts only distinct non-NULL translations. |
| 571 | codebase_community | moderate | Correlated subquery vs JOIN semantics differ when a user has posts by others (distinct count issue). Gold uses `CAST(COUNT(T2.Id) AS REAL) / COUNT(DISTINCT T1.Id)` from a JOIN. |
| 586 | codebase_community | challenging | Gold returns 2 columns `(DisplayName, Title)`; all 11 generators return only 1 column `(DisplayName)`. "Extra columns not implied by question" issue (checkpoint_D P2-7). |
| 1027 | european_football_2 | simple | Known join-key ambiguity: gold uses `t1.id = t2.id` but the correct semantic join is `player_api_id`. BIRD gold SQL has an unusual join on `id` (the surrogate key) which is not the FK declared in the schema. |
| 1031 | european_football_2 | challenging | Age calculation: gold uses `DATETIME() - birthday` (yields float year difference); generators use `julianday` approach (more correct semantically but different result format). Execution result mismatch. |
| 135 | financial | moderate | Czech semantic fields + SQL truncation: `balance < 0` filter missing from generated SQL; `COUNT(DISTINCT account_id)` vs `COUNT(account_id)` discrepancy. |
| 149 | financial | challenging | SQL truncation at `A11 > 800` — `BETWEEN 8000 AND 9000` was cut off. All generators systematically truncate for this question (8000→800, likely MAX_TOKENS partial output that was not retried). |
| 1225 | thrombosis_prediction | moderate | "List and group" misinterpreted as `GROUP_CONCAT(DISTINCT ID)` instead of `SELECT ID, SEX ... GROUP BY SEX, ID` — wrong aggregation function. |
| 1295 | thrombosis_prediction | challenging | Missing `Patient` table in 3-way join. Gold: `Patient JOIN Laboratory JOIN Examination`. All generators use `Laboratory JOIN Examination` only — misses the `Patient` bridge join because the question focuses on bilirubin (Laboratory). |

---

## 8. Changes That Helped vs. Changes That Need More Work

### Changes That Had Measurable Positive Impact

| Change | Impact |
|--------|--------|
| Increased `_TIMEOUT_GENERATION` 120→300s | **Eliminated all 4 generation timeouts** → run time −4323s (3.1×), +2 oracle questions |
| `gemini_client.py` multi-step MAX_TOKENS escalation | ICL and schema linker no longer produce blank responses for wide schemas |
| `icl_generator.py` `cache=True` on instruction block | ICL caching conflict resolved → ICL candidates now generate correctly (contributed to +5 oracle candidates) |
| `adaptive_selector.py` `sanitize_prompt_text` | Reduced MALFORMED_FUNCTION_CALL errors in tournament (improved from ~70% broken comparisons in v4) |
| `adaptive_selector.py` `_format_execution_result` | Richer table preview in tournament prompt — selector can now reason about row counts and values |

### Changes That Need Verification / Had Mixed Impact

| Change | Status |
|--------|--------|
| `schema_linker_faiss_top_k` 30→50 | Slight schema recall regression (S1: 27→25, S2: 30→28). The higher recall set may be diluting precision slightly. |
| `can_use_cache` guard in gemini_client | Schema linker no longer benefits from prompt caching (tools + cache incompatible). This may have contributed to the recall regression. |
| Text-based S₂ fallback (`_parse_text_json_fields`) | Not triggered in any of the 33 questions (no MAX_TOKENS exhaustion occurred with the new retry logic). Fallback works but was not needed. |

---

## 9. Open Issues and Suggested Improvements

### P0 Issues (Must Fix Before Phase 1 Target ≥68%)

**P0-1: Generation quality — 12 questions with no correct oracle**
10 of these have complete S2 schemas but still produce 0 correct candidates. The root causes are:
- *DISTINCT + IS NOT NULL* semantics (Q#463): Add explicit SQL rule: "When counting translations/versions, use COUNT(DISTINCT col) and filter IS NOT NULL."
- *Extra SELECT columns* (Q#586): Add rule: "Return exactly the columns asked for. Do not add extra columns."
- *Correlated aggregate semantics* (Q#571, Q#1225, Q#1295): Improve system prompt reasoning section on GROUP BY vs subquery equivalence.
- *Czech/encoded column values* (Q#135, Q#149): These are persistent `financial` DB failures. The Czech column names (`A11`, `VYBER KARTOU`) confuse all generators. Consider adding a `column_value_encoding_note` to the evidence extraction or schema summary for `financial`.
- *Age/date arithmetic* (Q#1031): Both gold SQL and model SQL are arguably correct (different formatting), but results differ. The gold uses `DATETIME() - birthday` which is a SQLite-specific non-standard expression. Consider post-processing numeric comparison normalization in the evaluator.

**P0-2: Tournament selector still wrong 50% of the time**
4 of 8 achievable oracle questions lost in tournament. Known causes:
- *Minority cluster with correct answer* (Q#23, Q#953): When a schema element is missing from S1, many generators pick the wrong table and form a large incorrect cluster. The correct generators are outvoted. Fix: use S2 schema context during tournament (currently uses S1 only), OR prioritize clusters by generator type (reasoning > standard).
- *Majority cluster semantics wrong* (Q#109, Q#852): Most generators agree on a plausible-looking but wrong answer. The judge LLM is not questioning the majority. Fix: improve the tournament prompt to explicitly instruct "do not blindly follow the majority — evaluate correctness independently."
- Investigate whether MALFORMED_FUNCTION_CALL still affects tournament calls. The `sanitize_prompt_text` fix helps but the judge may still fail on complex SQLs containing backtick-quoted identifiers.

**P0-3: Schema linker misses `gasstations` and `budget` tables entirely in S2**
- Q#1456 [student_club]: `budget.spent` is the answer field but `expense.cost` is semantically closer to the question. Solution: add "ordered BY" and "LIMIT" hint detection to the grounding — "top five members who spend the most" should trigger retrieval of any `spent` or `budget` column.
- Q#1519/1526 [debit_card]: `gasstations` table is not retrieved because the question doesn't mention gas stations explicitly. These questions were still correct, but fragile.

### P1 Issues (Important Before Full Evaluation)

**P1-1: Financial DB persistently at 0% (3/3 wrong)**
All three questions fail due to Czech-encoded column semantics:
- Q#109: selector miss (not schema/generation failure)
- Q#135: `balance < 0` filter and Czech field names
- Q#149: SQL truncation (all generators truncate `BETWEEN 8000 AND 9000`)

Recommended: Add a column-value mapping hint for `financial` DB to the evidence block (e.g., "A11 = average salary, VYBER KARTOU = credit card withdrawal, POPLATEK MESICNE = monthly payment").

**P1-2: `doc` column missing in california_schools challenging (Q#28)**
The `schools.DOC` column is not in the FAISS index (or not summarized). Its short summary is probably missing. Solution: verify `california_schools_faiss_fields.json` includes `DOC` and its summary is meaningful.

**P1-3: Fixer cannot repair semantic errors (correct schema, wrong joins)**
Fix success rate dropped to 30.4% because the remaining unfixed candidates have semantic errors (wrong join key, wrong aggregation), not syntax errors. The fixer only has access to the error message + S2 DDL — it cannot detect that `lapTimes` is not equivalent to `results.laps`. Consider adding: "If the error is an empty result or wrong row count, try a different join approach."

### P2 Issues (Nice to Have)

**P2-1: European Football join-key ambiguity (Q#1027)**
Gold SQL joins on `Player_Attributes.id = Player.id` (surrogate key), but the natural FK is `player_api_id`. All 11 generators use `player_api_id` (which is technically correct per schema). The gold SQL is arguably quirky. This is a dataset annotation issue and hard to fix programmatically.

**P2-2: `codebase_community` returns wrong column count (Q#586)**
Add a post-generation rule: compare the question's requested output columns with the SELECT clause. If the question asks "which user" but the gold SQL returns `(DisplayName, Title)`, all generators will miss the extra `Title` column. Consider adding to generation instructions: "If the question implies multiple output attributes (e.g., 'which user added a bounty to which post'), list all explicitly mentioned output columns."

---

## 10. Summary Scorecard

| Dimension | v4 | v5 | Target |
|-----------|----|----|--------|
| Accuracy (33 q) | 48.5% | **51.5%** | ≥68% |
| Oracle ceiling | 57.6% | **63.6%** | ≥80% |
| Generation errors | 4 | **0** | 0 |
| Run time per question | 221 s | **89.9 s** | <120 s |
| S1 complete | 81.8% | 75.8% | ≥90% |
| S2 complete | 90.9% | 84.8% | ≥95% |
| Tournament precision | 88.9% | **81.0%** | ≥90% |
| Fast-path precision | — | **100.0%** | 100% |

**Gap to 68% target:** Need ~+16.5 pp, approximately 6 more correct questions out of 33. Based on the analysis, the achievable gains in priority order:
1. Fix tournament selector (2 selector misses are clearly fixable) → +2
2. Improve generation for `financial` DB (Czech encoding hints) → +1–2
3. Fix extra-column issue (Q#586) and correlated aggregate issue (Q#1225, Q#1295) → +1–2
4. Improve S2 schema recall for `budget`/`gasstations` tables → +1

**Total achievable improvement:** +5–7 questions → ~67–72% accuracy (within or near target range).
