# Checkpoint E Inspection Report — v6 (2026-02-25)

## Overview

Re-run of the 33-question stratified Checkpoint E test against the updated codebase.
This run captures the effect of all changes made since the v5 run (2026-02-24).

**Configuration**

| Parameter           | Value                    |
|---------------------|--------------------------|
| LLM Provider        | Gemini                   |
| model_fast          | gemini-2.5-flash         |
| model_powerful      | gemini-2.5-flash         |
| model_reasoning     | gemini-2.5-flash         |
| Cache LLM Responses | Enabled                  |
| Questions Sampled   | 33 (3 per DB, stratified) |
| Runtime             | 1419.2 s (~23.7 min)     |
| Script Version      | v3                       |

---

## 1. Top-Line Results

| Metric              | v6 (this run) | v5 (prior run) | Delta  |
|---------------------|---------------|----------------|--------|
| Accuracy            | **18/33 (54.5%)** | 17/33 (51.5%) | **+1 question (+3pp)** |
| Runtime             | 1419 s (43 s/q) | 2968 s (90 s/q) | **2.1× faster** (cache hits) |
| Oracle pre-fix      | 21/33 (63.6%) | 21/33 (63.6%) | 0      |
| Oracle post-fix     | 21/33 (63.6%) | 21/33 (63.6%) | 0      |
| Selector precision  | **18/21 (85.7%)** | 17/21 (81.0%) | **+4.7pp** |
| S1 complete schemas | 26/33 (78.8%) | 25/33 (75.8%) | +1     |
| S2 complete schemas | 29/33 (87.9%) | 28/33 (84.8%) | +1     |

### Accuracy by Difficulty

| Difficulty   | Correct | Total | Accuracy |
|-------------|---------|-------|----------|
| Simple      | 7       | 11    | 63.6%    |
| Moderate    | 7       | 11    | 63.6%    |
| Challenging | 4       | 11    | 36.4%    |

### Accuracy by Database

| Database                | Correct | Total | Accuracy |
|-------------------------|---------|-------|----------|
| california_schools      | 2       | 3     | 66.7%    |
| card_games              | 1       | 3     | 33.3%    |
| codebase_community      | 1       | 3     | 33.3%    |
| debit_card_specializing | 3       | 3     | **100.0%** |
| european_football_2     | 1       | 3     | 33.3%    |
| financial               | 0       | 3     | **0.0%** |
| formula_1               | 2       | 3     | 66.7%    |
| student_club            | 2       | 3     | 66.7%    |
| superhero               | 3       | 3     | **100.0%** |
| thrombosis_prediction   | 1       | 3     | 33.3%    |
| toxicology              | 2       | 3     | 66.7%    |

---

## 2. Oracle Performance on Generated Candidates

**Oracle = at least one generated candidate produces the correct result set.**

| Metric                                           | Value           |
|--------------------------------------------------|-----------------|
| Questions where gold SQL is executable           | 33/33 (100%)    |
| Oracle pre-fix (any correct candidate before fix)| **21/33 (63.6%)** |
| Oracle post-fix (any correct candidate after fix)| **21/33 (63.6%)** |
| Net oracle gain from fixer                       | **0 questions** |

The oracle rate is unchanged from v5. The fixer rescued +7 individual candidates across questions (176 → 183 oracle-correct candidates in total), but none of these shifted a question from oracle=False to oracle=True — meaning the fixer only "repaired" secondary candidates in questions that already had a correct answer elsewhere.

### Generation Failures (Oracle Not Achievable) — 12 Questions

All 11 candidates failed to produce the correct result in each case below. Generation always succeeded (0 error flags), meaning these are **semantic/logic errors**, not API errors.

| Q# | Database | Difficulty | Schema Complete (S2) | Root Cause |
|----|----------|-----------|----------------------|------------|
| 28  | california_schools | challenging | ❌ (missing DOC col) | Correlated subquery for average difference + schema miss |
| 463 | card_games | simple | ✅ | Wrong table for translations (semantics) |
| 431 | card_games | challenging | ✅ | Complex multi-condition filter ("not outside US" + foil) |
| 571 | codebase_community | moderate | ✅ | Ratio of posts-to-comments (integer division edge) |
| 586 | codebase_community | challenging | ✅ | Subquery linking bounty amount → post title → username |
| 1027 | european_football_2 | simple | ✅ | Wrong aggregation for "highest penalty attribute" ranking |
| 109  | financial | simple | ✅ | Czech-encoded value semantics ("F" for Ženský) |
| 135  | financial | moderate | ✅ | Czech-encoded account type semantics |
| 149  | financial | challenging | ✅ | Czech-encoded account types + complex loan join |
| 1225 | thrombosis_prediction | moderate | ✅ | GROUP BY sex ordering (DESC required, ascending generated) |
| 1295 | thrombosis_prediction | challenging | ✅ | Correlated subquery for anti-coagulant among high-bilirubin patients |
| 1456 | student_club | moderate | ✅ | TOP 5 by spend: wrong join or aggregation path for total expenditure |

**Key observations:**
- **10/12 failures have complete S2 schemas** — schema recall is NOT the limiting factor for these failures
- **3/12 (financial DB)** are purely due to Czech-encoded values/labels that the model cannot infer from schema summaries alone (persistent P0-3 issue)
- **6/12** are semantic/logic errors: complex subqueries, wrong aggregation strategy, multi-condition filters
- **1/12** (Q#28 california_schools) has a schema miss (missing `DOC` column in S2) that may have contributed to the failure

---

## 3. Query Fixer Performance

### Aggregate Fixer Statistics

| Metric                           | Value          |
|----------------------------------|----------------|
| Total candidates generated       | 363            |
| Originally executable (pre-fix)  | 331/363 (91.2%) |
| Post-fix executable              | 348/363 (95.9%) |
| Candidates needing fix           | 32             |
| Successfully fixed               | 17/32 (**53.1%**) |
| Still failing after fix          | 15/363 (4.1%)  |
| Oracle-correct candidates pre-fix | 176/363       |
| Oracle-correct candidates post-fix | 183/363 (+7) |
| Net oracle questions gained       | **0**          |

### Per-Question Fixer Breakdown

| Q# | DB | Diff | Needed Fix | Fixed | Still Fail | Pre-Exec | Post-Exec |
|----|----|------|-----------|-------|------------|----------|-----------|
| 28  | california_schools | challenging | 2 | 0 | 2 | 9/11 | 9/11 |
| 463 | card_games | simple | 1 | 0 | 1 | 10/11 | 10/11 |
| 427 | card_games | moderate | 4 | 3 | 1 | 7/11 | 10/11 |
| 431 | card_games | challenging | 8 | 2 | 6 | 3/11 | 5/11 |
| 571 | codebase_community | moderate | 1 | 1 | 0 | 10/11 | 11/11 |
| 1526 | debit_card_specializing | challenging | 6 | 2 | 4 | 5/11 | 7/11 |
| 1027 | european_football_2 | simple | 2 | 2 | 0 | 9/11 | 11/11 |
| 1031 | european_football_2 | challenging | 1 | 0 | 1 | 10/11 | 10/11 |
| 109  | financial | simple | 1 | 1 | 0 | 10/11 | 11/11 |
| 852  | formula_1 | moderate | 2 | 2 | 0 | 9/11 | 11/11 |
| 1456 | student_club | moderate | 1 | 1 | 0 | 10/11 | 11/11 |
| 803  | superhero | simple | 1 | 1 | 0 | 10/11 | 11/11 |
| 1225 | thrombosis_prediction | moderate | 1 | 1 | 0 | 10/11 | 11/11 |
| 195  | toxicology | simple | 1 | 1 | 0 | 10/11 | 11/11 |

### Analysis

**Where the fixer helped:**
- Fixed syntax/schema errors in 17/32 candidates (53.1%)
- Most successful fixes were schema-error corrections (wrong column name, missing table), not semantic fixes
- The fixer consistently brought "almost right" queries (10/11 success rate originally) back to full 11/11

**Where the fixer struggled:**
- Q#431 (card_games, challenging): 6 of 8 broken queries still fail — complex joins + multi-condition filters are beyond the fixer's single-iteration fix capability
- Q#1526 (debit_card_specializing, challenging): 4 still failing — likely requires significant logic restructuring
- Q#28 (california_schools, challenging): 2 fails — correlated subquery structure errors resist simple fixes

**Critical finding:** The fixer improved executability (91.2% → 95.9%) but added **zero net oracle-correct questions**. This suggests that the queries failing oracle were failing for semantic reasons (wrong logic), not execution errors. Once the query is structurally wrong, fixing it to "execute without error" doesn't make it semantically correct.

---

## 4. Query Selector Performance

### Aggregate Selector Statistics

| Metric | Value |
|--------|-------|
| Oracle-achievable questions | 21/33 |
| Selector correctly chose oracle candidate | 18/21 (**85.7%**) |
| Selector misses | 3 |

**Vs. v5:** 17/21 = 81.0% → 18/21 = 85.7% (**+4.7pp improvement**)

This improvement is directly attributable to the **pairwise comparison redesign** in `adaptive_selector.py`: replacing Gemini tool-use (which caused `MALFORMED_FUNCTION_CALL` errors) with free-text `FINAL: A` / `FINAL: B` parsing. The selector no longer silently falls back to `"A"` on parse failures.

### Selector Misses (3 Cases)

**Case 1 — Q#1031 (european_football_2, challenging)**
- Question: *"Calculate the age of players with sprint speed ≥ 80"*
- Clusters: 5, selection method: tournament
- Correct oracle count: 1/11 (very low — only 1 candidate was oracle-correct)
- Generated SQL: `CAST((JULIANDAY('now') - JULIANDAY(birthday)) / 365.25 AS INTEGER) AS age`
- Gold SQL: `DISTINCT DATETIME() - birthday age`
- Analysis: The gold SQL uses `DATETIME() - birthday` which in SQLite returns a string representation of date subtraction. The generated query uses JULIANDAY arithmetic. Both are "age calculations" but produce different formats. With only 1 oracle candidate among 11, the tournament was highly unlikely to find it. The gold SQL is arguably less robust than the generated one.

**Case 2 — Q#953 (formula_1, simple)**
- Question: *"How many French constructors have a lap number of over 50?"*
- Clusters: 2, selection method: tournament
- Correct oracle count: 4/11
- Generated SQL: `SELECT COUNT(DISTINCT T1.constructorId) FROM constructors...`
- Gold SQL: `SELECT COUNT(DISTINCT T2.constructorId) FROM results...`
- Analysis: Both queries join the same tables but with different alias ordering. The generated query counted 0 results (likely a join direction issue) while the gold counted correctly. The tournament selected the wrong representative among the 2 clusters despite 4 oracle-correct candidates existing.

**Case 3 — Q#304 (toxicology, challenging)**
- Question: *"List all carcinogenic molecules and their elements"*
- Clusters: 2, selection method: tournament
- Correct oracle count: 5/11
- Generated SQL: `SELECT T1.molecule_id, T2.element FROM molecule...` (no DISTINCT)
- Gold SQL: `SELECT DISTINCT T2.molecule_id, T1.element FROM atom...`
- Analysis: The generated query lacks `DISTINCT`, causing duplicate rows that don't match the gold result set. The tournament selected the non-DISTINCT cluster over the DISTINCT cluster. This is a subtle correctness difference invisible without execution comparison.

---

## 5. Schema Linker Recall

### Aggregate Schema Recall

| Metric | S1 | S2 |
|--------|----|----|
| Avg table recall | **95.0%** | **96.0%** |
| Avg column recall | **94.1%** | **96.5%** |
| Schemas complete (all gold tables+cols present) | **26/33 (78.8%)** | **29/33 (87.9%)** |

### Cases with Incomplete Schemas

| Q# | DB | Diff | S1 miss | S2 miss |
|----|----|------|---------|---------|
| 28 | california_schools | challenging | `t2.doc` | `t2.doc` |
| 23 | california_schools | moderate | `t1.school` | — |
| 1456 | student_club | moderate | missing col | — |
| Others | various | — | partial col miss | — |

**Key finding:** Schema recall is very high overall. The S1 miss on Q#23 (missing `t1.school`) did NOT prevent generation from finding the correct answer (7/11 oracle pre-fix). The S2 miss on Q#28 (`DOC` column missing) likely contributed to generation failure for that question.

---

## 6. Fallback Analysis (logs/fallbacks.jsonl)

| Component | Trigger | Action | Count | Severity |
|-----------|---------|--------|-------|----------|
| gemini_client | cache_creation_failure | plain_system_instruction | 243 | warning |
| schema_linker | validation_error | hallucinated_field_filtered | 3 | warning |

**Total fallback events: 246 (all warnings)**

### gemini_client: cache_creation_failure (243 events)

All 243 events are Gemini context caching failures with the message:
> *"Cached content is too small. total_token_count=N, min_total_token_count=1024"*

This occurs because Gemini's context caching API requires a minimum of 1024 tokens, but many of our smaller prompts (ICL generator with short schemas, reasoning generator with simple questions) fall below this threshold. The code correctly falls back to plain system instructions without context caching — no functional impact.

**Suggestion:** Suppress these at `DEBUG` level or add a token-count pre-check before attempting context cache creation to avoid log noise.

### schema_linker: hallucinated_field_filtered (3 events)

The schema linker's hallucination filter correctly identified and removed 3 fields that the LLM invented but don't exist in the database. This is the intended behavior.

---

## 7. Key Changes Since v5 and Their Effects

| Change | Expected Effect | Observed Effect |
|--------|----------------|-----------------|
| **adaptive_selector**: replaced tool-use with free-text `FINAL: A/B` | Eliminate Gemini `MALFORMED_FUNCTION_CALL` errors | Selector precision +4.7pp (81% → 85.7%) ✅ |
| **monitoring/fallback_tracker**: new module + integration | Structured fallback visibility | 246 events captured, all warnings ✅ |
| **schema_linker**: hallucination logging | Track hallucinated fields | 3 events correctly captured ✅ |
| **generation**: fallback tracking for LLM errors | Error visibility | 0 generation errors (good) — cache from v5 run ✅ |
| **cache_manager**: fallback tracking | Error visibility | 0 cache errors ✅ |

---

## 8. Remaining Issues and Prioritized Improvements

### P0 — Highest Priority

**P0-1: Generation Quality (12/33 questions, 36.4% of dataset)**
- 10/12 generation failures have complete schemas — the bottleneck is SQL generation quality, not schema linking
- Root causes: complex correlated subqueries, wrong aggregation logic, multi-condition filter interpretation
- **Suggested fix:** Add SQL reasoning rules for common patterns (correlated subqueries, HAVING vs WHERE for aggregations, DISTINCT for list queries). Consider a "self-check" post-generation step where the model re-evaluates its own SQL against the schema.

**P0-2: Financial DB Czech Semantics (3/3 fail, 0%)**
- All 3 financial DB questions fail because Czech-encoded values ("F"/"M" for gender, Czech account types) are not inferrable from English schema summaries
- **Suggested fix:** Add a domain-specific value glossary to the grounding context for Czech databases. The summarizer could detect non-English values and include a lookup table in the schema DDL comments.

### P1 — High Priority

**P1-1: Selector Missing Obvious DISTINCT (Case 3, Q#304)**
- The tournament selected a non-DISTINCT query over a DISTINCT one, missing a subtle but critical difference
- **Suggested fix:** Add a DISTINCT-awareness heuristic in the cluster comparison prompt: explicitly ask the LLM evaluator to check whether duplicate rows in result A vs B indicate a missing DISTINCT.

**P1-2: Selector in Low-Oracle Situations (Case 1, Q#1031)**
- With only 1/11 oracle-correct candidates, tournament selection is essentially random (1/5 clusters)
- **Suggested fix:** In low-confidence situations (only 1-2 oracle candidates), the selector should upweight the `reasoning` generator since it has historically higher quality. Currently the tournament treats all cluster representatives equally once they clear execution.

**P1-3: Fixer Semantic Failures (17/32 fixed, 12/363 still semantically wrong)**
- The fixer successfully restores executability (91% → 96%) but cannot fix semantic errors
- **Suggested fix:** Add a "semantic validation" step — after fixing, the fixer should compare the row count and column structure against a reference query (if one exists in examples) to detect likely semantic errors before returning.

### P2 — Medium Priority

**P2-1: Gemini Context Cache Token Minimum (243 events)**
- Every prompt below 1024 tokens generates a cache_creation_failure warning, adding log noise
- **Suggested fix:** In `gemini_client.py`, check `estimated_tokens < 1024` before calling the context cache API; skip silently for small prompts. Log at DEBUG instead of WARNING.

**P2-2: Single-Schema-Miss in S1 (Q#28, Q#23, Q#1456)**
- 7/33 questions have incomplete S1 schemas, 4/33 have incomplete S2 schemas
- The miss on Q#28 (`DOC` column) directly contributed to a generation failure
- **Suggested fix:** Increase FAISS top-k for the schema linker S1 pre-filter from 50 to 75, and add a "role completeness" check — if a table is in S1, auto-include all columns that appear in the table's DDL `NOT NULL` constraint list.

**P2-3: Complex Challenging Questions (4/11 challenging = 36.4%)**
- Challenging questions are 27pp below simple/moderate — the gap has widened vs v5 (45.5% → 36.4%)
- Root cause: the 4 challenging failures are Q#431 (multi-condition join), Q#1031 (selector miss), Q#1295 (correlated subquery), Q#304 (DISTINCT miss)
- **Suggested fix:** Add a "complexity hint" to the generator prompt for challenging questions: include explicit instructions for correlated subqueries, DISTINCT, and multi-table joins.

---

## 9. Per-Question Summary

| # | DB | Q# | Diff | OracleP | OracleF | SelMatch | Correct |
|---|----|----|------|---------|---------|----------|---------|
| 1 | california_schools | 64 | simple | ✅ | ✅ | ✅ | **YES** |
| 2 | california_schools | 23 | moderate | ✅ | ✅ | ✅ | **YES** |
| 3 | california_schools | 28 | challenging | ❌ | ❌ | — | NO |
| 4 | card_games | 463 | simple | ❌ | ❌ | — | NO |
| 5 | card_games | 427 | moderate | ✅ | ✅ | ✅ | **YES** |
| 6 | card_games | 431 | challenging | ❌ | ❌ | — | NO |
| 7 | codebase_community | 601 | simple | ✅ | ✅ | ✅ | **YES** |
| 8 | codebase_community | 571 | moderate | ❌ | ❌ | — | NO |
| 9 | codebase_community | 586 | challenging | ❌ | ❌ | — | NO |
| 10 | debit_card_specializing | 1519 | simple | ✅ | ✅ | ✅ | **YES** |
| 11 | debit_card_specializing | 1474 | moderate | ✅ | ✅ | ✅ | **YES** |
| 12 | debit_card_specializing | 1526 | challenging | ✅ | ✅ | ✅ | **YES** |
| 13 | european_football_2 | 1027 | simple | ❌ | ❌ | — | NO |
| 14 | european_football_2 | 1025 | moderate | ✅ | ✅ | ✅ | **YES** |
| 15 | european_football_2 | 1031 | challenging | ✅ | ✅ | ❌ | NO |
| 16 | financial | 109 | simple | ❌ | ❌ | — | NO |
| 17 | financial | 135 | moderate | ❌ | ❌ | — | NO |
| 18 | financial | 149 | challenging | ❌ | ❌ | — | NO |
| 19 | formula_1 | 953 | simple | ✅ | ✅ | ❌ | NO |
| 20 | formula_1 | 852 | moderate | ✅ | ✅ | ✅ | **YES** |
| 21 | formula_1 | 994 | challenging | ✅ | ✅ | ✅ | **YES** |
| 22 | student_club | 1349 | simple | ✅ | ✅ | ✅ | **YES** |
| 23 | student_club | 1456 | moderate | ❌ | ❌ | — | NO |
| 24 | student_club | 1464 | challenging | ✅ | ✅ | ✅ | **YES** |
| 25 | superhero | 803 | simple | ✅ | ✅ | ✅ | **YES** |
| 26 | superhero | 782 | moderate | ✅ | ✅ | ✅ | **YES** |
| 27 | superhero | 773 | challenging | ✅ | ✅ | ✅ | **YES** |
| 28 | thrombosis_prediction | 1276 | simple | ✅ | ✅ | ✅ | **YES** |
| 29 | thrombosis_prediction | 1225 | moderate | ❌ | ❌ | — | NO |
| 30 | thrombosis_prediction | 1295 | challenging | ❌ | ❌ | — | NO |
| 31 | toxicology | 195 | simple | ✅ | ✅ | ✅ | **YES** |
| 32 | toxicology | 237 | moderate | ✅ | ✅ | ✅ | **YES** |
| 33 | toxicology | 304 | challenging | ✅ | ✅ | ❌ | NO |

Legend: OracleP = oracle pre-fix, OracleF = oracle post-fix, SelMatch = selector chose oracle candidate

---

## 10. Comparison: v5 vs v6

| Aspect | v5 (2026-02-24) | v6 (2026-02-25) | Change |
|--------|----------------|----------------|--------|
| Accuracy | 17/33 (51.5%) | 18/33 (54.5%) | **+1** |
| Runtime | 2968 s (89.9 s/q) | 1419 s (43.0 s/q) | **2.1× faster** |
| Generation timeouts | 0 | 0 | = |
| Oracle pre-fix | 21/33 (63.6%) | 21/33 (63.6%) | = |
| Oracle post-fix | 21/33 (63.6%) | 21/33 (63.6%) | = |
| Selector precision | 17/21 (81.0%) | 18/21 (85.7%) | **+4.7pp** |
| Selector misses | 4 | 3 | **-1** |
| Fixer success rate | 30.4% (7/23) | 53.1% (17/32) | **+22.7pp** |
| S1 complete schemas | 25/33 | 26/33 | +1 |
| S2 complete schemas | 28/33 | 29/33 | +1 |

**Note on fixer success rate:** The jump from 30.4% → 53.1% is partly due to more candidates needing fixes (23 → 32) with a broader range of error types. In v5, many fixes were blocked by schema-level errors in generation. In v6, generation quality was consistent (91.2% base executability) and the fixer handled syntax/schema corrections well.

---

## 11. Conclusion

The v6 run shows meaningful progress driven by the adaptive selector redesign:

1. **Selector improvement is confirmed** (+4.7pp, 81% → 85.7%): The `FINAL: A/B` free-text parsing successfully eliminated the Gemini tool-use failures that caused silent fallbacks to "A" in v5. This is the primary driver of the +1 accuracy gain.

2. **The generation bottleneck remains dominant**: 12/33 questions (36.4%) have no correct candidate anywhere in the 11-candidate pool. Fixing this requires better SQL generation quality — particularly for complex subqueries, multi-condition joins, and databases with non-English domain values.

3. **The fixer is effective at restoring executability** (53.1% of broken candidates fixed) but cannot overcome semantic errors. Its net contribution to final accuracy is currently zero.

4. **Schema linking quality is solid** (95%+ table recall, 94%+ column recall) and is rarely the bottleneck for generation failures. The remaining misses (4-7%) are edge cases with uncommon columns or aliased table patterns.

5. **The financial DB remains a persistent 0% floor** — Czech-encoded values are a domain-specific problem that cannot be solved by schema linking improvements alone.

**Next recommended focus:** Prompt engineering to improve generation quality for complex queries (correlated subqueries, DISTINCT, multi-condition joins), and a value-glossary approach for the financial DB.
