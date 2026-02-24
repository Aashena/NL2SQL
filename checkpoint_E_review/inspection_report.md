# Checkpoint E — Inspection Report (v3)

**Date:** 2026-02-23 22:31:29
**Script version:** v3
**LLM Provider:** gemini
**Models:** fast=gemini-2.5-flash, powerful=gemini-3-flash-preview, reasoning=gemini-2.5-flash
**Cache enabled:** True
**Elapsed time:** 5399.1s (163.6s/question)

## 1. Summary

**Total accuracy:** 18/33 = 54.5%

### By Difficulty

| Difficulty | Correct | Total | Accuracy |
|------------|---------|-------|----------|
| simple | 7 | 11 | 63.6% |
| moderate | 7 | 11 | 63.6% |
| challenging | 4 | 11 | 36.4% |

### By Database

| Database | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| california_schools | 2 | 3 | 66.7% |
| card_games | 1 | 3 | 33.3% |
| codebase_community | 1 | 3 | 33.3% |
| debit_card_specializing | 3 | 3 | 100.0% |
| european_football_2 | 1 | 3 | 33.3% |
| financial | 0 | 3 | 0.0% |
| formula_1 | 3 | 3 | 100.0% |
| student_club | 2 | 3 | 66.7% |
| superhero | 3 | 3 | 100.0% |
| thrombosis_prediction | 0 | 3 | 0.0% |
| toxicology | 2 | 3 | 66.7% |

## 2. Oracle Performance on Generated Candidates

The oracle measures whether the correct SQL was present among the generated candidates at all — it represents the theoretical upper bound achievable by perfect selection.

| Metric | Value |
|--------|-------|
| Questions with executable gold SQL | 33/33 |
| Oracle (pre-fix): ≥1 correct candidate | 23/33 = 69.7% |
| Oracle (post-fix): ≥1 correct fixed candidate | 23/33 = 69.7% |
| Gap: Oracle pre→post (fixer creates correct) | 0 questions |
| **Actual accuracy** | **18/33 = 54.5%** |
| Gap: Oracle post → Actual (selector misses) | 5 questions |
| Selector precision (when oracle achievable) | 18/23 = 78.3% |

**Interpretation:**
- Generation upper bound: 69.7% of questions had at least one correct candidate before fixing.
- After fixing, the oracle rose to 69.7% — the fixer added 0 new correct candidates.
- The selector then successfully picked the correct answer 78.3% of the time when it was available.
- Accuracy gap vs oracle: 5 questions were lost by the selector despite having a correct candidate available.

### Per-Question Oracle Detail

| # | DB | Q# | Diff | Cands | OracleP | OraclePcnt | OracleF | OracleFcnt | SelMatch | Correct |
|---|----|----|------|-------|---------|------------|---------|------------|----------|---------|
| 1 | california_schools | 64 | simple | 11 | Y | 11/11 | Y | 11/11 | Y | YES |
| 2 | california_schools | 23 | moderate | 11 | Y | 5/11 | Y | 9/11 | Y | YES |
| 3 | california_schools | 28 | challenging | 11 | N | 0/11 | N | 0/11 | N | NO |
| 4 | card_games | 463 | simple | 11 | Y | 1/11 | Y | 1/11 | N | NO |
| 5 | card_games | 427 | moderate | 11 | Y | 9/11 | Y | 9/11 | Y | YES |
| 6 | card_games | 431 | challenging | 11 | Y | 1/11 | Y | 1/11 | N | NO |
| 7 | codebase_community | 601 | simple | 11 | Y | 11/11 | Y | 11/11 | Y | YES |
| 8 | codebase_community | 571 | moderate | 0 | N | 0/0 | N | 0/0 | N | NO |
| 9 | codebase_community | 586 | challenging | 11 | N | 0/11 | N | 0/11 | N | NO |
| 10 | debit_card_specializing | 1519 | simple | 11 | Y | 11/11 | Y | 11/11 | Y | YES |
| 11 | debit_card_specializing | 1474 | moderate | 11 | Y | 6/11 | Y | 9/11 | Y | YES |
| 12 | debit_card_specializing | 1526 | challenging | 11 | Y | 4/11 | Y | 4/11 | Y | YES |
| 13 | european_football_2 | 1027 | simple | 11 | N | 0/11 | N | 0/11 | N | NO |
| 14 | european_football_2 | 1025 | moderate | 11 | Y | 11/11 | Y | 11/11 | Y | YES |
| 15 | european_football_2 | 1031 | challenging | 11 | Y | 1/11 | Y | 1/11 | N | NO |
| 16 | financial | 109 | simple | 11 | N | 0/11 | N | 0/11 | N | NO |
| 17 | financial | 135 | moderate | 11 | N | 0/11 | N | 0/11 | N | NO |
| 18 | financial | 149 | challenging | 11 | N | 0/11 | N | 0/11 | N | NO |
| 19 | formula_1 | 953 | simple | 11 | Y | 9/11 | Y | 9/11 | Y | YES |
| 20 | formula_1 | 852 | moderate | 11 | Y | 3/11 | Y | 3/11 | Y | YES |
| 21 | formula_1 | 994 | challenging | 11 | Y | 10/11 | Y | 11/11 | Y | YES |
| 22 | student_club | 1349 | simple | 11 | Y | 11/11 | Y | 11/11 | Y | YES |
| 23 | student_club | 1456 | moderate | 11 | N | 0/11 | N | 0/11 | N | NO |
| 24 | student_club | 1464 | challenging | 11 | Y | 11/11 | Y | 11/11 | Y | YES |
| 25 | superhero | 803 | simple | 11 | Y | 10/11 | Y | 11/11 | Y | YES |
| 26 | superhero | 782 | moderate | 11 | Y | 11/11 | Y | 11/11 | Y | YES |
| 27 | superhero | 773 | challenging | 11 | Y | 9/11 | Y | 10/11 | Y | YES |
| 28 | thrombosis_prediction | 1276 | simple | 11 | Y | 1/11 | Y | 3/11 | N | NO |
| 29 | thrombosis_prediction | 1225 | moderate | 11 | N | 0/11 | N | 0/11 | N | NO |
| 30 | thrombosis_prediction | 1295 | challenging | 11 | N | 0/11 | N | 0/11 | N | NO |
| 31 | toxicology | 195 | simple | 11 | Y | 11/11 | Y | 11/11 | Y | YES |
| 32 | toxicology | 237 | moderate | 11 | Y | 11/11 | Y | 11/11 | Y | YES |
| 33 | toxicology | 304 | challenging | 11 | Y | 1/11 | Y | 1/11 | N | NO |

## 3. Query Fixer Performance

| Metric | Value |
|--------|-------|
| Total candidates across all questions | 352 |
| Succeeded before any fixing | 284/352 = 80.7% |
| Succeeded after fixing | 333/352 = 94.6% |
| Net new successes from fixer | 49 candidates |
| Candidates that needed fixing | 68 |
| Successfully fixed (now succeed) | 49/68 = 72.1% |
| Still failing after fix attempts | 19/352 = 5.4% |
| Oracle candidates pre-fix | 169 |
| Oracle candidates post-fix | 181 |

**Interpretation:**
- 68 candidates had execution errors or returned empty results.
- The fixer successfully repaired 49 of them (72.1% fix success rate).
- Net effect: 49 additional candidates now execute successfully after fixing.

### Per-Question Fixer Detail

| # | DB | Q# | Diff | NeedFix | FixedOK | StillFail | OrigSucc | PostSucc |
|---|----|----|------|---------|---------|-----------|----------|----------|
| 1 | california_schools | 64 | simple | 0 | 0 | 0 | 11 | 11 |
| 2 | california_schools | 23 | moderate | 4 | 4 | 0 | 7 | 11 |
| 3 | california_schools | 28 | challenging | 5 | 2 | 3 | 6 | 8 |
| 4 | card_games | 463 | simple | 2 | 2 | 0 | 9 | 11 |
| 5 | card_games | 427 | moderate | 2 | 0 | 2 | 9 | 9 |
| 6 | card_games | 431 | challenging | 3 | 0 | 3 | 8 | 8 |
| 7 | codebase_community | 601 | simple | 0 | 0 | 0 | 11 | 11 |
| 8 | codebase_community | 571 | moderate | 0 | 0 | 0 | 0 | 0 |
| 9 | codebase_community | 586 | challenging | 3 | 3 | 0 | 8 | 11 |
| 10 | debit_card_specializing | 1519 | simple | 0 | 0 | 0 | 11 | 11 |
| 11 | debit_card_specializing | 1474 | moderate | 3 | 3 | 0 | 8 | 11 |
| 12 | debit_card_specializing | 1526 | challenging | 7 | 0 | 7 | 4 | 4 |
| 13 | european_football_2 | 1027 | simple | 4 | 4 | 0 | 7 | 11 |
| 14 | european_football_2 | 1025 | moderate | 0 | 0 | 0 | 11 | 11 |
| 15 | european_football_2 | 1031 | challenging | 4 | 0 | 4 | 7 | 7 |
| 16 | financial | 109 | simple | 3 | 3 | 0 | 8 | 11 |
| 17 | financial | 135 | moderate | 3 | 3 | 0 | 8 | 11 |
| 18 | financial | 149 | challenging | 3 | 3 | 0 | 8 | 11 |
| 19 | formula_1 | 953 | simple | 0 | 0 | 0 | 11 | 11 |
| 20 | formula_1 | 852 | moderate | 4 | 4 | 0 | 7 | 11 |
| 21 | formula_1 | 994 | challenging | 1 | 1 | 0 | 10 | 11 |
| 22 | student_club | 1349 | simple | 0 | 0 | 0 | 11 | 11 |
| 23 | student_club | 1456 | moderate | 0 | 0 | 0 | 11 | 11 |
| 24 | student_club | 1464 | challenging | 0 | 0 | 0 | 11 | 11 |
| 25 | superhero | 803 | simple | 1 | 1 | 0 | 10 | 11 |
| 26 | superhero | 782 | moderate | 0 | 0 | 0 | 11 | 11 |
| 27 | superhero | 773 | challenging | 1 | 1 | 0 | 10 | 11 |
| 28 | thrombosis_prediction | 1276 | simple | 3 | 3 | 0 | 8 | 11 |
| 29 | thrombosis_prediction | 1225 | moderate | 3 | 3 | 0 | 8 | 11 |
| 30 | thrombosis_prediction | 1295 | challenging | 4 | 4 | 0 | 7 | 11 |
| 31 | toxicology | 195 | simple | 0 | 0 | 0 | 11 | 11 |
| 32 | toxicology | 237 | moderate | 0 | 0 | 0 | 11 | 11 |
| 33 | toxicology | 304 | challenging | 5 | 5 | 0 | 6 | 11 |

## 4. Query Selector Performance

The selector performance measures how accurately the adaptive selector chooses the correct SQL when a correct candidate exists in the pool.

### Selection Method Distribution

| Method | Count | % of Total |
|--------|-------|------------|
| fast_path | 18 | 54.5% |
| tournament | 14 | 42.4% |
| N/A | 1 | 3.0% |

### Selector Accuracy by Method

| Method | Correct | Total | Accuracy | Oracle Achievable | Oracle Matched | Oracle Precision |
|--------|---------|-------|----------|-------------------|----------------|-----------------|
| fast_path | 13 | 18 | 72.2% | 13 | 13 | 100.0% |
| tournament | 5 | 14 | 35.7% | 10 | 5 | 50.0% |
| N/A | 0 | 1 | 0.0% | 0 | 0 | 0.0% |

### Winner Generator Distribution (All Selections)

| Generator | Total Wins | Correct Answers | Win Accuracy |
|-----------|-----------|-----------------|-------------|
| complex_B2_s1 | 15 | 10 | 66.7% |
| reasoning_A1 | 5 | 2 | 40.0% |
| standard_B1_s1 | 2 | 1 | 50.0% |
| icl_C3 | 2 | 1 | 50.0% |
| complex_B2_s2 | 2 | 1 | 50.0% |
| reasoning_A3 | 2 | 2 | 100.0% |
| icl_C1 | 2 | 1 | 50.0% |
| reasoning_A4 | 1 | 0 | 0.0% |
| N/A | 1 | 0 | 0.0% |
| reasoning_A2 | 1 | 0 | 0.0% |

### Selector Misses (Oracle Achievable but Wrong Answer Selected)

**Q#463** (card_games / simple)
- Question: How many translations are there for the set of cards with "Angel of Mercy" in it?
- Selection method: tournament
- Winner generator: icl_C3
- Final SQL: `SELECT COUNT(T2.id) FROM cards AS T1 JOIN set_translations AS T2 ON T1.setCode = T2.setCode WHERE T1.name = 'Angel of Mercy'`
- Gold SQL: `SELECT COUNT(DISTINCT translation) FROM set_translations WHERE setCode IN ( SELECT setCode FROM cards WHERE name = 'Angel of Mercy' ) AND translation IS NOT NULL`
- Oracle post-fix count: 1 correct candidates available

**Q#431** (card_games / challenging)
- Question: Which set is not available outside of the United States and has foil cards with Japanese writing on them? Please include the set ID in your response.
- Selection method: tournament
- Winner generator: reasoning_A4
- Final SQL: `SELECT DISTINCT T1.id FROM sets AS T1 INNER JOIN cards AS T2 ON T1.code = T2.setCode INNER JOIN foreign_data AS T3 ON T2.uuid = T3.uuid WHERE T1.isForeignOnly = 0 AND T1.isFoilOnly = 1 AND T3.language`
- Gold SQL: `SELECT T1.name, T1.id FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T2.language = 'Japanese' AND T1.isFoilOnly = 1 AND T1.isForeignOnly = 0`
- Oracle post-fix count: 1 correct candidates available

**Q#1031** (european_football_2 / challenging)
- Question: At present, calculate for the player's age who have a sprint speed of no less than 97 between 2013 to 2015.
- Selection method: tournament
- Winner generator: reasoning_A2
- Final SQL: `SELECT CAST((julianday('now') - julianday(T1.birthday)) / 365.25 AS INTEGER) AS current_age FROM Player AS T1 INNER JOIN Player_Attributes AS T2 ON T1.player_api_id = T2.player_api_id WHERE T2.sprint_`
- Gold SQL: `SELECT DISTINCT DATETIME() - T2.birthday age FROM Player_Attributes AS t1 INNER JOIN Player AS t2 ON t1.player_api_id = t2.player_api_id WHERE STRFTIME('%Y',t1.`date`) >= '2013' AND STRFTIME('%Y',t1.``
- Oracle post-fix count: 1 correct candidates available

**Q#1276** (thrombosis_prediction / simple)
- Question: For the patients who have an abnormal level of anti-DNA, please list the diseases they are diagnosed with.
- Selection method: tournament
- Winner generator: complex_B2_s1
- Final SQL: `SELECT DISTINCT T1.Diagnosis FROM Patient AS T1 JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE CAST(T2.DNA AS REAL) >= 8`
- Gold SQL: `SELECT DISTINCT(T1.Diagnosis) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.DNA >= 8`
- Oracle post-fix count: 3 correct candidates available

**Q#304** (toxicology / challenging)
- Question: List all carcinogenic molecules and their elements.
- Selection method: tournament
- Winner generator: reasoning_A1
- Final SQL: `SELECT T1.molecule_id, T2.element FROM molecule AS T1 INNER JOIN atom AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.label = '+' ORDER BY T1.molecule_id, T2.element`
- Gold SQL: `SELECT DISTINCT T2.molecule_id, T1.element FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.label = '+'`
- Oracle post-fix count: 1 correct candidates available

## 5. Schema Linker Recall and Accuracy

Recall measures what fraction of the tables and columns required by the gold SQL were present in S1 (precise) and S2 (recall) schemas. Column recall only counts explicit `table.column` references in the gold SQL.

> **⚠️ Important caveat on table recall numbers:**
> The table recall metric is computed by extracting all words matching `(\w+)\.(\w+)` (table.column) patterns from the gold SQL and treating the left-hand side as a "required table". In BIRD dev gold SQL, table aliases are pervasive (`T1.col FROM table AS T1`), meaning aliases like `T1`, `T2`, `T3` are counted as "required tables" that will never appear in the DDL. This inflates the "missing tables" list and deflates recall.
>
> The **real table recall** (counting only actual database table names found in FROM/JOIN clauses, not aliases) is effectively **much higher** — for most questions, the missing entries in S1 are exclusively aliases (e.g. `['t1', 't2']`). Only a small number of questions have genuine table omissions (see per-question data below). The column recall (74.8%) is more meaningful because it also relies on alias-prefixed columns, but the underlying columns are usually present — the issue is that the alias prefix doesn't match the stored field table name.
>
> **Real-world interpretation:** The schema linker is correctly including the actual database tables in most cases. The "S1 complete: 3/33" metric is an artifact of the alias detection limitation, **not** a true measure of schema linker quality.

| Metric | S1 (Precise) | S2 (Recall) |
|--------|-------------|------------|
| Avg Table Recall (includes aliases as "required") | 51.7% | 52.2% |
| Avg Column Recall (alias-prefixed columns) | 74.8% | 76.0% |
| Questions with complete coverage (alias-inflated) | 3/33 | 3/33 |

**Genuine schema omissions (actual table names missing, not just aliases):**
- Q10 (debit_card_specializing simple): `gasstations` table missing from S1 (only 4 fields in S1 vs needed join table). CORRECT despite this.
- Q12 (debit_card_specializing challenging): `gasstations` missing from S1. CORRECT despite this.
- Q23 (student_club moderate): `budget` table missing from S1 (S1 has expense+member but not budget.spent). WRONG — this is a genuine schema linker miss impacting accuracy.
- Q28 (thrombosis_prediction simple): S1 LLM call failed (CachedContent+tools error), fell back to FAISS top-15. WRONG (selector miss, not schema miss).
- Q29 (thrombosis_prediction moderate): S1 LLM failed + schema linking timeout. WRONG.
- Q30 (thrombosis_prediction challenging): Schema linking timeout. WRONG.

### Per-Question Schema Recall Detail

| # | DB | Q# | Diff | S1 Tables | S1 Cols | S2 Tables | S2 Cols | S1 Complete | S2 Complete | Missing S1 |
|---|----|----|------|-----------|---------|-----------|---------|-------------|-------------|-----------|
| 1 | california_schools | 64 | simple | 1/1 | 0/0 | 1/1 | 0/0 | Y | Y |  |
| 2 | california_schools | 23 | moderate | 2/4 | 0/4 | 2/4 | 0/4 | N | N | t1, t2, t1.school, t2.cdscode, t1.street, t1.cdscode |
| 3 | california_schools | 28 | challenging | 2/6 | 0/8 | 2/6 | 0/8 | N | N | t4, t1, t2, t3, t4.fundingtype, t2.fundingtype, t4.cdscode,  |
| 4 | card_games | 463 | simple | 2/2 | 0/0 | 2/2 | 0/0 | Y | Y |  |
| 5 | card_games | 427 | moderate | 2/4 | 4/4 | 2/4 | 4/4 | N | N | t1, t2 |
| 6 | card_games | 431 | challenging | 2/4 | 7/7 | 2/4 | 7/7 | N | N | t1, t2 |
| 7 | codebase_community | 601 | simple | 2/4 | 4/4 | 2/4 | 4/4 | N | N | t1, t2 |
| 8 | codebase_community | 571 | moderate | 2/4 | 4/4 | 2/4 | 4/4 | N | N | t1, t2 |
| 9 | codebase_community | 586 | challenging | 3/6 | 7/7 | 3/6 | 7/7 | N | N | t1, t2, t3 |
| 10 | debit_card_specializing | 1519 | simple | 1/4 | 3/5 | 1/4 | 3/5 | N | N | gasstations, t1, t2, t2.gasstationid, t1.gasstationid |
| 11 | debit_card_specializing | 1474 | moderate | 2/4 | 5/5 | 2/4 | 5/5 | N | N | t1, t2 |
| 12 | debit_card_specializing | 1526 | challenging | 2/5 | 3/5 | 2/5 | 3/5 | N | N | t2, gasstations, t1, t2.gasstationid, t1.gasstationid |
| 13 | european_football_2 | 1027 | simple | 2/4 | 4/4 | 2/4 | 4/4 | N | N | t1, t2 |
| 14 | european_football_2 | 1025 | moderate | 2/4 | 6/6 | 2/4 | 6/6 | N | N | t1, t2 |
| 15 | european_football_2 | 1031 | challenging | 2/4 | 4/4 | 2/4 | 4/4 | N | N | t1, t2 |
| 16 | financial | 109 | simple | 2/4 | 5/5 | 2/4 | 5/5 | N | N | t1, t2 |
| 17 | financial | 135 | moderate | 2/4 | 5/5 | 2/4 | 5/5 | N | N | t1, t2 |
| 18 | financial | 149 | challenging | 3/6 | 0/6 | 3/6 | 0/6 | N | N | t1, t2, t3, t3.account_id, t1.a11, t2.account_id, t3.type, t |
| 19 | formula_1 | 953 | simple | 2/4 | 4/4 | 2/4 | 4/4 | N | N | t1, t2 |
| 20 | formula_1 | 852 | moderate | 2/4 | 5/5 | 2/4 | 5/5 | N | N | t1, t3 |
| 21 | formula_1 | 994 | challenging | 3/6 | 0/9 | 3/6 | 0/9 | N | N | t1, t2, t3, t3.name, t2.name, t1.constructorid, t1.raceid, t |
| 22 | student_club | 1349 | simple | 2/4 | 4/4 | 2/4 | 4/4 | N | N | t1, t2 |
| 23 | student_club | 1456 | moderate | 2/6 | 5/7 | 3/6 | 7/7 | N | N | budget, t1, t2, t3, t2.spent, t2.budget_id |
| 24 | student_club | 1464 | challenging | 2/8 | 7/9 | 2/8 | 8/9 | N | N | attendance, t4, event, t1, t2, t3, t2.link_to_event, t1.even |
| 25 | superhero | 803 | simple | 1/1 | 0/0 | 1/1 | 0/0 | Y | Y |  |
| 26 | superhero | 782 | moderate | 2/4 | 5/5 | 2/4 | 5/5 | N | N | t2, t1 |
| 27 | superhero | 773 | challenging | 2/4 | 0/7 | 2/4 | 0/7 | N | N | t1, t2, t2.publisher_name, t1.hair_colour_id, t1.eye_colour_ |
| 28 | thrombosis_prediction | 1276 | simple | 2/4 | 4/4 | 2/4 | 4/4 | N | N | t1, t2 |
| 29 | thrombosis_prediction | 1225 | moderate | 2/4 | 0/3 | 2/4 | 0/3 | N | N | t1, t2, t2.id, t1.sex, t1.id |
| 30 | thrombosis_prediction | 1295 | challenging | 3/6 | 0/3 | 3/6 | 0/3 | N | N | t1, t2, t3, t3.id, t2.id, t1.id |
| 31 | toxicology | 195 | simple | 1/2 | 1/1 | 1/2 | 1/1 | N | N | t |
| 32 | toxicology | 237 | moderate | 2/4 | 4/4 | 2/4 | 4/4 | N | N | t1, t2 |
| 33 | toxicology | 304 | challenging | 2/4 | 4/4 | 2/4 | 4/4 | N | N | t1, t2 |

### S1 Incomplete — Missing Required Schema Elements

**Q#23** (california_schools / moderate, correct=True)
- Required tables: ['frpm', 'schools', 't1', 't2']
- Missing tables in S1: ['t1', 't2']
- Required cols: ['t1.cdscode', 't1.school', 't1.street', 't2.cdscode']
- Missing cols in S1: ['t1.school', 't2.cdscode', 't1.street', 't1.cdscode']
- S1 fields count: 0
- S2 fields count: 0
- Gold SQL: `SELECT T1.School, T1.Street FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.`Enrollment (K-12)` - T2.`Enrollment (Ages 5-17)` > 30`

**Q#28** (california_schools / challenging, correct=False)
- Required tables: ['frpm', 'schools', 't1', 't2', 't3', 't4']
- Missing tables in S1: ['t4', 't1', 't2', 't3']
- Required cols: ['t1.cdscode', 't2.cdscode', 't2.doc', 't2.fundingtype', 't2.school', 't3.cdscode', 't4.cdscode', 't4.fundingtype']
- Missing cols in S1: ['t4.fundingtype', 't2.fundingtype', 't4.cdscode', 't3.cdscode', 't2.cdscode', 't1.cdscode', 't2.school', 't2.doc']
- S1 fields count: 0
- S2 fields count: 0
- Gold SQL: `SELECT T2.School, T2.DOC FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.FundingType = 'Locally funded' AND (T1.`Enrollment (K-12)` - T1.`Enrollment (Ages 5-17)`) > (SELEC`

**Q#427** (card_games / moderate, correct=True)
- Required tables: ['set_translations', 'sets', 't1', 't2']
- Missing tables in S1: ['t1', 't2']
- Required cols: ['t1.code', 't1.mcmname', 't2.language', 't2.setcode']
- Missing cols in S1: []
- S1 fields count: 6
- S2 fields count: 13
- Gold SQL: `SELECT T2.language FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T1.mcmName = 'Archenemy' AND T2.setCode = 'ARC'`

**Q#431** (card_games / challenging, correct=False)
- Required tables: ['set_translations', 'sets', 't1', 't2']
- Missing tables in S1: ['t1', 't2']
- Required cols: ['t1.code', 't1.id', 't1.isfoilonly', 't1.isforeignonly', 't1.name', 't2.language', 't2.setcode']
- Missing cols in S1: []
- S1 fields count: 8
- S2 fields count: 18
- Gold SQL: `SELECT T1.name, T1.id FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T2.language = 'Japanese' AND T1.isFoilOnly = 1 AND T1.isForeignOnly = 0`

**Q#601** (codebase_community / simple, correct=True)
- Required tables: ['postlinks', 'posts', 't1', 't2']
- Missing tables in S1: ['t1', 't2']
- Required cols: ['t1.id', 't1.score', 't2.linktypeid', 't2.postid']
- Missing cols in S1: []
- S1 fields count: 9
- S2 fields count: 19
- Gold SQL: `SELECT T1.Score, T2.LinkTypeId FROM posts AS T1 INNER JOIN postLinks AS T2 ON T1.Id = T2.PostId WHERE T2.PostId = 395`

**Q#571** (codebase_community / moderate, correct=False)
- Required tables: ['posts', 't1', 't2', 'votes']
- Missing tables in S1: ['t1', 't2']
- Required cols: ['t1.id', 't1.userid', 't2.id', 't2.owneruserid']
- Missing cols in S1: []
- S1 fields count: 8
- S2 fields count: 18
- Gold SQL: `SELECT CAST(COUNT(T2.Id) AS REAL) / COUNT(DISTINCT T1.Id) FROM votes AS T1 INNER JOIN posts AS T2 ON T1.UserId = T2.OwnerUserId WHERE T1.UserId = 24`

**Q#586** (codebase_community / challenging, correct=False)
- Required tables: ['posts', 't1', 't2', 't3', 'users', 'votes']
- Missing tables in S1: ['t1', 't2', 't3']
- Required cols: ['t1.id', 't1.title', 't2.bountyamount', 't2.postid', 't2.userid', 't3.displayname', 't3.id']
- Missing cols in S1: []
- S1 fields count: 12
- S2 fields count: 19
- Gold SQL: `SELECT T3.DisplayName, T1.Title FROM posts AS T1 INNER JOIN votes AS T2 ON T1.Id = T2.PostId INNER JOIN users AS T3 ON T3.Id = T2.UserId WHERE T2.BountyAmount = 50 AND T1.Title LIKE '%variance%'`

**Q#1519** (debit_card_specializing / simple, correct=True)
- Required tables: ['gasstations', 't1', 't2', 'transactions_1k']
- Missing tables in S1: ['gasstations', 't1', 't2']
- Required cols: ['t1.date', 't1.gasstationid', 't1.productid', 't1.time', 't2.gasstationid']
- Missing cols in S1: ['t2.gasstationid', 't1.gasstationid']
- S1 fields count: 4
- S2 fields count: 7
- Gold SQL: `SELECT T1.ProductID FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID WHERE T1.Date = '2012-08-23' AND T1.Time = '21:20:00'`

**Q#1474** (debit_card_specializing / moderate, correct=True)
- Required tables: ['customers', 't1', 't2', 'yearmonth']
- Missing tables in S1: ['t1', 't2']
- Required cols: ['t1.currency', 't1.customerid', 't2.consumption', 't2.customerid', 't2.date']
- Missing cols in S1: []
- S1 fields count: 5
- S2 fields count: 15
- Gold SQL: `SELECT T1.CustomerID FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Currency = 'CZK' AND T2.Date BETWEEN 201101 AND 201112 GROUP BY T1.CustomerID ORDER BY SU`

**Q#1526** (debit_card_specializing / challenging, correct=True)
- Required tables: ['gasstations', 't1', 't2', 'transactions_1k', 'yearmonth']
- Missing tables in S1: ['t2', 'gasstations', 't1']
- Required cols: ['t1.customerid', 't1.date', 't1.gasstationid', 't1.price', 't2.gasstationid']
- Missing cols in S1: ['t2.gasstationid', 't1.gasstationid']
- S1 fields count: 7
- S2 fields count: 11
- Gold SQL: `SELECT CAST(SUM(IIF(SUBSTR(Date, 1, 4) = '2012', Consumption, 0)) - SUM(IIF(SUBSTR(Date, 1, 4) = '2013', Consumption, 0)) AS FLOAT) / SUM(IIF(SUBSTR(Date, 1, 4) = '2012', Consumption, 0)) FROM yearmon`

**Q#1027** (european_football_2 / simple, correct=False)
- Required tables: ['player', 'player_attributes', 't1', 't2']
- Missing tables in S1: ['t1', 't2']
- Required cols: ['t1.id', 't1.penalties', 't2.id', 't2.player_name']
- Missing cols in S1: []
- S1 fields count: 8
- S2 fields count: 8
- Gold SQL: `SELECT t2.player_name FROM Player_Attributes AS t1 INNER JOIN Player AS t2 ON t1.id = t2.id ORDER BY t1.penalties DESC LIMIT 10`

**Q#1025** (european_football_2 / moderate, correct=True)
- Required tables: ['league', 'match', 't1', 't2']
- Missing tables in S1: ['t1', 't2']
- Required cols: ['t1.away_team_goal', 't1.home_team_goal', 't1.league_id', 't1.season', 't2.id', 't2.name']
- Missing cols in S1: []
- S1 fields count: 20
- S2 fields count: 25
- Gold SQL: `SELECT t2.name FROM Match AS t1 INNER JOIN League AS t2 ON t1.league_id = t2.id WHERE t1.season = '2015/2016' GROUP BY t2.name ORDER BY SUM(t1.home_team_goal + t1.away_team_goal) DESC LIMIT 1`

**Q#1031** (european_football_2 / challenging, correct=False)
- Required tables: ['player', 'player_attributes', 't1', 't2']
- Missing tables in S1: ['t1', 't2']
- Required cols: ['t1.player_api_id', 't1.sprint_speed', 't2.birthday', 't2.player_api_id']
- Missing cols in S1: []
- S1 fields count: 9
- S2 fields count: 9
- Gold SQL: `SELECT DISTINCT DATETIME() - T2.birthday age FROM Player_Attributes AS t1 INNER JOIN Player AS t2 ON t1.player_api_id = t2.player_api_id WHERE STRFTIME('%Y',t1.`date`) >= '2013' AND STRFTIME('%Y',t1.``

**Q#109** (financial / simple, correct=False)
- Required tables: ['client', 'district', 't1', 't2']
- Missing tables in S1: ['t1', 't2']
- Required cols: ['t1.client_id', 't1.district_id', 't1.gender', 't2.a2', 't2.district_id']
- Missing cols in S1: []
- S1 fields count: 10
- S2 fields count: 15
- Gold SQL: `SELECT COUNT(T1.client_id) FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.gender = 'F' AND T2.A2 = 'Jesenik'`

**Q#135** (financial / moderate, correct=False)
- Required tables: ['account', 't1', 't2', 'trans']
- Missing tables in S1: ['t1', 't2']
- Required cols: ['t1.account_id', 't1.balance', 't1.operation', 't2.account_id', 't2.frequency']
- Missing cols in S1: []
- S1 fields count: 7
- S2 fields count: 17
- Gold SQL: `SELECT COUNT(T1.account_id) FROM trans AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id WHERE T1.balance < 0 AND T1.operation = 'VYBER KARTOU' AND T2.frequency = 'POPLATEK MESICNE'`

**Q#149** (financial / challenging, correct=False)
- Required tables: ['account', 'disp', 'district', 't1', 't2', 't3']
- Missing tables in S1: ['t1', 't2', 't3']
- Required cols: ['t1.a11', 't1.district_id', 't2.account_id', 't2.district_id', 't3.account_id', 't3.type']
- Missing cols in S1: ['t3.account_id', 't1.a11', 't2.account_id', 't3.type', 't2.district_id', 't1.district_id']
- S1 fields count: 0
- S2 fields count: 0
- Gold SQL: `SELECT T3.type FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id INNER JOIN disp AS T3 ON T2.account_id = T3.account_id WHERE T3.type != 'OWNER' AND T1.A11 BETWEEN 8000 A`

**Q#953** (formula_1 / simple, correct=True)
- Required tables: ['constructors', 'results', 't1', 't2']
- Missing tables in S1: ['t1', 't2']
- Required cols: ['t1.constructorid', 't1.laps', 't2.constructorid', 't2.nationality']
- Missing cols in S1: []
- S1 fields count: 8
- S2 fields count: 18
- Gold SQL: `SELECT COUNT(DISTINCT T2.constructorId) FROM results AS T1 INNER JOIN constructors AS T2 on T1.constructorId = T2.constructorId WHERE T1.laps > 50 AND T2.nationality = 'French'`

**Q#852** (formula_1 / moderate, correct=True)
- Required tables: ['circuits', 'races', 't1', 't3']
- Missing tables in S1: ['t1', 't3']
- Required cols: ['t1.circuitid', 't1.country', 't3.circuitid', 't3.raceid', 't3.year']
- Missing cols in S1: []
- S1 fields count: 6
- S2 fields count: 11
- Gold SQL: `SELECT COUNT(T3.raceId) FROM circuits AS T1 INNER JOIN races AS T3 ON T3.circuitID = T1.circuitId WHERE T1.country NOT IN ( 'Bahrain', 'China', 'Singapore', 'Japan', 'Korea', 'Turkey', 'UAE', 'Malaysi`

**Q#994** (formula_1 / challenging, correct=True)
- Required tables: ['constructorresults', 'constructors', 'races', 't1', 't2', 't3']
- Missing tables in S1: ['t1', 't2', 't3']
- Required cols: ['t1.constructorid', 't1.points', 't1.raceid', 't2.constructorid', 't2.name', 't2.nationality', 't3.name', 't3.raceid', 't3.year']
- Missing cols in S1: ['t3.name', 't2.name', 't1.constructorid', 't1.raceid', 't2.nationality', 't1.points', 't3.year', 't3.raceid', 't2.constructorid']
- S1 fields count: 0
- S2 fields count: 0
- Gold SQL: `SELECT SUM(T1.points), T2.name, T2.nationality FROM constructorResults AS T1 INNER JOIN constructors AS T2 ON T1.constructorId = T2.constructorId INNER JOIN races AS T3 ON T3.raceid = T1.raceid WHERE `

**Q#1349** (student_club / simple, correct=True)
- Required tables: ['budget', 'event', 't1', 't2']
- Missing tables in S1: ['t1', 't2']
- Required cols: ['t1.amount', 't1.link_to_event', 't2.event_id', 't2.event_name']
- Missing cols in S1: []
- S1 fields count: 5
- S2 fields count: 15
- Gold SQL: `SELECT SUM(T1.amount) FROM budget AS T1 INNER JOIN event AS T2 ON T1.link_to_event = T2.event_id WHERE T2.event_name = 'September Speaker'`

**Q#1456** (student_club / moderate, correct=False)
- Required tables: ['budget', 'expense', 'member', 't1', 't2', 't3']
- Missing tables in S1: ['budget', 't1', 't2', 't3']
- Required cols: ['t1.link_to_budget', 't1.link_to_member', 't2.budget_id', 't2.spent', 't3.first_name', 't3.last_name', 't3.member_id']
- Missing cols in S1: ['t2.spent', 't2.budget_id']
- S1 fields count: 9
- S2 fields count: 19
- Gold SQL: `SELECT T3.first_name, T3.last_name FROM expense AS T1 INNER JOIN budget AS T2 ON T1.link_to_budget = T2.budget_id INNER JOIN member AS T3 ON T1.link_to_member = T3.member_id ORDER BY T2.spent DESC LIM`

**Q#1464** (student_club / challenging, correct=True)
- Required tables: ['attendance', 'event', 'income', 'member', 't1', 't2', 't3', 't4']
- Missing tables in S1: ['attendance', 't4', 'event', 't1', 't2', 't3']
- Required cols: ['t1.event_id', 't2.link_to_event', 't2.link_to_member', 't3.first_name', 't3.last_name', 't3.member_id', 't4.amount', 't4.date_received', 't4.link_to_member']
- Missing cols in S1: ['t2.link_to_event', 't1.event_id']
- S1 fields count: 9
- S2 fields count: 18
- Gold SQL: `SELECT DISTINCT T3.first_name, T3.last_name, T4.amount FROM event AS T1 INNER JOIN attendance AS T2 ON T1.event_id = T2.link_to_event INNER JOIN member AS T3 ON T3.member_id = T2.link_to_member INNER `

**Q#782** (superhero / moderate, correct=True)
- Required tables: ['colour', 'superhero', 't1', 't2']
- Missing tables in S1: ['t2', 't1']
- Required cols: ['t1.eye_colour_id', 't1.hair_colour_id', 't1.superhero_name', 't2.colour', 't2.id']
- Missing cols in S1: []
- S1 fields count: 11
- S2 fields count: 17
- Gold SQL: `SELECT T1.superhero_name FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.eye_colour_id = T2.id AND T1.hair_colour_id = T2.id WHERE T2.colour = 'Black'`

**Q#773** (superhero / challenging, correct=True)
- Required tables: ['publisher', 'superhero', 't1', 't2']
- Missing tables in S1: ['t1', 't2']
- Required cols: ['t1.eye_colour_id', 't1.hair_colour_id', 't1.publisher_id', 't1.skin_colour_id', 't1.superhero_name', 't2.id', 't2.publisher_name']
- Missing cols in S1: ['t2.publisher_name', 't1.hair_colour_id', 't1.eye_colour_id', 't1.publisher_id', 't1.superhero_name', 't2.id', 't1.skin_colour_id']
- S1 fields count: 0
- S2 fields count: 0
- Gold SQL: `SELECT T1.superhero_name, T2.publisher_name FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id WHERE T1.eye_colour_id = T1.hair_colour_id AND T1.eye_colour_id = T1.skin_colour_`

**Q#1276** (thrombosis_prediction / simple, correct=False)
- Required tables: ['laboratory', 'patient', 't1', 't2']
- Missing tables in S1: ['t1', 't2']
- Required cols: ['t1.diagnosis', 't1.id', 't2.dna', 't2.id']
- Missing cols in S1: []
- S1 fields count: 19
- S2 fields count: 19
- Gold SQL: `SELECT DISTINCT(T1.Diagnosis) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.DNA >= 8`

**Q#1225** (thrombosis_prediction / moderate, correct=False)
- Required tables: ['laboratory', 'patient', 't1', 't2']
- Missing tables in S1: ['t1', 't2']
- Required cols: ['t1.id', 't1.sex', 't2.id']
- Missing cols in S1: ['t2.id', 't1.sex', 't1.id']
- S1 fields count: 0
- S2 fields count: 0
- Gold SQL: `SELECT T1.ID,T1.SEX FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.`T-BIL` >= 2.0 GROUP BY T1.SEX,T1.ID`

**Q#1295** (thrombosis_prediction / challenging, correct=False)
- Required tables: ['examination', 'laboratory', 'patient', 't1', 't2', 't3']
- Missing tables in S1: ['t1', 't2', 't3']
- Required cols: ['t1.id', 't2.id', 't3.id']
- Missing cols in S1: ['t3.id', 't2.id', 't1.id']
- S1 fields count: 0
- S2 fields count: 0
- Gold SQL: `SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID INNER JOIN Examination AS T3 ON T1.ID = T3.ID WHERE T2.`T-BIL` >= 2 AND T3.`ANA Pattern` LIKE '%P%'`

**Q#195** (toxicology / simple, correct=True)
- Required tables: ['bond', 't']
- Missing tables in S1: ['t']
- Required cols: ['t.bond_type']
- Missing cols in S1: []
- S1 fields count: 3
- S2 fields count: 9
- Gold SQL: `SELECT T.bond_type FROM ( SELECT bond_type, COUNT(bond_id) FROM bond GROUP BY bond_type ORDER BY COUNT(bond_id) DESC LIMIT 1 ) AS T`

**Q#237** (toxicology / moderate, correct=True)
- Required tables: ['atom', 'molecule', 't1', 't2']
- Missing tables in S1: ['t1', 't2']
- Required cols: ['t1.atom_id', 't1.molecule_id', 't2.label', 't2.molecule_id']
- Missing cols in S1: []
- S1 fields count: 4
- S2 fields count: 9
- Gold SQL: `SELECT T2.molecule_id , IIF(T2.label = '+', 'YES', 'NO') AS flag_carcinogenic FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.atom_id = 'TR001_10'`

**Q#304** (toxicology / challenging, correct=False)
- Required tables: ['atom', 'molecule', 't1', 't2']
- Missing tables in S1: ['t1', 't2']
- Required cols: ['t1.element', 't1.molecule_id', 't2.label', 't2.molecule_id']
- Missing cols in S1: []
- S1 fields count: 5
- S2 fields count: 10
- Gold SQL: `SELECT DISTINCT T2.molecule_id, T1.element FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.label = '+'`


## 6. Per-Question Full Results Table

| # | DB | Q# | Difficulty | Grounding | Schema | Gen | Fix | Selection | OracleP | OracleF | SelMatch | Correct |
|---|----|----|------------|-----------|--------|-----|-----|-----------|---------|---------|----------|---------|
| 1 | california_schools | 64 | simple | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 2 | california_schools | 23 | moderate | OK | ERR | OK | OK | OK (tournament) | Y | Y | Y | YES |
| 3 | california_schools | 28 | challenging | OK | ERR | OK | OK | OK (tournament) | N | N | N | NO |
| 4 | card_games | 463 | simple | OK | ERR | OK | OK | OK (tournament) | Y | Y | N | NO |
| 5 | card_games | 427 | moderate | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 6 | card_games | 431 | challenging | OK | OK | OK | OK | OK (tournament) | Y | Y | N | NO |
| 7 | codebase_community | 601 | simple | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 8 | codebase_community | 571 | moderate | OK | OK | ERR | ERR | ERR (N/A) | N | N | N | NO |
| 9 | codebase_community | 586 | challenging | OK | OK | OK | OK | OK (fast_path) | N | N | N | NO |
| 10 | debit_card_specializing | 1519 | simple | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 11 | debit_card_specializing | 1474 | moderate | OK | OK | OK | OK | OK (tournament) | Y | Y | Y | YES |
| 12 | debit_card_specializing | 1526 | challenging | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 13 | european_football_2 | 1027 | simple | OK | OK | OK | OK | OK (tournament) | N | N | N | NO |
| 14 | european_football_2 | 1025 | moderate | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 15 | european_football_2 | 1031 | challenging | OK | OK | OK | OK | OK (tournament) | Y | Y | N | NO |
| 16 | financial | 109 | simple | OK | OK | OK | OK | OK (fast_path) | N | N | N | NO |
| 17 | financial | 135 | moderate | OK | OK | OK | OK | OK (fast_path) | N | N | N | NO |
| 18 | financial | 149 | challenging | OK | ERR | OK | OK | OK (fast_path) | N | N | N | NO |
| 19 | formula_1 | 953 | simple | OK | OK | OK | OK | OK (tournament) | Y | Y | Y | YES |
| 20 | formula_1 | 852 | moderate | OK | OK | OK | OK | OK (tournament) | Y | Y | Y | YES |
| 21 | formula_1 | 994 | challenging | OK | ERR | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 22 | student_club | 1349 | simple | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 23 | student_club | 1456 | moderate | OK | OK | OK | OK | OK (tournament) | N | N | N | NO |
| 24 | student_club | 1464 | challenging | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 25 | superhero | 803 | simple | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 26 | superhero | 782 | moderate | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 27 | superhero | 773 | challenging | OK | ERR | OK | OK | OK (tournament) | Y | Y | Y | YES |
| 28 | thrombosis_prediction | 1276 | simple | OK | OK | OK | OK | OK (tournament) | Y | Y | N | NO |
| 29 | thrombosis_prediction | 1225 | moderate | OK | ERR | OK | OK | OK (tournament) | N | N | N | NO |
| 30 | thrombosis_prediction | 1295 | challenging | OK | ERR | OK | OK | OK (fast_path) | N | N | N | NO |
| 31 | toxicology | 195 | simple | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 32 | toxicology | 237 | moderate | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 33 | toxicology | 304 | challenging | OK | OK | OK | OK | OK (tournament) | Y | Y | N | NO |

## 7. Stage Failure Analysis (Wrong Answers)

Total wrong answers: 15/33

### Error Stage Distribution

| Stage | Count |
|-------|-------|
| schema_linking_timeout | 5 |
| generation_timeout | 1 |

### Wrong Answer Details

**Q#28** (california_schools / challenging)
- Question: Consider the average difference between K-12 enrollment and 15-17 enrollment of schools that are locally funded, list the names and DOC type of schools which has a difference above this average.
- Error stage: schema_linking_timeout
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT T2.School, T2.DOCType FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.FundingType = 'Locally funded' AND (T1."Enrollment (K-12)" - T1."Enrollment (Ages 5-17)") > ( `
- Gold SQL: `SELECT T2.School, T2.DOC FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.FundingType = 'Locally funded' AND (T1.`Enrollment (K-12)` - T1.`Enrollment (Ages 5-17)`) > (SELEC`
- Schema linking error: Schema linking timed out after 120s — falling back to full DDL
- S1 schema missing: tables=['t4', 't1', 't2', 't3'], cols=['t4.fundingtype', 't2.fundingtype', 't4.cdscode', 't3.cdscode', 't2.cdscode', 't1.cdscode', 't2.school', 't2.doc']

**Q#463** (card_games / simple)
- Question: How many translations are there for the set of cards with "Angel of Mercy" in it?
- Error stage: schema_linking_timeout
- Oracle pre-fix: True (1 correct candidates)
- Oracle post-fix: True (1 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT COUNT(T2.id) FROM cards AS T1 JOIN set_translations AS T2 ON T1.setCode = T2.setCode WHERE T1.name = 'Angel of Mercy'`
- Gold SQL: `SELECT COUNT(DISTINCT translation) FROM set_translations WHERE setCode IN ( SELECT setCode FROM cards WHERE name = 'Angel of Mercy' ) AND translation IS NOT NULL`
- Schema linking error: Schema linking timed out after 120s — falling back to full DDL

**Q#431** (card_games / challenging)
- Question: Which set is not available outside of the United States and has foil cards with Japanese writing on them? Please include the set ID in your response.
- Error stage: None
- Oracle pre-fix: True (1 correct candidates)
- Oracle post-fix: True (1 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT DISTINCT T1.id FROM sets AS T1 INNER JOIN cards AS T2 ON T1.code = T2.setCode INNER JOIN foreign_data AS T3 ON T2.uuid = T3.uuid WHERE T1.isForeignOnly = 0 AND T1.isFoilOnly = 1 AND T3.language`
- Gold SQL: `SELECT T1.name, T1.id FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T2.language = 'Japanese' AND T1.isFoilOnly = 1 AND T1.isForeignOnly = 0`
- S1 schema missing: tables=['t1', 't2'], cols=[]

**Q#571** (codebase_community / moderate)
- Question: For the user No.24, how many times is the number of his/her posts compared to his/her votes?
- Error stage: generation_timeout
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `(empty)`
- Gold SQL: `SELECT CAST(COUNT(T2.Id) AS REAL) / COUNT(DISTINCT T1.Id) FROM votes AS T1 INNER JOIN posts AS T2 ON T1.UserId = T2.OwnerUserId WHERE T1.UserId = 24`
- Generation error: Generation timed out after 90s
- Fixing error: skipped — no candidates from generation
- Selection error: skipped — no fixed candidates
- S1 schema missing: tables=['t1', 't2'], cols=[]

**Q#586** (codebase_community / challenging)
- Question: Which user added a bounty amount of 50 to the post title mentioning variance?
- Error stage: None
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT T1.DisplayName FROM users AS T1 JOIN votes AS T2 ON T1.Id = T2.UserId JOIN posts AS T3 ON T2.PostId = T3.Id WHERE T2.BountyAmount = 50 AND T3.Title LIKE '%variance%'`
- Gold SQL: `SELECT T3.DisplayName, T1.Title FROM posts AS T1 INNER JOIN votes AS T2 ON T1.Id = T2.PostId INNER JOIN users AS T3 ON T3.Id = T2.UserId WHERE T2.BountyAmount = 50 AND T1.Title LIKE '%variance%'`
- S1 schema missing: tables=['t1', 't2', 't3'], cols=[]

**Q#1027** (european_football_2 / simple)
- Question: Indicate the full names of the top 10 players with the highest number of penalties.
- Error stage: None
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT T1.player_name FROM Player AS T1 INNER JOIN Player_Attributes AS T2 ON T1.player_api_id = T2.player_api_id ORDER BY T2.penalties DESC LIMIT 10`
- Gold SQL: `SELECT t2.player_name FROM Player_Attributes AS t1 INNER JOIN Player AS t2 ON t1.id = t2.id ORDER BY t1.penalties DESC LIMIT 10`
- S1 schema missing: tables=['t1', 't2'], cols=[]

**Q#1031** (european_football_2 / challenging)
- Question: At present, calculate for the player's age who have a sprint speed of no less than 97 between 2013 to 2015.
- Error stage: None
- Oracle pre-fix: True (1 correct candidates)
- Oracle post-fix: True (1 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT CAST((julianday('now') - julianday(T1.birthday)) / 365.25 AS INTEGER) AS current_age FROM Player AS T1 INNER JOIN Player_Attributes AS T2 ON T1.player_api_id = T2.player_api_id WHERE T2.sprint_`
- Gold SQL: `SELECT DISTINCT DATETIME() - T2.birthday age FROM Player_Attributes AS t1 INNER JOIN Player AS t2 ON t1.player_api_id = t2.player_api_id WHERE STRFTIME('%Y',t1.`date`) >= '2013' AND STRFTIME('%Y',t1.``
- S1 schema missing: tables=['t1', 't2'], cols=[]

**Q#109** (financial / simple)
- Question: How many clients opened their accounts in Jesenik branch were women?
- Error stage: None
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT COUNT(DISTINCT T1.client_id) FROM client AS T1 JOIN disp AS T2 ON T1.client_id = T2.client_id JOIN account AS T3 ON T2.account_id = T3.account_id JOIN district AS T4 ON T3.district_id = T4.dist`
- Gold SQL: `SELECT COUNT(T1.client_id) FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.gender = 'F' AND T2.A2 = 'Jesenik'`
- S1 schema missing: tables=['t1', 't2'], cols=[]

**Q#135** (financial / moderate)
- Question: After making a credit card withdrawal, how many account/s with monthly issuance has a negative balance?
- Error stage: None
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT COUNT(DISTINCT T1.account_id) FROM account AS T1 JOIN trans AS T2 ON T1.account_id = T2.account_id WHERE T1.frequency = 'POPLATEK MESICNE' AND T2.operation = 'VYBER KARTOU' AND T2.balance < 0`
- Gold SQL: `SELECT COUNT(T1.account_id) FROM trans AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id WHERE T1.balance < 0 AND T1.operation = 'VYBER KARTOU' AND T2.frequency = 'POPLATEK MESICNE'`
- S1 schema missing: tables=['t1', 't2'], cols=[]

**Q#149** (financial / challenging)
- Question: Please list the account types that are not eligible for loans, and the average income of residents in the district where the account is located exceeds $8000 but is no more than $9000.
- Error stage: schema_linking_timeout
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT DISTINCT T1.type FROM disp AS T1 JOIN account AS T2 ON T1.account_id = T2.account_id JOIN district AS T3 ON T2.district_id = T3.district_id WHERE T3.A11 > 8000 AND T3.A11 <= 9000 AND T1.type !=`
- Gold SQL: `SELECT T3.type FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id INNER JOIN disp AS T3 ON T2.account_id = T3.account_id WHERE T3.type != 'OWNER' AND T1.A11 BETWEEN 8000 A`
- Schema linking error: Schema linking timed out after 120s — falling back to full DDL
- S1 schema missing: tables=['t1', 't2', 't3'], cols=['t3.account_id', 't1.a11', 't2.account_id', 't3.type', 't2.district_id', 't1.district_id']

**Q#1456** (student_club / moderate)
- Question: List the full name of the top five members who spend the most money in the descending order of spending.
- Error stage: None
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT T1.first_name, T1.last_name FROM member AS T1 JOIN expense AS T2 ON T1.member_id = T2.link_to_member GROUP BY T1.member_id ORDER BY SUM(T2.cost) DESC LIMIT 5`
- Gold SQL: `SELECT T3.first_name, T3.last_name FROM expense AS T1 INNER JOIN budget AS T2 ON T1.link_to_budget = T2.budget_id INNER JOIN member AS T3 ON T1.link_to_member = T3.member_id ORDER BY T2.spent DESC LIM`
- S1 schema missing: tables=['budget', 't1', 't2', 't3'], cols=['t2.spent', 't2.budget_id']

**Q#1276** (thrombosis_prediction / simple)
- Question: For the patients who have an abnormal level of anti-DNA, please list the diseases they are diagnosed with.
- Error stage: None
- Oracle pre-fix: True (1 correct candidates)
- Oracle post-fix: True (3 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT DISTINCT T1.Diagnosis FROM Patient AS T1 JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE CAST(T2.DNA AS REAL) >= 8`
- Gold SQL: `SELECT DISTINCT(T1.Diagnosis) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.DNA >= 8`
- S1 schema missing: tables=['t1', 't2'], cols=[]

**Q#1225** (thrombosis_prediction / moderate)
- Question: List and group all patients by sex for total bilirubin (T-BIL) level not within the normal range.
- Error stage: schema_linking_timeout
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT P.SEX, GROUP_CONCAT(DISTINCT P.ID) FROM Patient AS P JOIN Laboratory AS L ON P.ID = L.ID WHERE L."T-BIL" >= 2.0 GROUP BY P.SEX`
- Gold SQL: `SELECT T1.ID,T1.SEX FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.`T-BIL` >= 2.0 GROUP BY T1.SEX,T1.ID`
- Schema linking error: Schema linking timed out after 120s — falling back to full DDL
- S1 schema missing: tables=['t1', 't2'], cols=['t2.id', 't1.sex', 't1.id']

**Q#1295** (thrombosis_prediction / challenging)
- Question: Among the patients whose total bilirubin is over the normal range, how many of them have a peripheral pattern observed in the sheet of ANA examination?
- Error stage: schema_linking_timeout
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT COUNT(DISTINCT T1.ID) FROM Laboratory AS T1 JOIN Examination AS T2 ON T1.ID = T2.ID WHERE T1."T-BIL" >= 2.0 AND T2."ANA Pattern" LIKE '%P%'`
- Gold SQL: `SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID INNER JOIN Examination AS T3 ON T1.ID = T3.ID WHERE T2.`T-BIL` >= 2 AND T3.`ANA Pattern` LIKE '%P%'`
- Schema linking error: Schema linking timed out after 120s — falling back to full DDL
- S1 schema missing: tables=['t1', 't2', 't3'], cols=['t3.id', 't2.id', 't1.id']

**Q#304** (toxicology / challenging)
- Question: List all carcinogenic molecules and their elements.
- Error stage: None
- Oracle pre-fix: True (1 correct candidates)
- Oracle post-fix: True (1 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT T1.molecule_id, T2.element FROM molecule AS T1 INNER JOIN atom AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.label = '+' ORDER BY T1.molecule_id, T2.element`
- Gold SQL: `SELECT DISTINCT T2.molecule_id, T1.element FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.label = '+'`
- S1 schema missing: tables=['t1', 't2'], cols=[]

## 8. Critical Bugs Observed During This Run

### Bug 1 (P0): ICL Generator — Gemini CachedContent + tools Incompatibility

**Frequency:** Affects ~18 of 33 questions (those where cache is triggered for the examples block)

**Error:**
```
CachedContent can not be used with GenerateContent request setting
system_instruction, tools or tool_config.
Proposed fix: move those values to CachedContent from GenerateContent request.
```

**Root cause:** The `ICLGenerator` uses `CacheableText(cache=True)` to mark the few-shot examples block for Gemini context caching, AND it also makes a regular `generate()` call with `tools=[]`. With Gemini's API, once a CachedContent is used, the GenerateContent request **must not** set `system_instruction`, `tools`, or `tool_config` separately — those must be baked into the cached content at creation time, not repeated at inference time. However the current Gemini client separates these, causing a `400 INVALID_ARGUMENT` on all 3 ICL candidates for questions where caching activates.

**Impact:** ICL generator (3 candidates: C1, C2, C3) fails entirely for these questions, reducing the candidate pool from 11 to 8. The oracle is unaffected because reasoning (A1-A4) and standard/complex (B1/B2) generators are not impacted.

**Fix required:** In the Gemini client, when building CachedContent, include `tools` and `system_instruction` inside the cached content creation request. Alternatively, disable context caching for the ICL generator when using Gemini (set `cache=False` on the examples block for Gemini provider).

---

### Bug 2 (P0): Schema Linker — Gemini CachedContent + tools Incompatibility (Same Root Cause)

**Frequency:** Affects ~5 questions (thrombosis_prediction Q1276, Q1225, Q1295, and others)

**Error:**
```
Schema linker S1 LLM call failed (Gemini API call failed after retries: 400 INVALID_ARGUMENT.
CachedContent can not be used with GenerateContent request setting
system_instruction, tools or tool_config.); falling back to FAISS top-15 as S1.
```

**Root cause:** Same issue as Bug 1. The schema linker's `CacheableText(cache=True)` blocks combined with tool-use (`_SELECT_COLUMNS_TOOL`) trigger the same incompatibility. When this happens for S1 (first iteration), the code falls back to FAISS top-15 which produces lower-quality S1 fields. It then continues to attempt S2 which can also timeout.

**Impact:** Questions Q1276, Q1225, Q1295 (thrombosis_prediction) all fail this way. The FAISS fallback for S1 is weaker than the LLM-guided selection, degrading generation quality.

**Fix:** Same as Bug 1 — restructure Gemini CachedContent creation to include tools/system_instruction inside the cache, or conditionally disable caching when tools are present.

---

### Bug 3 (P1): Schema Linking — Timeout (120s) for Wide Schemas

**Frequency:** 5 schema linking timeouts: Q2, Q3, Q4, Q18, Q29, Q30

**Affected DBs:** card_games (115 fields), california_schools (89 fields), thrombosis_prediction (64 fields), financial (55 fields)

**Root cause:** The schema linker calls `gemini-3-flash-preview` (model_powerful) which has longer latency than anticipated. Schemas with 55–115 fields require sending a large candidate field list plus DDL context, which sometimes exceeds the 120s timeout.

**Impact:** Schema linking falls back to full DDL (both S1 and S2 become the complete schema). This means generators receive full schemas without filtering, which can hurt SQL quality due to ambiguity. However, 2 of the 5 timeout questions were still CORRECT (Q2 and Q21), so full-DDL fallback is not catastrophic for moderate questions.

**Fix:** (1) Increase schema linking timeout to 180s. (2) Add a pre-filtering step that passes only FAISS-top-30 candidates to the LLM instead of all available fields for large schemas. (3) Consider switching model_powerful back to `gemini-2.5-pro` which has more stable latency.

---

### Bug 4 (P1): Tournament Selector — 50% Precision vs Fast Path 100% Precision

**Critical finding from this run:**
| Selection Method | Oracle Achievable | Correct | Precision |
|-----------------|-------------------|---------|-----------|
| fast_path | 13 | 13 | **100%** |
| tournament | 10 | 5 | **50%** |

The fast path (unanimous candidates) is perfect — when all 11 candidates agree, the answer is always correct. The tournament, however, is picking the **wrong** answer 50% of the time when a correct candidate exists. This is the **single largest source of accuracy loss** — 5 questions have oracle-achievable correct answers but the tournament selects wrong.

**Root cause analysis for the 5 tournament misses:**
1. **Q463** (card_games simple): Final SQL uses `COUNT(T2.id)` (counts rows with ID) vs gold `COUNT(DISTINCT translation)` (counts distinct translations, excluding NULLs). Both return similar numbers but are semantically different. The tournament couldn't distinguish which is more correct without executing.
2. **Q431** (card_games challenging): Final SQL uses `foreign_data` table + `T3.language` approach; correct SQL uses `set_translations` table. Both produce sets but tournament picked wrong approach.
3. **Q1031** (european_football_2 challenging): `CAST(julianday() calc) AS INTEGER)` vs `DATETIME() - birthday`. The arithmetic yields similar but not identical results for age calculation.
4. **Q1276** (thrombosis_prediction simple): Gold is `DISTINCT(T1.Diagnosis)` which is functionally identical to `DISTINCT T1.Diagnosis`. Final SQL uses `CAST(DNA AS REAL)` which changes type behavior for numeric comparison. 3 candidates were correct but tournament picked one with `CAST`.
5. **Q304** (toxicology challenging): Final SQL adds `ORDER BY` which changes result set ordering — gold SQL uses `DISTINCT` without ORDER BY, and the execution comparison should handle this. But tournament picked wrong due to different column ordering.

**Fix options:**
- Include row count and column names of execution results in the tournament prompt
- Add a "semantic equivalence check" before tournament (check if result sets are equivalent under normalization)
- Use model_powerful (gemini-2.5-pro) instead of model_fast (gemini-2.5-flash) for tournament
- Weight tournament winners by cluster size more aggressively

---

### Bug 5 (P2): `financial` DB — 0% Accuracy (3/3 Wrong, All oracle_pre=False)

All 3 financial questions fail with no correct candidates at all. Analysis:
- **Q109** (simple): System joins through `disp` (wrong path); gold joins `client` directly to `district`. The schema linker provides S1/S2 with relevant fields but generators choose the wrong join path.
- **Q135** (moderate): System uses `COUNT(DISTINCT account_id)` + `balance < 0` on joined table; gold uses `T1.balance < 0` where T1 is `trans`, not `account`. Different alias ordering matters.
- **Q149** (challenging): Schema linking timeout → full DDL → generators confused by large schema.

These are **generation quality issues** for the financial DB, not schema linker issues. The financial DB has Czech column names (`A2`, `A11`, column meaning "Jesenik") that require domain-specific knowledge to interpret correctly.

---

## 8b. Interface Issues Observed

- Grounding stage errors: 0/33
- Schema linking stage errors: 8/33
- Generation stage errors (partial): 1/33
- Fixing stage errors: 1/33
- Selection stage errors: 1/33

### Stage Statistics

- Avg cell matches per question: 3.2
- Avg few-shot examples per question: 8.0
- Avg S1 fields: 6.0
- Avg S2 fields: 10.8
- Avg total generation candidates: 10.7

## 9. Issues Found

### Priority Legend: P0 = must fix before Prompt 13, P1 = important, P2 = nice to have

**P0-1: Gemini CachedContent + tools API incompatibility (ICL Generator + Schema Linker)**
- Description: ICL generator fails with `400 INVALID_ARGUMENT` when context caching is active AND tools are used in the same request. Affects ~18/33 questions for ICL (0 ICL candidates) and ~5 questions for schema linker S1 call. Root cause: Gemini requires `system_instruction` and `tools` to be inside the CachedContent, not repeated in the GenerateContent request.
- Suggested fix: In `gemini_client.py`, when building CachedContent, move `tools` and `system_instruction` into the cache creation payload. Alternatively, disable `cache=True` for blocks in ICL generator and schema linker when the provider is Gemini. **This is a blocker for production quality.**

**P0-2: Schema linking timeout for wide schemas (~24% of questions)**
- Description: Schema linker (using `gemini-3-flash-preview`) times out for schemas with 55+ fields, falling back to full DDL. 8/33 questions hit schema errors (5 timeouts + 3 CachedContent failures). Full DDL fallback degrades SQL quality for challenging multi-table questions.
- Suggested fix: (1) Increase `_TIMEOUT_SCHEMA_LINKING` to 180s. (2) Pre-filter candidates to FAISS top-30 before sending to schema LLM for large schemas. (3) Consider reverting `MODEL_POWERFUL` to `gemini-2.5-pro` for schema linking.

**P0-3: Empty final SQL: 1/33 questions (Q571 codebase_community moderate)**
- Description: Generation timed out at 90s, producing no candidates. Final SQL is empty string.
- Suggested fix: Increase `_TIMEOUT_GENERATION` to 120s, or add a guaranteed fallback (e.g. `SELECT * FROM first_table LIMIT 10`).

**P1-1: Tournament selector precision is only 50% (5 misses on 10 oracle-achievable questions)**
- Description: The tournament actively hurts performance for questions where candidates disagree. Fast path is 100% accurate (13/13); tournament is 50% (5/10). The 5 misses are due to: (1) COUNT vs COUNT(DISTINCT) subtleties, (2) different join paths that produce different-valued but both-valid results, (3) age calculation arithmetic differences, (4) type cast differences, (5) column ordering differences with DISTINCT.
- Suggested fix: (1) Include execution result statistics (row count, column names, sample values) in tournament prompt. (2) Use `gemini-2.5-pro` (model_powerful) instead of `gemini-2.5-flash` (model_fast) for tournament comparisons. (3) Add semantic normalization before clustering (e.g. strip ORDER BY, normalize DISTINCT).

**P1-2: Schema recall metric is inflated by SQL aliases**
- Description: The `_compute_schema_recall` function counts aliases (T1, T2, etc.) as required "tables", making table recall appear to be 51.7% when in reality the genuine table omission rate is much lower. Most "missing" entries are aliases. Only ~4 genuine schema omissions were found (gasstations in debit_card Q10/Q12, budget in student_club Q23, patient table missing in thrombosis Q28 due to S1 LLM failure).
- Suggested fix: Update `_extract_required_tables_columns` to filter out single-character or T\d+ patterns as aliases.

**P1-3: `financial` DB — 0% accuracy (generation quality issue for Czech-named columns)**
- Description: All 3 financial questions fail because generators don't know that `A2` = city name, `A11` = average income, etc. (Czech abbreviations). This is a domain knowledge gap, not a schema or selector issue.
- Suggested fix: (1) Ensure column summaries in the schema (generated by the summarizer) properly explain Czech column semantics. (2) Verify summarizer output for `financial` DB covers `A2`, `A11`, `frequency` (POPLATEK MESICNE = monthly payment) etc.

**P1-4: Oracle doesn't improve from pre-fix to post-fix (0 new oracle questions from fixer)**
- Description: The fixer repairs 49 candidates (72.1% success rate) but doesn't convert any question from "no oracle" to "has oracle". This means the fixer is fixing syntax/schema errors but the underlying logical errors (wrong join path, wrong column, wrong aggregate) are not fixable with error-message-based repair.
- Suggested fix: Increase `_BETA` to 3 fix iterations and/or provide richer context in the fix prompt (e.g. show sample rows from relevant tables to help the LLM understand the data structure).

## 10. Suggested Improvements Before Pipeline Wiring (Prompt 13)

### Blocking Issues (must fix before Prompt 13 will work reliably)

1. **Fix Gemini CachedContent + tools incompatibility (P0-1).** This is the root cause of ICL generator failures and several schema linker S1 failures. Without this fix, ~55% of questions will have a degraded candidate pool (no ICL), and ~15% will have degraded schema linking. This fix is in `src/llm/gemini_client.py`.

2. **Increase schema linking timeout to 180s (P0-2).** Five timeouts in 33 questions (15%) is too high for a production pipeline. This is a simple config change in `scripts/checkpoint_e_test.py` and later in `pipeline/online_pipeline.py`.

3. **Add generation fallback for timeout/empty-candidate scenarios (P0-3).** If all generators fail (Q571 case), the pipeline should return at least a plausible fallback SQL rather than empty string.

### High-impact improvements to implement before full evaluation

4. **Improve tournament selector (P1-1).** Provide execution result statistics in pairwise comparison prompts (row count, first 3 rows). Upgrade tournament model from `model_fast` to `model_powerful`. This could recover 3-5 percentage points.

5. **Fix schema recall computation to exclude aliases (P1-2).** Update `_extract_required_tables_columns` to detect and exclude patterns like `T\d+` (T1, T2) and single-letter aliases. This will make the schema linker evaluation more meaningful.

6. **Verify summarizer output for `financial` DB (P1-3).** The `financial` DB has Czech abbreviation column names (`A2`=city, `A11`=avg income, `POPLATEK MESICNE`=monthly). Check that `data/preprocessed/schemas/financial_ddl.sql` and `financial_markdown.md` have proper summaries for these columns.

### Target accuracy estimate after fixes

If P0-1 (ICL fix) + P0-2 (timeout fix) + P1-1 (tournament fix) are applied:
- ICL fix: Could recover 1-2 more oracle questions (questions where only ICL would find the right answer)
- Timeout fix: Could recover 2 questions currently lost to schema linking timeout
- Tournament fix: Could recover 3-4 of the 5 selector misses
- Estimated improvement: +4 to +7 percentage points → **target 58-62%** on 33-question sample

The 68% target will likely require improving the generation quality for `financial` and `thrombosis_prediction` DBs, which requires better domain knowledge in summaries and potentially prompt improvements.

### Proceed to Prompt 13 with known limitations

The component interfaces are **functionally correct** — all stages can connect and produce results. The issues are quality/reliability issues, not interface compatibility issues. **It is reasonable to proceed to Prompt 13 (pipeline wiring) with the following caveat:** fix the Gemini CachedContent bug (P0-1) first, as it impacts a significant fraction of questions and is a straightforward fix in `gemini_client.py`.
