# Checkpoint E — Inspection Report (v3)

**Date:** 2026-02-24 13:33:05
**Script version:** v3
**LLM Provider:** gemini
**Models:** fast=gemini-2.5-flash, powerful=gemini-2.5-flash, reasoning=gemini-2.5-flash
**Cache enabled:** True
**Elapsed time:** 2967.8s (89.9s/question)

## 1. Summary

**Total accuracy:** 17/33 = 51.5%

### By Difficulty

| Difficulty | Correct | Total | Accuracy |
|------------|---------|-------|----------|
| simple | 7 | 11 | 63.6% |
| moderate | 5 | 11 | 45.5% |
| challenging | 5 | 11 | 45.5% |

### By Database

| Database | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| california_schools | 1 | 3 | 33.3% |
| card_games | 1 | 3 | 33.3% |
| codebase_community | 1 | 3 | 33.3% |
| debit_card_specializing | 3 | 3 | 100.0% |
| european_football_2 | 1 | 3 | 33.3% |
| financial | 0 | 3 | 0.0% |
| formula_1 | 1 | 3 | 33.3% |
| student_club | 2 | 3 | 66.7% |
| superhero | 3 | 3 | 100.0% |
| thrombosis_prediction | 1 | 3 | 33.3% |
| toxicology | 3 | 3 | 100.0% |

## 2. Oracle Performance on Generated Candidates

The oracle measures whether the correct SQL was present among the generated candidates at all — it represents the theoretical upper bound achievable by perfect selection.

| Metric | Value |
|--------|-------|
| Questions with executable gold SQL | 33/33 |
| Oracle (pre-fix): ≥1 correct candidate | 21/33 = 63.6% |
| Oracle (post-fix): ≥1 correct fixed candidate | 21/33 = 63.6% |
| Gap: Oracle pre→post (fixer creates correct) | 0 questions |
| **Actual accuracy** | **17/33 = 51.5%** |
| Gap: Oracle post → Actual (selector misses) | 4 questions |
| Selector precision (when oracle achievable) | 17/21 = 81.0% |

**Interpretation:**
- Generation upper bound: 63.6% of questions had at least one correct candidate before fixing.
- After fixing, the oracle rose to 63.6% — the fixer added 0 new correct candidates.
- The selector then successfully picked the correct answer 81.0% of the time when it was available.
- Accuracy gap vs oracle: 4 questions were lost by the selector despite having a correct candidate available.

### Per-Question Oracle Detail

| # | DB | Q# | Diff | Cands | OracleP | OraclePcnt | OracleF | OracleFcnt | SelMatch | Correct |
|---|----|----|------|-------|---------|------------|---------|------------|----------|---------|
| 1 | california_schools | 64 | simple | 11 | Y | 11/11 | Y | 11/11 | Y | YES |
| 2 | california_schools | 23 | moderate | 11 | Y | 3/11 | Y | 3/11 | N | NO |
| 3 | california_schools | 28 | challenging | 11 | N | 0/11 | N | 0/11 | N | NO |
| 4 | card_games | 463 | simple | 11 | N | 0/11 | N | 0/11 | N | NO |
| 5 | card_games | 427 | moderate | 11 | Y | 6/11 | Y | 8/11 | Y | YES |
| 6 | card_games | 431 | challenging | 11 | N | 0/11 | N | 0/11 | N | NO |
| 7 | codebase_community | 601 | simple | 11 | Y | 11/11 | Y | 11/11 | Y | YES |
| 8 | codebase_community | 571 | moderate | 11 | N | 0/11 | N | 0/11 | N | NO |
| 9 | codebase_community | 586 | challenging | 11 | N | 0/11 | N | 0/11 | N | NO |
| 10 | debit_card_specializing | 1519 | simple | 11 | Y | 11/11 | Y | 11/11 | Y | YES |
| 11 | debit_card_specializing | 1474 | moderate | 11 | Y | 7/11 | Y | 8/11 | Y | YES |
| 12 | debit_card_specializing | 1526 | challenging | 11 | Y | 7/11 | Y | 7/11 | Y | YES |
| 13 | european_football_2 | 1027 | simple | 11 | N | 0/11 | N | 0/11 | N | NO |
| 14 | european_football_2 | 1025 | moderate | 11 | Y | 11/11 | Y | 11/11 | Y | YES |
| 15 | european_football_2 | 1031 | challenging | 11 | N | 0/11 | N | 0/11 | N | NO |
| 16 | financial | 109 | simple | 11 | Y | 5/11 | Y | 5/11 | N | NO |
| 17 | financial | 135 | moderate | 11 | N | 0/11 | N | 0/11 | N | NO |
| 18 | financial | 149 | challenging | 11 | N | 0/11 | N | 0/11 | N | NO |
| 19 | formula_1 | 953 | simple | 11 | Y | 3/11 | Y | 3/11 | N | NO |
| 20 | formula_1 | 852 | moderate | 11 | Y | 2/11 | Y | 2/11 | N | NO |
| 21 | formula_1 | 994 | challenging | 11 | Y | 10/11 | Y | 11/11 | Y | YES |
| 22 | student_club | 1349 | simple | 11 | Y | 11/11 | Y | 11/11 | Y | YES |
| 23 | student_club | 1456 | moderate | 11 | N | 0/11 | N | 0/11 | N | NO |
| 24 | student_club | 1464 | challenging | 11 | Y | 9/11 | Y | 9/11 | Y | YES |
| 25 | superhero | 803 | simple | 11 | Y | 11/11 | Y | 11/11 | Y | YES |
| 26 | superhero | 782 | moderate | 11 | Y | 11/11 | Y | 11/11 | Y | YES |
| 27 | superhero | 773 | challenging | 11 | Y | 11/11 | Y | 11/11 | Y | YES |
| 28 | thrombosis_prediction | 1276 | simple | 11 | Y | 5/11 | Y | 5/11 | Y | YES |
| 29 | thrombosis_prediction | 1225 | moderate | 11 | N | 0/11 | N | 0/11 | N | NO |
| 30 | thrombosis_prediction | 1295 | challenging | 11 | N | 0/11 | N | 0/11 | N | NO |
| 31 | toxicology | 195 | simple | 11 | Y | 10/11 | Y | 11/11 | Y | YES |
| 32 | toxicology | 237 | moderate | 11 | Y | 11/11 | Y | 11/11 | Y | YES |
| 33 | toxicology | 304 | challenging | 11 | Y | 6/11 | Y | 6/11 | Y | YES |

## 3. Query Fixer Performance

| Metric | Value |
|--------|-------|
| Total candidates across all questions | 363 |
| Succeeded before any fixing | 340/363 = 93.7% |
| Succeeded after fixing | 347/363 = 95.6% |
| Net new successes from fixer | 7 candidates |
| Candidates that needed fixing | 23 |
| Successfully fixed (now succeed) | 7/23 = 30.4% |
| Still failing after fix attempts | 16/363 = 4.4% |
| Oracle candidates pre-fix | 172 |
| Oracle candidates post-fix | 177 |

**Interpretation:**
- 23 candidates had execution errors or returned empty results.
- The fixer successfully repaired 7 of them (30.4% fix success rate).
- Net effect: 7 additional candidates now execute successfully after fixing.

### Per-Question Fixer Detail

| # | DB | Q# | Diff | NeedFix | FixedOK | StillFail | OrigSucc | PostSucc |
|---|----|----|------|---------|---------|-----------|----------|----------|
| 1 | california_schools | 64 | simple | 0 | 0 | 0 | 11 | 11 |
| 2 | california_schools | 23 | moderate | 0 | 0 | 0 | 11 | 11 |
| 3 | california_schools | 28 | challenging | 2 | 1 | 1 | 9 | 10 |
| 4 | card_games | 463 | simple | 0 | 0 | 0 | 11 | 11 |
| 5 | card_games | 427 | moderate | 5 | 2 | 3 | 6 | 8 |
| 6 | card_games | 431 | challenging | 7 | 0 | 7 | 4 | 4 |
| 7 | codebase_community | 601 | simple | 0 | 0 | 0 | 11 | 11 |
| 8 | codebase_community | 571 | moderate | 0 | 0 | 0 | 11 | 11 |
| 9 | codebase_community | 586 | challenging | 0 | 0 | 0 | 11 | 11 |
| 10 | debit_card_specializing | 1519 | simple | 0 | 0 | 0 | 11 | 11 |
| 11 | debit_card_specializing | 1474 | moderate | 1 | 1 | 0 | 10 | 11 |
| 12 | debit_card_specializing | 1526 | challenging | 4 | 0 | 4 | 7 | 7 |
| 13 | european_football_2 | 1027 | simple | 0 | 0 | 0 | 11 | 11 |
| 14 | european_football_2 | 1025 | moderate | 0 | 0 | 0 | 11 | 11 |
| 15 | european_football_2 | 1031 | challenging | 1 | 0 | 1 | 10 | 10 |
| 16 | financial | 109 | simple | 0 | 0 | 0 | 11 | 11 |
| 17 | financial | 135 | moderate | 0 | 0 | 0 | 11 | 11 |
| 18 | financial | 149 | challenging | 0 | 0 | 0 | 11 | 11 |
| 19 | formula_1 | 953 | simple | 0 | 0 | 0 | 11 | 11 |
| 20 | formula_1 | 852 | moderate | 1 | 1 | 0 | 10 | 11 |
| 21 | formula_1 | 994 | challenging | 1 | 1 | 0 | 10 | 11 |
| 22 | student_club | 1349 | simple | 0 | 0 | 0 | 11 | 11 |
| 23 | student_club | 1456 | moderate | 0 | 0 | 0 | 11 | 11 |
| 24 | student_club | 1464 | challenging | 0 | 0 | 0 | 11 | 11 |
| 25 | superhero | 803 | simple | 0 | 0 | 0 | 11 | 11 |
| 26 | superhero | 782 | moderate | 0 | 0 | 0 | 11 | 11 |
| 27 | superhero | 773 | challenging | 0 | 0 | 0 | 11 | 11 |
| 28 | thrombosis_prediction | 1276 | simple | 0 | 0 | 0 | 11 | 11 |
| 29 | thrombosis_prediction | 1225 | moderate | 0 | 0 | 0 | 11 | 11 |
| 30 | thrombosis_prediction | 1295 | challenging | 0 | 0 | 0 | 11 | 11 |
| 31 | toxicology | 195 | simple | 1 | 1 | 0 | 10 | 11 |
| 32 | toxicology | 237 | moderate | 0 | 0 | 0 | 11 | 11 |
| 33 | toxicology | 304 | challenging | 0 | 0 | 0 | 11 | 11 |

## 4. Query Selector Performance

The selector performance measures how accurately the adaptive selector chooses the correct SQL when a correct candidate exists in the pool.

### Selection Method Distribution

| Method | Count | % of Total |
|--------|-------|------------|
| fast_path | 21 | 63.6% |
| tournament | 12 | 36.4% |

### Selector Accuracy by Method

| Method | Correct | Total | Accuracy | Oracle Achievable | Oracle Matched | Oracle Precision |
|--------|---------|-------|----------|-------------------|----------------|-----------------|
| fast_path | 13 | 21 | 61.9% | 13 | 13 | 100.0% |
| tournament | 4 | 12 | 33.3% | 8 | 4 | 50.0% |

### Winner Generator Distribution (All Selections)

| Generator | Total Wins | Correct Answers | Win Accuracy |
|-----------|-----------|-----------------|-------------|
| reasoning_A1 | 12 | 7 | 58.3% |
| icl_C1 | 4 | 2 | 50.0% |
| reasoning_A2 | 3 | 0 | 0.0% |
| standard_B1_s1 | 3 | 2 | 66.7% |
| reasoning_A3 | 2 | 0 | 0.0% |
| complex_B2_s1 | 2 | 2 | 100.0% |
| standard_B1_s2 | 2 | 0 | 0.0% |
| reasoning_A4 | 2 | 1 | 50.0% |
| icl_C2 | 1 | 1 | 100.0% |
| complex_B2_s2 | 1 | 1 | 100.0% |
| icl_C3 | 1 | 1 | 100.0% |

### Selector Misses (Oracle Achievable but Wrong Answer Selected)

**Q#23** (california_schools / moderate)
- Question: List the names of schools with more than 30 difference in enrollements between K-12 and ages 5-17? Please also give the full street adress of the schools.
- Selection method: tournament
- Winner generator: reasoning_A2
- Final SQL: `SELECT f."School Name", s.Street FROM frpm AS f JOIN schools AS s ON f.CDSCode = s.CDSCode WHERE ( f."Enrollment (K-12)" - f."Enrollment (Ages 5-17)" ) > 30`
- Gold SQL: `SELECT T1.School, T1.Street FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.`Enrollment (K-12)` - T2.`Enrollment (Ages 5-17)` > 30`
- Oracle post-fix count: 3 correct candidates available

**Q#109** (financial / simple)
- Question: How many clients opened their accounts in Jesenik branch were women?
- Selection method: tournament
- Winner generator: reasoning_A3
- Final SQL: `SELECT COUNT(DISTINCT T3.client_id) FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id INNER JOIN disp AS T4 ON T2.account_id = T4.account_id INNER JOIN client AS T3 ON T4`
- Gold SQL: `SELECT COUNT(T1.client_id) FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.gender = 'F' AND T2.A2 = 'Jesenik'`
- Oracle post-fix count: 5 correct candidates available

**Q#953** (formula_1 / simple)
- Question: How many French constructors have a lap number of over 50?
- Selection method: tournament
- Winner generator: reasoning_A4
- Final SQL: `SELECT COUNT(DISTINCT T1.constructorId) FROM constructors AS T1 INNER JOIN results AS T2 ON T1.constructorId = T2.constructorId INNER JOIN lapTimes AS T3 ON T2.raceId = T3.raceId WHERE T1.nationality `
- Gold SQL: `SELECT COUNT(DISTINCT T2.constructorId) FROM results AS T1 INNER JOIN constructors AS T2 on T1.constructorId = T2.constructorId WHERE T1.laps > 50 AND T2.nationality = 'French'`
- Oracle post-fix count: 3 correct candidates available

**Q#852** (formula_1 / moderate)
- Question: How many races in the year 2010 are held on grand prixs outside Asia and Europe?
- Selection method: tournament
- Winner generator: standard_B1_s1
- Final SQL: `SELECT COUNT(T1.raceId) FROM races AS T1 INNER JOIN circuits AS T2 ON T1.circuitId = T2.circuitId WHERE T1.year = 2010 AND T2.country NOT IN ('Asia', 'Europe')`
- Gold SQL: `SELECT COUNT(T3.raceId) FROM circuits AS T1 INNER JOIN races AS T3 ON T3.circuitID = T1.circuitId WHERE T1.country NOT IN ( 'Bahrain', 'China', 'Singapore', 'Japan', 'Korea', 'Turkey', 'UAE', 'Malaysi`
- Oracle post-fix count: 2 correct candidates available

## 5. Schema Linker Recall and Accuracy

Recall measures what fraction of the tables and columns required by the gold SQL were present in S1 (precise) and S2 (recall) schemas. Column recall only counts explicit `table.column` references in the gold SQL.

| Metric | S1 (Precise) | S2 (Recall) |
|--------|-------------|------------|
| Avg Table Recall | 95.0% | 95.0% |
| Avg Column Recall | 93.3% | 95.7% |
| Questions with complete coverage | 25/33 | 28/33 |

### Per-Question Schema Recall Detail

| # | DB | Q# | Diff | S1 Tables | S1 Cols | S2 Tables | S2 Cols | S1 Complete | S2 Complete | Missing S1 |
|---|----|----|------|-----------|---------|-----------|---------|-------------|-------------|-----------|
| 1 | california_schools | 64 | simple | 1/1 | 0/0 | 1/1 | 0/0 | Y | Y |  |
| 2 | california_schools | 23 | moderate | 2/2 | 3/4 | 2/2 | 4/4 | N | Y | t1.school |
| 3 | california_schools | 28 | challenging | 2/2 | 6/8 | 2/2 | 7/8 | N | N | t2.school, t2.doc |
| 4 | card_games | 463 | simple | 2/2 | 0/0 | 2/2 | 0/0 | Y | Y |  |
| 5 | card_games | 427 | moderate | 2/2 | 4/4 | 2/2 | 4/4 | Y | Y |  |
| 6 | card_games | 431 | challenging | 2/2 | 6/7 | 2/2 | 7/7 | N | Y | t1.name |
| 7 | codebase_community | 601 | simple | 2/2 | 4/4 | 2/2 | 4/4 | Y | Y |  |
| 8 | codebase_community | 571 | moderate | 2/2 | 4/4 | 2/2 | 4/4 | Y | Y |  |
| 9 | codebase_community | 586 | challenging | 3/3 | 7/7 | 3/3 | 7/7 | Y | Y |  |
| 10 | debit_card_specializing | 1519 | simple | 1/2 | 3/5 | 1/2 | 3/5 | N | N | gasstations, t2.gasstationid, t1.gasstationid |
| 11 | debit_card_specializing | 1474 | moderate | 2/2 | 5/5 | 2/2 | 5/5 | Y | Y |  |
| 12 | debit_card_specializing | 1526 | challenging | 2/3 | 3/5 | 2/3 | 3/5 | N | N | gasstations, t2.gasstationid, t1.gasstationid |
| 13 | european_football_2 | 1027 | simple | 2/2 | 4/4 | 2/2 | 4/4 | Y | Y |  |
| 14 | european_football_2 | 1025 | moderate | 2/2 | 6/6 | 2/2 | 6/6 | Y | Y |  |
| 15 | european_football_2 | 1031 | challenging | 2/2 | 4/4 | 2/2 | 4/4 | Y | Y |  |
| 16 | financial | 109 | simple | 2/2 | 5/5 | 2/2 | 5/5 | Y | Y |  |
| 17 | financial | 135 | moderate | 2/2 | 5/5 | 2/2 | 5/5 | Y | Y |  |
| 18 | financial | 149 | challenging | 3/3 | 6/6 | 3/3 | 6/6 | Y | Y |  |
| 19 | formula_1 | 953 | simple | 2/2 | 3/4 | 2/2 | 4/4 | N | Y | t1.laps |
| 20 | formula_1 | 852 | moderate | 2/2 | 5/5 | 2/2 | 5/5 | Y | Y |  |
| 21 | formula_1 | 994 | challenging | 3/3 | 9/9 | 3/3 | 9/9 | Y | Y |  |
| 22 | student_club | 1349 | simple | 2/2 | 4/4 | 2/2 | 4/4 | Y | Y |  |
| 23 | student_club | 1456 | moderate | 2/3 | 5/7 | 2/3 | 5/7 | N | N | budget, t2.budget_id, t2.spent |
| 24 | student_club | 1464 | challenging | 2/4 | 7/9 | 2/4 | 7/9 | N | N | event, attendance, t2.link_to_event, t1.event_id |
| 25 | superhero | 803 | simple | 1/1 | 0/0 | 1/1 | 0/0 | Y | Y |  |
| 26 | superhero | 782 | moderate | 2/2 | 5/5 | 2/2 | 5/5 | Y | Y |  |
| 27 | superhero | 773 | challenging | 2/2 | 7/7 | 2/2 | 7/7 | Y | Y |  |
| 28 | thrombosis_prediction | 1276 | simple | 2/2 | 4/4 | 2/2 | 4/4 | Y | Y |  |
| 29 | thrombosis_prediction | 1225 | moderate | 2/2 | 3/3 | 2/2 | 3/3 | Y | Y |  |
| 30 | thrombosis_prediction | 1295 | challenging | 3/3 | 3/3 | 3/3 | 3/3 | Y | Y |  |
| 31 | toxicology | 195 | simple | 1/1 | 1/1 | 1/1 | 1/1 | Y | Y |  |
| 32 | toxicology | 237 | moderate | 2/2 | 4/4 | 2/2 | 4/4 | Y | Y |  |
| 33 | toxicology | 304 | challenging | 2/2 | 4/4 | 2/2 | 4/4 | Y | Y |  |

### S1 Incomplete — Missing Required Schema Elements

**Q#23** (california_schools / moderate, correct=False)
- Required tables: ['frpm', 'schools']
- Missing tables in S1: []
- Required cols: ['t1.cdscode', 't1.school', 't1.street', 't2.cdscode']
- Missing cols in S1: ['t1.school']
- S1 fields count: 8
- S2 fields count: 15
- Gold SQL: `SELECT T1.School, T1.Street FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.`Enrollment (K-12)` - T2.`Enrollment (Ages 5-17)` > 30`

**Q#28** (california_schools / challenging, correct=False)
- Required tables: ['frpm', 'schools']
- Missing tables in S1: []
- Required cols: ['t1.cdscode', 't2.cdscode', 't2.doc', 't2.fundingtype', 't2.school', 't3.cdscode', 't4.cdscode', 't4.fundingtype']
- Missing cols in S1: ['t2.school', 't2.doc']
- S1 fields count: 8
- S2 fields count: 13
- Gold SQL: `SELECT T2.School, T2.DOC FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.FundingType = 'Locally funded' AND (T1.`Enrollment (K-12)` - T1.`Enrollment (Ages 5-17)`) > (SELEC`

**Q#431** (card_games / challenging, correct=False)
- Required tables: ['set_translations', 'sets']
- Missing tables in S1: []
- Required cols: ['t1.code', 't1.id', 't1.isfoilonly', 't1.isforeignonly', 't1.name', 't2.language', 't2.setcode']
- Missing cols in S1: ['t1.name']
- S1 fields count: 7
- S2 fields count: 17
- Gold SQL: `SELECT T1.name, T1.id FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T2.language = 'Japanese' AND T1.isFoilOnly = 1 AND T1.isForeignOnly = 0`

**Q#1519** (debit_card_specializing / simple, correct=True)
- Required tables: ['gasstations', 'transactions_1k']
- Missing tables in S1: ['gasstations']
- Required cols: ['t1.date', 't1.gasstationid', 't1.productid', 't1.time', 't2.gasstationid']
- Missing cols in S1: ['t2.gasstationid', 't1.gasstationid']
- S1 fields count: 4
- S2 fields count: 4
- Gold SQL: `SELECT T1.ProductID FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID WHERE T1.Date = '2012-08-23' AND T1.Time = '21:20:00'`

**Q#1526** (debit_card_specializing / challenging, correct=True)
- Required tables: ['gasstations', 'transactions_1k', 'yearmonth']
- Missing tables in S1: ['gasstations']
- Required cols: ['t1.customerid', 't1.date', 't1.gasstationid', 't1.price', 't2.gasstationid']
- Missing cols in S1: ['t2.gasstationid', 't1.gasstationid']
- S1 fields count: 7
- S2 fields count: 10
- Gold SQL: `SELECT CAST(SUM(IIF(SUBSTR(Date, 1, 4) = '2012', Consumption, 0)) - SUM(IIF(SUBSTR(Date, 1, 4) = '2013', Consumption, 0)) AS FLOAT) / SUM(IIF(SUBSTR(Date, 1, 4) = '2012', Consumption, 0)) FROM yearmon`

**Q#953** (formula_1 / simple, correct=False)
- Required tables: ['constructors', 'results']
- Missing tables in S1: []
- Required cols: ['t1.constructorid', 't1.laps', 't2.constructorid', 't2.nationality']
- Missing cols in S1: ['t1.laps']
- S1 fields count: 11
- S2 fields count: 12
- Gold SQL: `SELECT COUNT(DISTINCT T2.constructorId) FROM results AS T1 INNER JOIN constructors AS T2 on T1.constructorId = T2.constructorId WHERE T1.laps > 50 AND T2.nationality = 'French'`

**Q#1456** (student_club / moderate, correct=False)
- Required tables: ['budget', 'expense', 'member']
- Missing tables in S1: ['budget']
- Required cols: ['t1.link_to_budget', 't1.link_to_member', 't2.budget_id', 't2.spent', 't3.first_name', 't3.last_name', 't3.member_id']
- Missing cols in S1: ['t2.budget_id', 't2.spent']
- S1 fields count: 9
- S2 fields count: 9
- Gold SQL: `SELECT T3.first_name, T3.last_name FROM expense AS T1 INNER JOIN budget AS T2 ON T1.link_to_budget = T2.budget_id INNER JOIN member AS T3 ON T1.link_to_member = T3.member_id ORDER BY T2.spent DESC LIM`

**Q#1464** (student_club / challenging, correct=True)
- Required tables: ['attendance', 'event', 'income', 'member']
- Missing tables in S1: ['event', 'attendance']
- Required cols: ['t1.event_id', 't2.link_to_event', 't2.link_to_member', 't3.first_name', 't3.last_name', 't3.member_id', 't4.amount', 't4.date_received', 't4.link_to_member']
- Missing cols in S1: ['t2.link_to_event', 't1.event_id']
- S1 fields count: 9
- S2 fields count: 15
- Gold SQL: `SELECT DISTINCT T3.first_name, T3.last_name, T4.amount FROM event AS T1 INNER JOIN attendance AS T2 ON T1.event_id = T2.link_to_event INNER JOIN member AS T3 ON T3.member_id = T2.link_to_member INNER `


## 6. Per-Question Full Results Table

| # | DB | Q# | Difficulty | Grounding | Schema | Gen | Fix | Selection | OracleP | OracleF | SelMatch | Correct |
|---|----|----|------------|-----------|--------|-----|-----|-----------|---------|---------|----------|---------|
| 1 | california_schools | 64 | simple | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 2 | california_schools | 23 | moderate | OK | OK | OK | OK | OK (tournament) | Y | Y | N | NO |
| 3 | california_schools | 28 | challenging | OK | OK | OK | OK | OK (tournament) | N | N | N | NO |
| 4 | card_games | 463 | simple | OK | OK | OK | OK | OK (fast_path) | N | N | N | NO |
| 5 | card_games | 427 | moderate | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 6 | card_games | 431 | challenging | OK | OK | OK | OK | OK (tournament) | N | N | N | NO |
| 7 | codebase_community | 601 | simple | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 8 | codebase_community | 571 | moderate | OK | OK | OK | OK | OK (fast_path) | N | N | N | NO |
| 9 | codebase_community | 586 | challenging | OK | OK | OK | OK | OK (fast_path) | N | N | N | NO |
| 10 | debit_card_specializing | 1519 | simple | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 11 | debit_card_specializing | 1474 | moderate | OK | OK | OK | OK | OK (tournament) | Y | Y | Y | YES |
| 12 | debit_card_specializing | 1526 | challenging | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 13 | european_football_2 | 1027 | simple | OK | OK | OK | OK | OK (fast_path) | N | N | N | NO |
| 14 | european_football_2 | 1025 | moderate | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 15 | european_football_2 | 1031 | challenging | OK | OK | OK | OK | OK (tournament) | N | N | N | NO |
| 16 | financial | 109 | simple | OK | OK | OK | OK | OK (tournament) | Y | Y | N | NO |
| 17 | financial | 135 | moderate | OK | OK | OK | OK | OK (fast_path) | N | N | N | NO |
| 18 | financial | 149 | challenging | OK | OK | OK | OK | OK (fast_path) | N | N | N | NO |
| 19 | formula_1 | 953 | simple | OK | OK | OK | OK | OK (tournament) | Y | Y | N | NO |
| 20 | formula_1 | 852 | moderate | OK | OK | OK | OK | OK (tournament) | Y | Y | N | NO |
| 21 | formula_1 | 994 | challenging | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 22 | student_club | 1349 | simple | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 23 | student_club | 1456 | moderate | OK | OK | OK | OK | OK (tournament) | N | N | N | NO |
| 24 | student_club | 1464 | challenging | OK | OK | OK | OK | OK (tournament) | Y | Y | Y | YES |
| 25 | superhero | 803 | simple | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 26 | superhero | 782 | moderate | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 27 | superhero | 773 | challenging | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 28 | thrombosis_prediction | 1276 | simple | OK | OK | OK | OK | OK (tournament) | Y | Y | Y | YES |
| 29 | thrombosis_prediction | 1225 | moderate | OK | OK | OK | OK | OK (fast_path) | N | N | N | NO |
| 30 | thrombosis_prediction | 1295 | challenging | OK | OK | OK | OK | OK (fast_path) | N | N | N | NO |
| 31 | toxicology | 195 | simple | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 32 | toxicology | 237 | moderate | OK | OK | OK | OK | OK (fast_path) | Y | Y | Y | YES |
| 33 | toxicology | 304 | challenging | OK | OK | OK | OK | OK (tournament) | Y | Y | Y | YES |

## 7. Stage Failure Analysis (Wrong Answers)

Total wrong answers: 16/33

### Error Stage Distribution

| Stage | Count |
|-------|-------|

### Wrong Answer Details

**Q#23** (california_schools / moderate)
- Question: List the names of schools with more than 30 difference in enrollements between K-12 and ages 5-17? Please also give the full street adress of the schools.
- Error stage: None
- Oracle pre-fix: True (3 correct candidates)
- Oracle post-fix: True (3 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT f."School Name", s.Street FROM frpm AS f JOIN schools AS s ON f.CDSCode = s.CDSCode WHERE ( f."Enrollment (K-12)" - f."Enrollment (Ages 5-17)" ) > 30`
- Gold SQL: `SELECT T1.School, T1.Street FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.`Enrollment (K-12)` - T2.`Enrollment (Ages 5-17)` > 30`
- S1 schema missing: tables=[], cols=['t1.school']

**Q#28** (california_schools / challenging)
- Question: Consider the average difference between K-12 enrollment and 15-17 enrollment of schools that are locally funded, list the names and DOC type of schools which has a difference above this average.
- Error stage: None
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT T1.School, T1.DOCType FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.FundingType = 'Locally funded' AND (T2."Enrollment (K-12)" - T2."Enrollment (Ages 5-17)") > ( `
- Gold SQL: `SELECT T2.School, T2.DOC FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.FundingType = 'Locally funded' AND (T1.`Enrollment (K-12)` - T1.`Enrollment (Ages 5-17)`) > (SELEC`
- S1 schema missing: tables=[], cols=['t2.school', 't2.doc']

**Q#463** (card_games / simple)
- Question: How many translations are there for the set of cards with "Angel of Mercy" in it?
- Error stage: None
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT COUNT(T1.id) FROM set_translations AS T1 INNER JOIN cards AS T2 ON T1.setCode = T2.setCode WHERE T2.name = 'Angel of Mercy'`
- Gold SQL: `SELECT COUNT(DISTINCT translation) FROM set_translations WHERE setCode IN ( SELECT setCode FROM cards WHERE name = 'Angel of Mercy' ) AND translation IS NOT NULL`

**Q#431** (card_games / challenging)
- Question: Which set is not available outside of the United States and has foil cards with Japanese writing on them? Please include the set ID in your response.
- Error stage: None
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT DISTINCT T1.id FROM sets AS T1 INNER JOIN cards AS T2 ON T1.code = T2.setCode INNER JOIN foreign_data AS T3 ON T2.uuid = T3.uuid WHERE T1.isForeignOnly = 0 AND T1.isFoilOnly = 1 AND T3.language`
- Gold SQL: `SELECT T1.name, T1.id FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T2.language = 'Japanese' AND T1.isFoilOnly = 1 AND T1.isForeignOnly = 0`
- S1 schema missing: tables=[], cols=['t1.name']

**Q#571** (codebase_community / moderate)
- Question: For the user No.24, how many times is the number of his/her posts compared to his/her votes?
- Error stage: None
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT CAST( ( SELECT COUNT(Id) FROM posts WHERE OwnerUserId = 24 ) AS REAL ) / ( SELECT COUNT(Id) FROM votes WHERE UserId = 24 )`
- Gold SQL: `SELECT CAST(COUNT(T2.Id) AS REAL) / COUNT(DISTINCT T1.Id) FROM votes AS T1 INNER JOIN posts AS T2 ON T1.UserId = T2.OwnerUserId WHERE T1.UserId = 24`

**Q#586** (codebase_community / challenging)
- Question: Which user added a bounty amount of 50 to the post title mentioning variance?
- Error stage: None
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT T3.DisplayName FROM votes AS T1 INNER JOIN posts AS T2 ON T1.PostId = T2.Id INNER JOIN users AS T3 ON T1.UserId = T3.Id WHERE T1.BountyAmount = 50 AND T2.Title LIKE '%variance%'`
- Gold SQL: `SELECT T3.DisplayName, T1.Title FROM posts AS T1 INNER JOIN votes AS T2 ON T1.Id = T2.PostId INNER JOIN users AS T3 ON T3.Id = T2.UserId WHERE T2.BountyAmount = 50 AND T1.Title LIKE '%variance%'`

**Q#1027** (european_football_2 / simple)
- Question: Indicate the full names of the top 10 players with the highest number of penalties.
- Error stage: None
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT P.player_name FROM Player AS P JOIN Player_Attributes AS PA ON P.player_api_id = PA.player_api_id ORDER BY PA.penalties DESC LIMIT 10`
- Gold SQL: `SELECT t2.player_name FROM Player_Attributes AS t1 INNER JOIN Player AS t2 ON t1.id = t2.id ORDER BY t1.penalties DESC LIMIT 10`

**Q#1031** (european_football_2 / challenging)
- Question: At present, calculate for the player's age who have a sprint speed of no less than 97 between 2013 to 2015.
- Error stage: None
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT DISTINCT CAST((julianday('now') - julianday(T1.birthday)) / 365.25 AS INTEGER) AS player_age FROM Player AS T1 INNER JOIN Player_Attributes AS T2 ON T1.player_api_id = T2.player_api_id WHERE T2`
- Gold SQL: `SELECT DISTINCT DATETIME() - T2.birthday age FROM Player_Attributes AS t1 INNER JOIN Player AS t2 ON t1.player_api_id = t2.player_api_id WHERE STRFTIME('%Y',t1.`date`) >= '2013' AND STRFTIME('%Y',t1.``

**Q#109** (financial / simple)
- Question: How many clients opened their accounts in Jesenik branch were women?
- Error stage: None
- Oracle pre-fix: True (5 correct candidates)
- Oracle post-fix: True (5 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT COUNT(DISTINCT T3.client_id) FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id INNER JOIN disp AS T4 ON T2.account_id = T4.account_id INNER JOIN client AS T3 ON T4`
- Gold SQL: `SELECT COUNT(T1.client_id) FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.gender = 'F' AND T2.A2 = 'Jesenik'`

**Q#135** (financial / moderate)
- Question: After making a credit card withdrawal, how many account/s with monthly issuance has a negative balance?
- Error stage: None
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT COUNT(DISTINCT T1.account_id) FROM account AS T1 INNER JOIN trans AS T2 ON T1.account_id = T2.account_id WHERE T2.operation = 'VYBER KARTOU' AND T1.frequency = 'POPLATEK MESICNE' AND T2.balance`
- Gold SQL: `SELECT COUNT(T1.account_id) FROM trans AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id WHERE T1.balance < 0 AND T1.operation = 'VYBER KARTOU' AND T2.frequency = 'POPLATEK MESICNE'`

**Q#149** (financial / challenging)
- Question: Please list the account types that are not eligible for loans, and the average income of residents in the district where the account is located exceeds $8000 but is no more than $9000.
- Error stage: None
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT DISTINCT T1.type FROM disp AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id INNER JOIN district AS T3 ON T2.district_id = T3.district_id WHERE T1.type <> 'OWNER' AND T3.A11 > 800`
- Gold SQL: `SELECT T3.type FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id INNER JOIN disp AS T3 ON T2.account_id = T3.account_id WHERE T3.type != 'OWNER' AND T1.A11 BETWEEN 8000 A`

**Q#953** (formula_1 / simple)
- Question: How many French constructors have a lap number of over 50?
- Error stage: None
- Oracle pre-fix: True (3 correct candidates)
- Oracle post-fix: True (3 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT COUNT(DISTINCT T1.constructorId) FROM constructors AS T1 INNER JOIN results AS T2 ON T1.constructorId = T2.constructorId INNER JOIN lapTimes AS T3 ON T2.raceId = T3.raceId WHERE T1.nationality `
- Gold SQL: `SELECT COUNT(DISTINCT T2.constructorId) FROM results AS T1 INNER JOIN constructors AS T2 on T1.constructorId = T2.constructorId WHERE T1.laps > 50 AND T2.nationality = 'French'`
- S1 schema missing: tables=[], cols=['t1.laps']

**Q#852** (formula_1 / moderate)
- Question: How many races in the year 2010 are held on grand prixs outside Asia and Europe?
- Error stage: None
- Oracle pre-fix: True (2 correct candidates)
- Oracle post-fix: True (2 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT COUNT(T1.raceId) FROM races AS T1 INNER JOIN circuits AS T2 ON T1.circuitId = T2.circuitId WHERE T1.year = 2010 AND T2.country NOT IN ('Asia', 'Europe')`
- Gold SQL: `SELECT COUNT(T3.raceId) FROM circuits AS T1 INNER JOIN races AS T3 ON T3.circuitID = T1.circuitId WHERE T1.country NOT IN ( 'Bahrain', 'China', 'Singapore', 'Japan', 'Korea', 'Turkey', 'UAE', 'Malaysi`

**Q#1456** (student_club / moderate)
- Question: List the full name of the top five members who spend the most money in the descending order of spending.
- Error stage: None
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT T1.first_name, T1.last_name FROM member AS T1 INNER JOIN expense AS T2 ON T1.member_id = T2.link_to_member GROUP BY T1.member_id ORDER BY SUM(T2.cost) DESC LIMIT 5`
- Gold SQL: `SELECT T3.first_name, T3.last_name FROM expense AS T1 INNER JOIN budget AS T2 ON T1.link_to_budget = T2.budget_id INNER JOIN member AS T3 ON T1.link_to_member = T3.member_id ORDER BY T2.spent DESC LIM`
- S1 schema missing: tables=['budget'], cols=['t2.budget_id', 't2.spent']

**Q#1225** (thrombosis_prediction / moderate)
- Question: List and group all patients by sex for total bilirubin (T-BIL) level not within the normal range.
- Error stage: None
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT P.SEX, GROUP_CONCAT(DISTINCT P.ID) FROM Patient AS P JOIN Laboratory AS L ON P.ID = L.ID WHERE L."T-BIL" >= 2.0 GROUP BY P.SEX`
- Gold SQL: `SELECT T1.ID,T1.SEX FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.`T-BIL` >= 2.0 GROUP BY T1.SEX,T1.ID`

**Q#1295** (thrombosis_prediction / challenging)
- Question: Among the patients whose total bilirubin is over the normal range, how many of them have a peripheral pattern observed in the sheet of ANA examination?
- Error stage: None
- Oracle pre-fix: False (0 correct candidates)
- Oracle post-fix: False (0 correct candidates)
- Selector matched oracle: False
- Final SQL: `SELECT COUNT(DISTINCT T1.ID) FROM Laboratory AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID WHERE T1."T-BIL" >= 2.0 AND T2."ANA Pattern" LIKE '%P%'`
- Gold SQL: `SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID INNER JOIN Examination AS T3 ON T1.ID = T3.ID WHERE T2.`T-BIL` >= 2 AND T3.`ANA Pattern` LIKE '%P%'`

## 8. Interface Issues Observed

- Grounding stage errors: 0/33
- Schema linking stage errors: 0/33
- Generation stage errors (partial): 0/33
- Fixing stage errors: 0/33
- Selection stage errors: 0/33

### Stage Statistics

- Avg cell matches per question: 2.9
- Avg few-shot examples per question: 8.0
- Avg S1 fields: 7.3
- Avg S2 fields: 11.2
- Avg total generation candidates: 11.0

## 9. Issues Found

### Priority Legend: P0 = must fix, P1 = important, P2 = nice to have

**P1: Selector missing 4 achievable correct answers**
- Description: Oracle post-fix is 21/33 but actual accuracy is 17/33. 4 questions had a correct candidate but the selector chose wrong.
- Suggested fix: Review tournament prompt quality and cluster comparison logic.

**P1: Imperfect S1 table recall: 95.0%**
- Description: S1 occasionally misses required tables.
- Suggested fix: Review edge cases in schema linking (e.g. aliases, junction tables).

## 10. Suggested Improvements Before Pipeline Wiring (Prompt 13)

- Overall accuracy (51.5%) is below the 68% target. Review wrong answers to identify patterns before wiring Prompt 13.
- Query fixer success rate is low (30.4%). 23 candidates needed fixing but only 7 were fixed. Review fix prompt construction — it may need more context about the error.
- Run the full online_pipeline.py integration (Prompt 13) once the component interfaces are confirmed correct from this checkpoint.
- Consider adding a configurable semaphore (e.g. max_concurrent_questions=3) in the online pipeline to prevent API rate limit errors during batch evaluation.
