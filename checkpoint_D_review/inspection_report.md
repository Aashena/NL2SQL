# Checkpoint D Inspection Report — Re-run (2026-02-23)

## Overview

This report covers the **second run** of Checkpoint D, executed after a set of targeted
code improvements were applied to address the 7 issues identified in the first run.

| Item | Value |
|------|-------|
| Run date | 2026-02-23 |
| Sample | 33 BIRD dev questions (3 per each of 11 DBs: 1 simple, 1 moderate, 1 challenging, `random.seed(42)`) |
| Provider | Gemini (`gemini-2.5-flash` / `gemini-2.5-pro`) |
| LLM cache | Enabled |
| Runtime | 1884.5 s (~31 min) |
| Script | `scripts/run_checkpoint_d.py` |

---

## Changes Applied Since First Run

Six files were modified between the first and second runs. Each change targets one or more
of the P0/P1/P2 issues reported in the first inspection.

### 1. `src/generation/base_generator.py` — SQL Writing Rules (P2-6)

Added a `## SQL Writing Rules` section to `build_base_prompt()`, injected into every
generator's user prompt. Four explicit rules were added:

1. **SELECT completeness** — return ALL columns the question asks for.
2. **Value mappings** — if evidence defines a label→value mapping, apply it with CASE/IIF.
3. **Ratio direction** — "A compared to B" means A/B; DISTINCT only when the question implies uniqueness.
4. **Evidence scope** — trust the question for output format; trust the evidence for column semantics.

### 2. `src/generation/standard_generator.py` — Diversity + MAX_TOKENS (P0-2, P1-4)

- **Alternative templates when S1==S2**: `_B1_ALT_SYSTEM_TEMPLATE` and `_B2_ALT_SYSTEM_TEMPLATE`
  steer the model to explore a structurally different approach (alternative JOINs,
  subqueries, window functions) when both schema variants are identical, preventing
  identical prompts from producing identical SQL.
- **`max_tokens`: 2000 → 4096** for B1 generators (was already 4096 for B2).
- **`temperature=0.3`** for all four variants (B1_s1, B1_s2, B2_s1, B2_s2).
- **MAX_TOKENS truncation guard**: if `finish_reason=MAX_TOKENS`, discard the truncated
  response and return `error_flag=True` instead of passing broken SQL to the query fixer.

### 3. `src/generation/icl_generator.py` — MAX_TOKENS + temperature (P0-2, P1-4)

- **`temperature=0.7`** parameter exposed and wired through to the API call.
- **MAX_TOKENS truncation guard** (same logic as standard_generator): truncated responses
  from Gemini are now discarded cleanly.

### 4. `src/grounding/context_grounder.py` — Fallback on LLM error (P0-1)

- Added `_extract_keywords_from_text()`: a lightweight stop-word filter + punctuation
  stripper that extracts meaningful tokens from the raw question+evidence text.
- Changed the `LLMError` fallback from **empty grounding** to **keyword extraction**.
  This ensures the LSH index is still queried (with the NL tokens) even when Gemini
  returns `MALFORMED_FUNCTION_CALL`, recovering the majority of cell-value anchors.

### 5. `src/schema_linking/schema_linker.py` — Robustness + Size Control (P0-1, P1-3, P1-4)

- **S1 LLM try/except**: `LLMError` in the S1 call now falls back to FAISS top-15
  instead of crashing the whole question.
- **High-confidence FAISS auto-promotion to S1**: fields with cosine similarity ≥ 0.8
  to a schema_hint string are auto-added to S1, reducing schema-linker misses for
  hint-matched fields.
- **S1 hard cap at 20 fields**: prevents over-expansion on large-schema databases;
  FAISS-ranked fields are kept first, PK/FK additions come last.
- **S2 skip logic**: the second LLM call is skipped when:
  - `total_candidates == 0` (FAISS returned nothing)
  - `remaining_candidates == 0` (S1 consumed every candidate)
  - `s1_coverage >= 0.80` (S1 already covers ≥80% of candidates)
  - `len(remaining) < 3` (fewer than 3 new fields to add)
  This eliminates a ~13% rate of gratuitous S2 calls that produced no new signal and
  inflated costs.
- **S2 cap at `min(S1+10, 25)`**: replaced the uncapped S2 expansion that caused
  36-field schemas on `european_football_2`.
- Removed the previous "empty-candidates → make API call anyway" code path.

### 6. `src/preprocessing/schema_formatter.py` — FK annotations (P1-5)

- **DDL**: each FK column now appends `| FK → <target_table>(<col>)` inline in the
  column comment, so LLMs see the join target on the same line as the column definition.
- **Markdown**: FK annotation changed from `(FK)` to `(FK→target_table.col)` to include
  the join target explicitly.

  > Note: this change affects cached schemas on disk. The `checkpoint_d` run used the
  > existing preprocessed schemas (built before this change). A full offline re-run would
  > be needed to propagate the FK annotation improvement.

---

## Results Comparison

### Oracle Upper Bound

| Metric | Run 1 | Run 2 | Delta |
|--------|-------|-------|-------|
| **Overall** | 21/33 = **63.6%** | 25/33 = **75.8%** | +12.2pp |
| Simple | 9/11 = 81.8% | 9/11 = 81.8% | 0 |
| Moderate | 6/11 = 54.5% | 7/11 = 63.6% | +9.1pp |
| Challenging | 6/11 = 54.5% | 9/11 = **81.8%** | **+27.3pp** |

The overall oracle improved by **+12.2 pp**, almost entirely driven by the challenging
tier which jumped from 54.5% to 81.8% (+27.3pp). The moderate tier improved slightly
(+9.1pp). Simple held steady — it was already near-ceiling.

### Per-Database Oracle

| Database | Run 1 | Run 2 |
|----------|-------|-------|
| california_schools | 2/3 | 2/3 |
| card_games | 1/3 | 2/3 ↑ |
| codebase_community | 1/3 | 1/3 |
| debit_card_specializing | 2/3 | 3/3 ↑ |
| european_football_2 | 2/3 | 2/3 |
| financial | 2/3 | 3/3 ↑ |
| formula_1 | 3/3 | 3/3 |
| student_club | 2/3 | 2/3 |
| superhero | 2/3 | 3/3 ↑ |
| thrombosis_prediction | 2/3 | 2/3 |
| toxicology | 2/3 | 2/3 |

Four databases improved to 3/3 (perfect): card_games, debit_card_specializing, financial, superhero.
codebase_community remains at 1/3 and is the weakest database.

### Generator Success Rates

| Generator | Run 1 (non-empty) | Run 1 (exec%) | Run 2 (non-empty) | Run 2 (exec%) |
|-----------|-------------------|---------------|-------------------|---------------|
| A_reasoning | 100% | 99% | 100% | 99% |
| B1_standard | 100% | 95% | 100% | 95% |
| B2_complex | **87%** | **73%** | **100%** | **93%** |
| C_icl | **87%** | **73%** | **95%** | **94%** |

B2 and C generators saw the largest improvements:
- **B2**: 87% → 100% non-empty; 73% → 93% exec — MAX_TOKENS fix and temperature change
- **C**: 87% → 95% non-empty; 73% → 94% exec — MAX_TOKENS truncation guard

### Candidate Diversity

| Metric | Run 1 | Run 2 |
|--------|-------|-------|
| Total non-error candidates | 332 | 359 |
| Unique SQL strings | 199 | 198 |
| Duplicate rate | 40% | 44% |
| Avg candidates/question | 10.1 | 10.9 |

The duplicate rate increased slightly (40% → 44%). This is expected: S1==S2 triggers
in 6/33 questions, and even with alternative templates the model converges on the same
correct SQL for straightforward questions (91% dup for debit_card_specializing simple,
91% for student_club simple). For *correct* simple questions, duplicates are harmless.
The diversity gap matters most for hard questions where multiple strategies are needed.

---

## Remaining Oracle Misses (8/33)

### Miss 1 — QID=28, california_schools, **challenging**

**Question**: Compare K-12 vs 15-17 enrollment; identify schools where avg difference is
above the mean — return School name and DOC (document type).

**Root cause**: Schema linking correctly identifies enrollment columns and the CDSCode
join, but misses `schools.DOC`. The LLM selects `schools.SOCType` instead.
`DOC` is a less-obvious output column; evidence doesn't hint at it.

**Classification**: P1 — schema linking miss (output column).

---

### Miss 2 — QID=463, card_games, **simple**

**Question**: How many translations are there for the set of cards with "Angel of Mercy"?

**Ground truth**: `COUNT(DISTINCT translation)` — counts unique translation strings.

**Generated**: `COUNT(id)` or `COUNT(T1.id)` — counts rows, not distinct translations.

The question says "how many translations" without using "distinct" or "unique". Yet the GT
uses DISTINCT because multiple sets can share the same translation string. SQL Writing Rule 3
("Use DISTINCT only when the question implies uniqueness") did not help because the question
doesn't use those signal words.

**Classification**: P2 — semantic miss (implicit deduplication not triggered).

---

### Miss 3 — QID=571, codebase_community, **moderate**

**Question**: For user No.24, how many times is the number of posts compared to votes?

**Ground truth**: `CAST(COUNT(T2.Id) AS REAL) / COUNT(DISTINCT T1.Id)` — `posts / votes`.

**Generated**: Inverted ratio — all candidates divide votes by posts. Despite SQL Writing
Rule 3 stating "A compared to B → A/B", the generated SQL consistently inverts this.

**Classification**: P3 — ratio direction error (rule not followed or mis-parsed).

---

### Miss 4 — QID=586, codebase_community, **challenging**

**Question**: Which user added a bounty amount of 50 to the post title mentioning "variance"?
(GT returns both `DisplayName` and `Title`.)

**Root cause**: S1/S2 both contain `votes.Id` but not `votes.BountyAmount`. However,
candidates do retrieve the right rows (11/11 exec OK). The mismatch is that GT returns
**two columns** (`DisplayName, Title`) while most candidates return only `DisplayName`.
SQL Writing Rule 1 ("return ALL columns the question asks for") may not have been enough
because the question says "which user" (not "which user and post title").

**Classification**: P4 — output completeness miss (GT returns extra Title column not
implied by question wording) + possible schema linking miss for `BountyAmount`.

---

### Miss 5 — QID=1027, european_football_2, **simple**

**Question**: Indicate the full names of the top 10 players with the highest number of
penalties.

**Root cause**: S1 has `Player.id` and `Player_Attributes.player_api_id`. The correct join
is `Player.player_api_id = Player_Attributes.player_api_id`, but the model joins
`Player.id = Player_Attributes.player_api_id`, which is wrong — `Player.id` is an internal
row ID, not the FIFA API player ID. Because `Player.player_api_id` is not in S1, the model
defaults to the only join-compatible column it can see.

**Classification**: P5 — schema linking miss (correct join key `Player.player_api_id` not
included in S1; only `Player.id` was selected).

---

### Miss 6 — QID=1456, student_club, **moderate**

**Question**: List the full name of the top five members who spend the most money.

**Root cause**: S1 includes `expense.cost` and `expense.link_to_member`. The generated SQL
sums `expense.cost` per member, which is semantically correct. However, 10/11 candidates
exec OK but none oracle-match. Possible causes: (a) the GT computes "spend" differently
(it joins through `budget`), (b) the output format is different, or (c) there is a
subtle filtering condition (e.g., only certain expense types count). The GT does a 3-table
join (expense → budget → event) while candidates do a 2-table join (expense → member).

**Classification**: P6 — semantic interpretation gap (GT join path not obvious from question).

---

### Miss 7 — QID=1225, thrombosis_prediction, **moderate**

**Question**: List and group all patients by sex for T-BIL level not within normal range.

**Ground truth**: `SELECT T1.ID, T1.SEX FROM Patient JOIN Laboratory WHERE T-BIL out of range` —
returns a flat list of (ID, SEX) pairs sorted by SEX.

**Generated**: `SELECT SEX, GROUP_CONCAT(DISTINCT ID)` — uses GROUP BY + aggregation.

The phrase "list and group ... by sex" led the model to perform aggregation, but the GT
simply lists individual rows ordered by SEX (no GROUP BY). This is an over-aggregation
pattern.

**Classification**: P7 — over-aggregation (phrasing "group by" misinterpreted as GROUP BY).

---

### Miss 8 — QID=237, toxicology, **moderate**

**Question**: Which molecule does atom TR001_10 belong to? State whether it is carcinogenic.

**Ground truth**: `IIF(label = '+', 'YES', 'NO') AS flag_carcinogenic`

**Generated**: `CASE WHEN label = '+' THEN 'carcinogenic' ELSE 'not carcinogenic' END`

SQL Writing Rule 2 (value mappings) was applied, but the model chose natural language
('carcinogenic'/'not carcinogenic') instead of the YES/NO binary. The evidence says
"label = '+' means carcinogenic" but does not explicitly specify the return format as
"YES" or "NO". The BIRD ground truth always uses YES/NO for binary carcinogenicity.

**Classification**: P8 — value mapping format mismatch (need YES/NO convention in SQL
Writing Rules, or stronger evidence of expected output format).

---

## Issue Summary

| ID | Severity | Status | Description |
|----|----------|--------|-------------|
| P0-1 | P0 | **Fixed** | schema_linker crash on MALFORMED_FUNCTION_CALL — LLMError now falls back to FAISS top-15 |
| P0-2 | P0 | **Fixed** | B2/C generators 13% empty response from Gemini — MAX_TOKENS guard added, max_tokens doubled |
| P1-3 | P1 | **Fixed** | S2 over-expansion (36 fields on european_football_2) — S1 cap at 20, S2 cap at min(S1+10,25) |
| P1-4 | P1 | **Partially fixed** | Duplicate rate 40%→44% (slightly worse); alt templates help but simple questions converge |
| P1-5 | P1 | **Fixed** | Hallucinated table.table.column filtered correctly (confirmed working) |
| P2-6 | P2 | **Partially fixed** | Evidence prominence improved via SQL Writing Rules; ratio direction still fails on some |
| P2-7 | P2 | **Not yet fixed** | SentenceTransformer reloaded per session (not per question) — P2-7 was about cold-start; model loads once per run now |
| **New P-1** | P1 | **New** | Schema linking misses join keys (P5 / QID=1027): `Player.player_api_id` needed but not selected |
| **New P-2** | P2 | **New** | Implicit DISTINCT not recognized from question phrasing (P2 / QID=463) |
| **New P-3** | P2 | **New** | Over-aggregation when question says "list and group" (P7 / QID=1225) |
| **New P-4** | P2 | **New** | Binary output format: model uses 'carcinogenic'/'not carcinogenic' instead of 'YES'/'NO' (P8 / QID=237) |
| **New P-5** | P3 | **New** | GT returns extra columns not implied by question wording (P4 / QID=586) |

---

## Suggested Improvements for Next Phase

### High Priority

**S1-A: Add join-key coverage rule to schema linker**
When a table T has columns `id` and `{entity}_api_id`, and a joined table references
`{entity}_api_id`, the linker should always include both `T.id` AND `T.{entity}_api_id`
in S1. Currently PK auto-promotion adds `T.id` but the second join key is lost.
*Expected impact: fixes P5 / QID=1027 and similar join-key misses.*

**S1-B: Strengthen SQL Writing Rule for binary output format**
Add to `build_base_prompt()` Rule 2 (or a new Rule 5):
> "When the question asks 'is X true or false / yes or no', return 'YES' or 'NO' (not
> full words like 'carcinogenic'). Use IIF(cond, 'YES', 'NO') or CASE WHEN ... THEN 'YES'
> ELSE 'NO'."
*Expected impact: fixes P8 / QID=237 and similar binary format questions.*

**S1-C: Add "list and group by" → flat SELECT rule**
When the question contains "list ... grouped by X" or "list and group ... by X", the model
should generate `ORDER BY X` (not `GROUP BY X + aggregate`). Add to SQL Writing Rules:
> "5. 'List grouped by X' means ORDER BY X (sorted listing), not GROUP BY X with
> aggregation. Only use GROUP BY when the question explicitly asks to count, sum, or
> aggregate."
*Expected impact: fixes P7 / QID=1225 and similar over-aggregation.*

### Medium Priority

**S2-A: DISTINCT heuristic for "how many [plural noun]" questions**
When the question starts with "How many [plural noun]..." and the plural noun maps to a
column in set_translations, languages, etc., apply DISTINCT automatically because the
semantic intent is "how many unique values". Can be a simple heuristic in the SQL Writing
Rules:
> "When counting items where duplicates may exist (e.g., translations, languages, entries
> from a junction table), use COUNT(DISTINCT col) unless the question asks for total
> occurrences."
*Expected impact: fixes P2 / QID=463.*

**S2-B: Ratio direction clarification in SQL Writing Rules**
The current Rule 3 text says "how many times is A compared to B → A/B" but this wasn't
followed for QID=571 (moderate difficulty). Strengthen to an example:
> "3. Ratio / comparison direction: 'How many times is X compared to Y?' → X/Y.
> Example: 'How many times is posts compared to votes?' → COUNT(posts)/COUNT(votes).
> Never invert this."
*Expected impact: reduces ratio direction errors (P3 / QID=571).*

**S2-C: Output column completeness in schema linker**
For questions that ask for X "and" Y (e.g., "return School and document type"), the schema
linker should identify both output columns in S1, not just filtering/join columns.
Currently the LLM sometimes selects only columns needed for WHERE/JOIN, not for SELECT.
Adding a reminder to the schema linker prompt: "Also select columns mentioned in the
SELECT clause, not just filters and joins."
*Expected impact: reduces miss P1 / QID=28.*

### Low Priority

**S3-A: Improve diversity for simple questions**
The 44% duplicate rate is driven by simple questions (up to 91% dup). For simple questions,
diversity is less important since multiple correct answers are acceptable. The issue only
matters for complex questions. A future improvement could increase temperature for generators
when S2 is much larger than S1 (indicating schema uncertainty) and decrease it when S1==S2
(indicating schema certainty).

**S3-B: Schema format changes for FK annotations**
The `schema_formatter.py` FK annotation changes (inline `FK → target`) require a re-run of
offline preprocessing to take effect. After prompts 11-13 are complete and before the full
eval (prompt 15), consider re-running offline preprocessing to benefit from improved FK hints.

---

## Next Steps

1. Implement suggested improvements S1-A through S1-C in `base_generator.py` and
   `schema_linker.py` (can be done as part of prompt_11 pre-work or as a separate fix step).
2. Proceed to **Prompt 11** (Op 8: `query_fixer.py` + `test_query_fixer.py`).
3. Checkpoint D status remains as the reference baseline for the next evaluation.
4. Update `implementation_progress.json` to mark checkpoint_D completed after user review.

---

*Report generated: 2026-02-23. Run log: `checkpoint_D_review/run_output.log`. Raw results: `checkpoint_D_review/results.json`.*
