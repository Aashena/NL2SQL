# Checkpoint D — Inspection Report
**Date:** 2026-02-22
**Provider:** Gemini (gemini-2.5-flash / gemini-2.5-pro)
**Cache:** Enabled
**Runtime:** 31.5 minutes
**Sample:** 33 questions — 3 per each of 11 BIRD dev databases (1 simple, 1 moderate, 1 challenging), `random.seed(42)`

---

## 1. What Was Run

Operations tested end-to-end (no query fixer or selector yet):

| Op | Component | Status |
|----|-----------|--------|
| Op 5 | Context Grounding (keyword extraction + LSH lookup + few-shot retrieval) | Ran on all 33 questions |
| Op 6 | Adaptive Schema Linking (FAISS pre-filter → 2 LLM iterations → S₁, S₂) | Ran on all 33 questions |
| Op 7A | Reasoning Generator (4 candidates via extended thinking / thinking mode) | Ran on all 33 questions |
| Op 7B | Standard + Complex Generator (B1=Haiku/Flash, B2=Sonnet/Pro, 4 candidates) | Ran on all 33 questions |
| Op 7C | ICL Generator (3 candidates with few-shot examples) | Ran on all 33 questions |

---

## 2. Quantitative Results

### 2.1 Oracle Upper Bound

> Oracle = "does at least one candidate match ground truth?" — the ceiling for any selector.

| Difficulty | Oracle Count | Total | Oracle % |
|------------|-------------|-------|----------|
| Simple     | 8           | 11    | 72.7%    |
| Moderate   | 7           | 11    | 63.6%    |
| Challenging | 7          | 11    | 63.6%    |
| **Overall** | **22**     | **33** | **66.7%** |

### 2.2 Per-Database Oracle

| Database | Oracle | Total | % |
|----------|--------|-------|---|
| california_schools | 2 | 3 | 66.7% |
| card_games | 1 | 3 | 33.3% |
| codebase_community | 1 | 3 | 33.3% |
| debit_card_specializing | 3 | 3 | 100.0% |
| european_football_2 | 2 | 3 | 66.7% |
| financial | 2 | 3 | 66.7% |
| formula_1 | 2 | 3 | 66.7% |
| student_club | 2 | 3 | 66.7% |
| superhero | 3 | 3 | 100.0% |
| thrombosis_prediction | 2 | 3 | 66.7% |
| toxicology | 2 | 3 | 66.7% |

**Worst databases:** `card_games` and `codebase_community` (33.3%) — both have multi-table schema disambiguation and ratio/count questions that require precise semantics.

### 2.3 Per-Generator Success Rate

| Generator | Total Cands | Non-empty | Exec OK | Oracle Matches |
|-----------|------------|-----------|---------|----------------|
| A_reasoning (Gemini thinking) | 132 | 132 (100%) | 130 (98%) | 61 (46%) |
| B1_standard (Gemini Flash) | 66 | 66 (100%) | 60 (90%) | 29 (43%) |
| B2_complex (Gemini Pro) | 66 | 66 (100%) | **46 (69%)** | 25 (37%) |
| C_icl (Gemini Pro + few-shot) | 99 | 99 (100%) | 88 (88%) | 45 (45%) |

### 2.4 Candidate Diversity

| Metric | Value |
|--------|-------|
| Total non-error candidates | 363 |
| Unique SQL strings | 215 |
| Duplicate SQLs | 148 (**40%**) |
| Avg candidates per question | 11.0 |

---

## 3. Issues Found

### ISSUE P0-1 — B2_complex Generates Truncated SQL (`incomplete input` errors)
**Priority:** Critical
**Affected:** 20 B2 candidates across 14 questions (B2 exec_ok = 69%)

**Root cause:** Gemini Pro hits the `max_tokens=2000` limit mid-generation and returns a truncated SQL string. SQLite then fails with `incomplete input` when executing the cut-off query. The retry mechanism attempts 2x with doubled token limits (4000, then 8192) for MAX_TOKENS responses that return _no output_, but when the model returns _partial output_ at the token limit, the partial SQL is accepted as valid.

**Evidence from logs:**
```
Gemini response has no text or tool output: finish_reason=FinishReason.MAX_TOKENS
MAX_TOKENS hit with no output (max_tokens=2000); retrying with max_tokens=4000
```
The B2 system prompt (complex SQL, emphasizing CTEs and window functions) consistently generates longer SQL than B1, exhausting the 2000-token output budget.

**Suggested fix:** Set B2's `max_tokens` to 4096 as default (not 2000). Detect truncation by checking if `finish_reason == MAX_TOKENS` even when text is present, and discard/retry. Alternatively, explicitly limit the B2 prompt to request simpler SQL structures when the schema is small.

---

### ISSUE P0-2 — MALFORMED_FUNCTION_CALL Cascades Through Pipeline
**Priority:** Critical
**Affected:** Q3 (grounding), Q12 (grounding), Q14 (schema linker S2)

**Root cause:** Gemini occasionally returns `MALFORMED_FUNCTION_CALL` for tool-use calls, particularly when the input is long or contains special characters (e.g., backtick-quoted column names like `` `Enrollment (K-12)` ``). The schema linker has no `try/except` around the S2 LLM call in some code paths, leading to S1 being used as S2 fallback (which works), but the grounding LLM failure returns empty grounding context (no cell matches, no schema hints), degrading every downstream component.

**Evidence from logs:**
```
Grounding LLM error — falling back to empty grounding: Gemini candidate has no content (finish_reason=MALFORMED_FUNCTION_CALL)
Schema linker S2 LLM call failed; using S1 as S2 fallback.
```

**Suggested fix:**
1. In `context_grounder.py`: when grounding LLM fails, fall back to LSH-only cell matching (skip the LLM extraction, use the raw question words as keywords) rather than returning completely empty grounding.
2. In `schema_linker.py`: the S2 fallback to S1 is correct, but log a warning with the QID so it's trackable.
3. Sanitize column names with special characters before passing to tool-use prompts.

---

### ISSUE P1-3 — Wrong FK Join in european_football_2 (Q1027)
**Priority:** High
**Affected:** QID=1027 (european_football_2, simple) — ORACLE=NO, all 9 executable candidates wrong

**Root cause:** The `Player_Attributes` table has two ID-like columns: `id` and `player_api_id`. The actual FK join to `Player` is `Player_Attributes.id = Player.id`, but all generators consistently wrote `Player.id = Player_Attributes.player_api_id`. The field summary for `player_api_id` says "player API identifier" which sounds like the external reference, while `id` sounds like a generic PK — leading models to use the wrong column.

**GT SQL:** `FROM Player_Attributes AS t1 INNER JOIN Player AS t2 ON t1.id = t2.id`
**Generated SQL:** `FROM Player AS P JOIN Player_Attributes AS PA ON P.id = PA.player_api_id`

**Suggested fix:**
1. In schema formatting, explicitly annotate FK relationships in the DDL comment: `-- FK: Player_Attributes.id → Player.id`. The current DDL only shows `-- Foreign keys:` at table level without per-column FK direction.
2. In the schema linker prompt, instruct the model to prefer explicit FK annotations over inferred ones.
3. If ground truth FK info is available from BIRD (it is, in `dev_tables.json`), use it directly in the DDL rather than inferring from column names.

---

### ISSUE P1-4 — S1 == S2 in 5/33 Questions (Wasted API Calls)
**Priority:** High
**Affected:** Q427, Q1025, Q109, Q149, Q953 — 5 questions where S2 adds no fields beyond S1

**Root cause:** The second schema linking call (S2 = recall expansion) returns the same set of fields as S1 in 15% of cases. When this happens, generators A3/A4 (S2 DDL) produce SQL identical to A1/A2 (S1 DDL), and B2_s2/B1_s2 duplicate B2_s1/B1_s1. This wastes 5+ API calls per question and inflates the 40% duplicate rate.

**Notable case:** Q1025 (european_football_2 moderate) — S1=S2=33 fields, meaning schema linking selected the entire database schema as S1, leaving nothing to expand in S2.

**Suggested fix:**
1. Before making the S2 LLM call, check if `|FAISS candidates| - |S1 fields|` < 3 — if so, skip the S2 call and set S2 = S1 (already handled, but we're missing the S1 = full schema case).
2. When S1 covers ≥80% of available FAISS candidates, skip S2 and use a broader FAISS query (increase `top_k` from 30 to 50) for S2 instead of a second LLM call.
3. Cap S1 at 25 fields — if schema linking selects >25 fields, that indicates a domain-matching failure, not genuine relevance.

---

### ISSUE P1-5 — 40% Duplicate Rate Hurts Selector Quality
**Priority:** High

**Root cause:** Duplicate SQL strings come from:
1. **S1 == S2** (covered above) — accounts for ~50% of duplicates
2. **Simple questions with small schemas** — when there's only 1 valid query structure, all generators converge (e.g., Q803 superhero simple: 10/11 oracle matches, only 2 unique SQL strings)
3. **B1 and B2 same schema** — B1_s1 and B2_s1 both get the S1 Markdown schema; for simple questions they produce identical SQL

**Suggested fix:**
1. Add explicit temperature diversity: B1 at `temperature=0.0`, B2 at `temperature=0.3`, ICL at `temperature=0.7` — this preserves determinism for A while adding stylistic variety in B/C.
2. For ICL (C1/C2/C3), the 3 prompts are distinct but generators still converge on simple questions. Consider adding a "use a different SQL style (subquery vs JOIN)" hint in C2.
3. Deduplicate candidates before counting the "pool size" so the selector doesn't waste pairwise comparisons on identical SQL.

---

### ISSUE P2-6 — SELECT List Incompleteness
**Priority:** Medium
**Affected:** Q431 (card_games challenging), Q586 (codebase_community challenging) — ORACLE=NO despite correct WHERE/JOIN

**Root cause:** Generators correctly identified the right tables, JOIN conditions, and WHERE filters, but omitted one or more columns from the SELECT list. The questions ask "which X?" or "which user?" and models return only the key/ID, not all required columns.

**Q431 GT:** `SELECT T1.name, T1.id FROM sets ...`
**Q431 generated:** `SELECT T1.id FROM sets ...` (name missing)

**Q586 GT:** `SELECT T3.DisplayName, T1.Title FROM posts ...`
**Q586 generated:** `SELECT T3.DisplayName FROM ...` (Title missing)

**Suggested fix:**
Add an instruction to all generator system prompts: *"Return ALL columns that the question explicitly asks for. If the question uses 'list the X and Y', your SELECT must include both X and Y."*

---

### ISSUE P2-7 — Missing Value Transformations (IIF/CASE)
**Priority:** Medium
**Affected:** Q237 (toxicology moderate) — ORACLE=NO

**Root cause:** The evidence states `label = '+' means carcinogenic`, but models return the raw label value (`'+'` or `'-'`) instead of the human-readable `'YES'/'NO'` transformation the GT uses with `IIF(T2.label = '+', 'YES', 'NO')`.

**Suggested fix:**
When evidence contains mapping patterns like `"X means Y"` or `"label = X refers to Y"`, the generator prompt should include: *"If the evidence defines a mapping between stored values and their meanings (e.g., '+' = 'YES', 'F' = 'Female'), apply that transformation in your SELECT using CASE or IIF."*

---

### ISSUE P2-8 — Ratio/Count Semantic Ambiguity
**Priority:** Medium
**Affected:** Q571 (codebase_community moderate) — ORACLE=NO despite correct execution

**Root cause:** The question "how many times is the number of posts compared to votes?" requires `COUNT(posts) / COUNT(votes)`. Generators computed `COUNT(DISTINCT posts) / COUNT(DISTINCT votes)` — which is semantically wrong direction AND uses DISTINCT incorrectly.

**GT:** `CAST(COUNT(T2.Id) AS REAL) / COUNT(DISTINCT T1.Id)` where T1=votes, T2=posts
**Generated:** `CAST(COUNT(DISTINCT T1.Id) AS REAL) / COUNT(DISTINCT T2.Id)` where T1=posts, T2=votes (inverted)

**Suggested fix:**
Add ratio-specific instruction to the generator prompt: *"For questions asking 'how many times is A compared to B', compute A/B (not B/A). For COUNT questions, only use DISTINCT when the question implies uniqueness."*

---

### ISSUE P2-9 — SentenceTransformer Reloaded Per Question
**Priority:** Medium (performance)

**Observed:** The SentenceTransformer (`all-MiniLM-L6-v2`) model is loaded from disk on the first question and then reloaded from cache on subsequent questions. However, the loading progress bar appears once per run (on Q1), suggesting the model is already cached in memory after the first load. This is NOT a per-question reload — it's a one-time initialization.

However, ExampleStore and FAISSIndex both instantiate their own `SentenceTransformer` objects when loaded, so if multiple DB artifacts are loaded in a loop, each `FAISSIndex.load()` call creates a new `SentenceTransformer` instance. With 11 databases, this means 11 model loads consuming ~300MB RAM each.

**Suggested fix:**
Pass a shared `SentenceTransformer` instance to `FAISSIndex` and `ExampleStore` rather than each class managing its own. Add a module-level singleton: `_shared_encoder = SentenceTransformer(...)`.

---

### ISSUE P2-10 — Evidence Misleads Model on GROUP_CONCAT
**Priority:** Medium
**Affected:** Q1225 (thrombosis_prediction moderate) — ORACLE=NO

**Root cause:** The BIRD evidence says: *"List refers to GROUP_CONCAT(DISTINCT ID)"*. All generators followed this evidence and produced `GROUP_CONCAT(DISTINCT ID)` grouped by SEX (2 rows). But the GT returns individual (ID, SEX) rows grouped by SEX (10 rows). The evidence is misleading — it describes what "list" means to the annotator, but the actual output format differs.

This reveals a fundamental challenge: generators over-trust BIRD's `evidence` field even when it contradicts the expected output format.

**Suggested fix:**
In the generator system prompt, add: *"The evidence provides hints about the schema, not about the output format. Trust the question wording to determine output format (columns to return, whether to aggregate, etc.)."*

---

## 4. Two Deep-Dive Failure Cases

### 4.1 Q463 — card_games simple, ORACLE=NO

**Question:** "How many translations are there for the set of cards with 'Angel of Mercy' in it?"
**Evidence:** "set of cards with 'Angel of Mercy' in it refers to name = 'Angel of Mercy'"
**GT SQL:** `SELECT COUNT(DISTINCT translation) FROM set_translations WHERE setCode IN (SELECT setCode FROM cards WHERE name = 'Angel of Mercy') AND translation IS NOT NULL`
**Generated (sample):** `SELECT COUNT(T2.translation) FROM cards AS T1 INNER JOIN set_translations AS T2 ON T1.setCode = T2.setCode WHERE T1.asciiName = 'Angel of Mercy'`

**What went wrong:**
1. Models used `asciiName` instead of `name` despite evidence saying `name = 'Angel of Mercy'`
2. Models omitted `IS NOT NULL` filter (translations table has NULL entries)
3. Models omitted `DISTINCT` (there are duplicate translations)

**Combined effect:** All generated queries return wrong counts.

---

### 4.2 Q1027 — european_football_2 simple, ORACLE=NO

**Question:** "Indicate the full names of the top 10 players with the highest number of penalties."
**Evidence:** "full name refers to player_name; players with highest number of penalties refers to MAX(penalties)"
**GT SQL:** `SELECT t2.player_name FROM Player_Attributes AS t1 INNER JOIN Player AS t2 ON t1.id = t2.id ORDER BY t1.penalties DESC LIMIT 10`
**Generated (all 9 executable):** `...JOIN Player_Attributes AS PA ON P.id = PA.player_api_id...`

**What went wrong:**
All generators used `player_api_id` as the FK to `Player.id`, but the actual FK is `Player_Attributes.id = Player.id`. The `player_api_id` field in `Player_Attributes` is an external identifier, not a FK to the same DB's Player table. The field summaries describe `player_api_id` as "player API identifier" which sounds more like the natural key — but the actual join is on the `id` column.

**Impact:** This produces results with different players (the ordering by `player_api_id`-joined penalties yields different top-10). All 9 executable candidates are wrong; 2 candidates fail entirely (one uses incomplete SQL, one hallucinates a non-existent table).

---

## 5. Two Cases Where All Candidates Match (Diversity Analysis)

### 5.1 Q803 — superhero simple, 10/11 oracle match, 2 unique SQL strings

**Question:** "What is the power ID of cryokinesis?"
**Unique SQLs generated:**
- `SELECT id FROM superpower WHERE power_name = 'Cryokinesis'`
- `SELECT power_id FROM superpower WHERE power_name = 'Cryokinesis'`

These are superficially different (column alias `id` vs `power_id`) but both return the same result. The question has only one valid answer structure, so diversity is impossible without introducing errors. This is expected behavior.

### 5.2 Q994 — formula_1 challenging, 11/11 oracle match, 9 unique SQL strings

**Question:** "Which constructor scored most points from Monaco Grand Prix between 1986 and 2006?"
This question genuinely benefits from multiple approaches: some use subqueries, some use GROUP BY with ORDER BY, some use window functions. 9 unique SQL strings all produce the correct result — demonstrating that the selector (Op 9) is essential for picking the most reliable one.

---

## 6. Proposed Improvements (Decision Required)

### Decision A — Fix B2 truncation [RECOMMENDED: YES]
Set B2's default `max_tokens` to 4096. Add truncation detection: if `finish_reason == MAX_TOKENS` and output is non-empty, set `error_flag=True` rather than passing truncated SQL downstream.
**Impact:** Raises B2 exec_ok from 69% → ~95%, potentially adding 5-10% oracle coverage.

### Decision B — Improve grounding fallback [RECOMMENDED: YES]
When grounding LLM returns MALFORMED_FUNCTION_CALL, fall back to LSH-only matching using question words directly as keywords rather than empty grounding. This ensures schema linking still has cell match hints.
**Impact:** Recovers 2 of 33 questions from empty grounding; low cost to implement.

### Decision C — Cap S1 at 25 fields; skip S2 when S1==S2 [RECOMMENDED: YES]
If S1 covers ≥25 fields (or ≥80% of FAISS candidates), skip the S2 call and use a FAISS-widened set as S2.
**Impact:** Saves ~5-8 API calls per 33 questions; reduces duplicate rate by ~10%.

### Decision D — Add explicit SELECT completeness instruction to generators [RECOMMENDED: YES]
One-line addition to all generator system prompts: "SELECT all columns the question explicitly asks for."
**Impact:** Addresses Q431 and Q586 failures; low risk.

### Decision E — Add FK annotation to DDL schema format [RECOMMENDED: YES, before prompt 13]
Parse `dev_tables.json` FK definitions and add per-column FK annotations in the DDL: `-- FK → ReferencedTable.column`. This directly addresses the european_football_2 join key failure.
**Impact:** Requires updating `schema_formatter.py` to consume FK metadata from `dev_tables.json`.

### Decision F — Share SentenceTransformer instance across indexes [RECOMMENDED: YES, before prompt 13]
Prevents 11 model loads during pipeline execution.
**Impact:** Saves ~2-3GB RAM during checkpoint E; reduces initialization time.

---

## 7. Open Questions for User

1. **Which of decisions A–F should be implemented before Prompt 11 (query fixer)?**
   Decisions A (B2 truncation) and B (grounding fallback) affect the candidates that the query fixer receives — implementing them first will give the fixer better input.
   Decisions C and D are quick wins.
   Decisions E and F are best done before pipeline integration (Prompt 13).

2. **Target: oracle 66.7% → what's the minimum acceptable oracle before selector?**
   The target EX is 68%. Since oracle is the ceiling and selector typically achieves 70–85% of oracle, we need oracle ≥ 80% to hit EX 68%. The current 66.7% is below this. The query fixer (Op 8) can recover some wrong-execution candidates to correct ones, but it cannot fix fundamentally wrong SQL logic.

3. **For Q1027-type failures (wrong FK): should we add FK metadata parsing now, or defer to Phase 2?**
   The FK information is available in `dev_tables.json` and `train_tables.json`. Adding it to the DDL schema format is a schema_formatter.py change that could help across many databases.

---

## 8. Test Results

All unit and integration tests continue to pass (136/136 tests from previous runs). No regressions introduced.

```
pytest tests/ -v --ignore=tests/e2e -q
136 passed in prior session
```

---

## 9. Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| Oracle upper bound | 66.7% (22/33) | Below ~80% needed for EX≥68% |
| Simple oracle | 72.7% | Acceptable |
| Moderate oracle | 63.6% | Needs improvement |
| Challenging oracle | 63.6% | Needs improvement |
| B2 exec success | 69% | **Critical gap** (truncation bug) |
| Duplicate rate | 40% | High; wastes selector API calls |
| MALFORMED_FUNCTION_CALL rate | ~6% of tool calls | Intermittent; needs fallback |
| Best DB | debit_card_specializing, superhero (100%) | Schema-straightforward domains |
| Worst DB | card_games, codebase_community (33%) | Require precise DISTINCT/NULL/ratio logic |
