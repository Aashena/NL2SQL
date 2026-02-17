# CHASE-SQL: Method Analysis

## 1. General Pipeline

The CHASE-SQL framework transforms a natural language question into an executable SQL query through the following end-to-end pipeline:

```
Natural Language Question + Database Schema + Hints
        │
        ▼
┌─────────────────────┐
│  Value Retrieval     │  Extract keywords → LSH matching → Re-ranking
│  (LSH-based)         │
└────────┬────────────┘
         │ Enriched context (question + schema + retrieved values)
         ▼
┌─────────────────────────────────────────────────────┐
│            Multi-Path Candidate Generation           │
│                                                       │
│  ┌───────────────┐ ┌───────────────┐ ┌─────────────┐ │
│  │ Divide &      │ │ Query Plan    │ │ Online      │ │
│  │ Conquer CoT   │ │ CoT           │ │ Synthetic   │ │
│  │ (7 candidates)│ │ (7 candidates)│ │ Examples    │ │
│  │               │ │               │ │(7 candidates)│ │
│  └───────┬───────┘ └───────┬───────┘ └──────┬──────┘ │
└──────────┼─────────────────┼────────────────┼────────┘
           │                 │                │
           ▼                 ▼                ▼
┌─────────────────────────────────────────────────────┐
│              Query Fixer (per candidate)              │
│         Up to β=3 iterations of self-reflection       │
└────────────────────────┬────────────────────────────┘
                         │ 21 fixed candidate SQL queries
                         ▼
┌─────────────────────────────────────────────────────┐
│              Selection Agent                          │
│   Pairwise comparison of all candidate pairs          │
│   using fine-tuned binary classifier                  │
│   → Cumulative scoring → Pick highest-scored query    │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
                  Final SQL Query
```

**Data flow summary:** Raw input (question, schema, hints) → keyword extraction and value retrieval → context enrichment → parallel multi-path SQL generation (3 generators × 7 samples = 21 candidates) → iterative query fixing → pairwise selection via trained classifier → single final SQL output.

---

## 2. Detailed Operation Analysis

### 2.1 Value Retrieval (LSH-based)

**Input:** Natural language question, target database with all its rows and values.

**Output:** A set of relevant database values (strings, numbers) that are likely referenced in the question, to be used in SQL clauses like `WHERE` and `HAVING`.

**Operation Details:**
1. An LLM extracts keywords from the natural language question using few-shot prompting.
2. For each keyword, Locality-Sensitive Hashing (LSH) retrieves the most syntactically similar values from the database.
3. Retrieved values are re-ranked using a combination of embedding-based semantic similarity and edit distance.
4. This approach is robust to typos and captures semantic meaning beyond exact string matching.

**Role in Pipeline:** This is the grounding step. It bridges the gap between natural language references (which may contain typos, abbreviations, or synonyms) and actual database values. Without it, generated SQL queries would frequently contain incorrect literal values in filter conditions.

**Significance Score: 7/10**
Removing LSH causes a 2.92% drop in execution accuracy (73.01% → 70.09%, Table 7). This is a substantial contribution. Value retrieval is particularly critical for databases with large vocabularies or when questions contain typos/abbreviations. It is a necessary prerequisite that enables all downstream generators to produce syntactically valid and semantically correct filter conditions.

**Complexity Score: 4/10**
- Keyword extraction requires one LLM call (low cost).
- LSH indexing and lookup is sub-linear in the number of database values — O(n) for index construction (done once per database) and approximately O(1) per lookup.
- Embedding computation and re-ranking add moderate cost but are performed only over the small set of LSH-retrieved candidates.
- Overall, this is a lightweight preprocessing step relative to the generation stages.

---

### 2.2 Candidate Generator 1: Divide and Conquer CoT

**Input:** Few-shot examples M with divide-and-conquer reasoning patterns, user question Q_u, target database schema D (enriched with retrieved values), and the LLM θ.

**Output:** A single SQL query produced through hierarchical decomposition and reassembly (repeated 7 times with temperature sampling for diversity).

**Operation Details:**
1. **Divide:** The LLM decomposes the original question into a set of sub-questions, each expressible as a simpler pseudo-SQL fragment.
2. **Conquer:** For each sub-question, a partial SQL query is generated. Each sub-query generation is conditioned on previously solved sub-queries, enabling incremental construction.
3. **Assemble:** All partial SQL queries are combined into a single final SQL query.
4. **Optimize:** Redundant clauses and conditions are removed from the assembled query.

All steps happen in a single LLM call with structured few-shot prompting.

**Role in Pipeline:** This generator excels at complex queries involving nested subqueries, intricate WHERE/HAVING conditions, and advanced mathematical operations. It provides a structured decomposition approach that complements the other generators. The Venn diagram (Fig. 3) shows 33 questions uniquely solved by this method alone.

**Significance Score: 7/10**
- Removing DC causes a 1.24% drop (Table 7), the largest among the three generators.
- As a single generator, it achieves 65.77% with the query fixer (Table 4), a +8.02% improvement over baseline.
- It uniquely solves complex queries the other methods cannot handle (Fig. 17 shows an example).
- Its contribution to diversity is essential for the selection agent to work effectively — the upper bound of all three combined (82.79%) is significantly higher than any single method.

**Complexity Score: 7/10**
- Each candidate requires one LLM call, but the prompt is long (few-shot examples with full decomposition chains + database schema).
- 7 candidates are generated, each with temperature sampling — 7 LLM calls per question.
- The structured decomposition within the prompt increases token count significantly compared to simple prompting.
- Overall moderate-to-high cost due to multiple long-context LLM inference calls.

---

### 2.3 Candidate Generator 2: Query Plan CoT

**Input:** Few-shot examples with query-plan-style reasoning, user question Q_u, target database schema D, and the LLM θ.

**Output:** A single SQL query produced by reasoning through the execution steps a database engine would take (repeated 7 times for diversity).

**Operation Details:**
1. The reasoning follows the structure of a database query execution plan: (a) identify and locate relevant tables, (b) perform operations such as counting, filtering, or matching between tables, (c) select the appropriate columns to return.
2. The EXPLAIN command output is converted from machine-readable format into human-readable text that aligns with LLM pretraining data.
3. The LLM generates the SQL by following this execution-plan-style chain of thought with few-shot demonstrations.

**Role in Pipeline:** This generator complements the divide-and-conquer approach. While DC excels at decomposing complex questions, Query Plan CoT excels when questions require reasoning about relationships between different parts of the question and the database schema. It systematically explains which tables to scan, how to match columns, and how to apply filters. 35 questions are uniquely solved by this method (Fig. 3).

**Significance Score: 6/10**
- Removing QP causes a 0.65% drop (Table 7), the smallest among the three generators.
- As a single generator, it achieves 65.51% with fixer (Table 4), a +7.76% improvement over baseline.
- Its unique contribution is primarily in moderate-difficulty queries (Fig. 3b) and in providing diversity that increases the upper bound for the selection agent.
- While individually the least impactful to remove, its presence increases the combined upper bound meaningfully.

**Complexity Score: 7/10**
- Similar to DC CoT: each candidate requires one LLM call with a long prompt containing few-shot execution plan examples.
- 7 candidates generated per question — 7 LLM calls.
- The execution plan reasoning adds moderate token overhead within each generation.
- Comparable computational cost to DC CoT.

---

### 2.4 Candidate Generator 3: Online Synthetic Example Generation

**Input:** User question Q_u, hint H_u, target database D, filtered relevant table columns t (from column selection), LLM θ, guidelines R_f (SQL features) and R_t (filtered schema), and target counts n_f and n_t.

**Output:** A set of synthetic few-shot examples tailored to the specific question and database, which are then used as in-context demonstrations to generate the SQL query (repeated 7 times).

**Operation Details:**
1. **Column Selection:** Relevant columns are identified for the given question (similar to schema linking from CHESS).
2. **Feature-based example generation (R_f):** The LLM generates synthetic question-SQL pairs that showcase common SQL features (equality predicates, JOINs, nested JOINs, ORDER BY, GROUP BY, HAVING, aggregations) using the full database schema.
3. **Schema-targeted example generation (R_t):** The LLM generates examples specifically using the filtered columns relevant to the current question, highlighting correct schema interpretation.
4. **Mixing:** Both sets of examples are combined to create a diverse example pool that avoids over-fitting to specific patterns.
5. **SQL Generation:** The combined synthetic examples are injected into the prompt as few-shot demonstrations, and the LLM generates the final SQL query.

**Role in Pipeline:** This is the highest-performing single generator (68.02% with fixer). It addresses a fundamental challenge: off-the-shelf few-shot examples may not help the LLM understand the specific database schema. By generating examples tailored to the target database and question, it provides instance-aware context that bridges the gap between the LLM's general SQL knowledge and the specific database structure. 30 questions are uniquely solved by this method (Fig. 3).

**Significance Score: 7/10**
- Removing OS causes a 0.85% drop (Table 7).
- As a single generator, it achieves the highest accuracy at 68.02% (Table 4), a +10.27% improvement over baseline.
- The ablation on example types (Table 8) shows that mixing R_f and R_t is critical: R_f alone gives +7.7%, R_t alone gives +9.0%, and combined +9.34%.
- It has the highest lower bound among generators (60.43% at T=0.5, Table 6), meaning it produces the most consistently correct candidates, reducing noise for the selection agent.

**Complexity Score: 9/10**
- This is the most expensive generator. It requires:
  - 1 LLM call for column selection/filtering.
  - 1 LLM call to generate R_f examples (feature-based, 75 total examples).
  - 1 LLM call to generate R_t examples (schema-targeted, 75 total examples).
  - 7 LLM calls to generate the actual SQL candidates using the synthetic examples.
- Total: ~10 LLM calls per question, with the example generation calls producing large outputs.
- The synthetic examples are generated online (per question), not cached, making this substantially more expensive than the other generators.

---

### 2.5 Query Fixer

**Input:** A candidate SQL query, the database schema, the original question, hints, and execution feedback (syntax errors or empty result sets).

**Output:** A corrected SQL query (or the original if no issues were detected).

**Operation Details:**
1. Each generated candidate SQL query is executed against the target database.
2. If execution produces a syntax error or an empty/unexpected result, the query enters a fixing loop.
3. The LLM receives the original query, the error message or execution result, the database schema, question, and hints.
4. Using self-reflection (Reflexion method), the LLM analyzes why the query failed and produces a corrected version.
5. This process repeats up to β=3 iterations.

**Role in Pipeline:** This is a quality assurance step that filters out syntactically invalid queries and attempts to rescue queries that are close to correct but have minor issues. It operates independently on each candidate before they enter the selection phase, ensuring that the candidate pool presented to the selection agent is of higher quality.

**Significance Score: 8/10**
- Removing the query fixer causes a 3.78% drop (Table 7), the second-largest single-component impact.
- Across all generators, it consistently improves performance by ~2% (Table 4): QP 63.62→65.51, DC 63.92→65.77, OS 67.09→68.02.
- It is critical because even small syntax errors make a query completely wrong, and fixing them is often straightforward for an LLM.
- The fixer's impact is amplified in the ensemble setting because it increases the proportion of correct candidates in the pool, making the selection agent's job easier.

**Complexity Score: 6/10**
- In the worst case, each candidate undergoes β=3 fixing iterations, each requiring one LLM call + one database query execution.
- With 21 candidates, this could mean up to 63 additional LLM calls, but in practice most queries either pass on the first execution or are fixed within 1-2 iterations.
- Database query execution is fast (milliseconds for typical queries).
- Moderate overall cost — significant but bounded by the small iteration limit.

---

### 2.6 Selection Agent (Pairwise Binary Classification)

**Input:** A set of n=21 candidate SQL queries (post-fixing), user question Q_u, hint H_u, target database D, and a fine-tuned binary classification model θ_p (Gemini 1.5 Flash with LoRA).

**Output:** A single final SQL query selected as the best answer.

**Operation Details:**
1. **Score initialization:** Each candidate starts with a score of 0.
2. **Pairwise comparison:** For every distinct pair (c_i, c_j):
   - If both queries produce the same execution result, one is marked as winner (consistency shortcut).
   - If execution results differ, the union of schemas used by both candidates is constructed, and the fine-tuned binary classifier determines which query is more likely correct.
   - Both orderings (c_i, c_j) and (c_j, c_i) are evaluated to mitigate order bias.
   - The winner's score is incremented by 1.
3. **Final selection:** The candidate with the highest cumulative score is chosen. Ties are broken arbitrarily.

**Training Details:**
- Training data: 3.8K samples generated by running candidate generators on the BIRD training set.
- Model: Gemini 1.5 Flash fine-tuned with LoRA (rank 16) for 10 epochs via Vertex AI.
- Training format: Tuples (Q_u, C_i, C_j, D_ij, y_ij) with randomized ordering to prevent order bias.
- Achieved 71.01% binary classification accuracy (Table 5), significantly above the ~60% of untuned LLMs.

**Role in Pipeline:** This is the critical decision-making component that converts diversity into accuracy. The entire multi-path generation strategy is predicated on the ability to identify the correct query from a diverse pool. Without an effective selection mechanism, generating more diverse candidates would not translate to improved final accuracy. The selection agent closes the gap between self-consistency (68.84%) and the upper bound (82.79%), achieving 73.01%.

**Significance Score: 10/10**
- This is the most impactful component. Replacing the selection agent with self-consistency drops performance by 4.17% (Table 7).
- Replacing it with a ranker agent (all candidates in one prompt) drops performance by 7.5%.
- The selection agent achieves ~6% improvement over self-consistency across all generator types (Table 6).
- It is the component that makes the multi-path generation strategy viable — without it, the framework would plateau at self-consistency levels.
- The pairwise comparison design is specifically chosen because comparing 2 candidates is more tractable for the model than ranking all 21 simultaneously.

**Complexity Score: 8/10**
- Number of pairwise comparisons: C(21, 2) × 2 = 420 comparisons (both orderings).
- Each comparison that involves different execution results requires one inference call to the fine-tuned model.
- Comparisons with identical execution results are short-circuited (no model call needed).
- The fine-tuned Gemini 1.5 Flash model is faster than Pro, but 420 potential calls is still substantial.
- Each call includes the database schema union, question, hint, and both candidate queries — moderate context length.
- Additionally, all 21 candidates must be executed against the database to obtain execution results for the consistency shortcut.

---

### 2.7 Column Selection / Schema Filtering (Supporting Operation)

**Input:** User question, database schema, LLM.

**Output:** A filtered set of relevant tables and columns.

**Operation Details:**
- Similar to the approach in CHESS (Talaei et al., 2024), relevant columns are identified for the given question.
- Used primarily by the Online Synthetic Example generator (to create schema-targeted examples) and by the Selection Agent (to construct schema unions for pairwise comparisons).

**Role in Pipeline:** A supporting operation that reduces noise in the schema presented to various components, helping both generation quality and selection accuracy.

**Significance Score: 4/10**
- Not independently ablated in the paper, but implicitly contributes to the OS generator's performance and the selection agent's schema union construction.
- Its impact is embedded within the OS and selection agent results.

**Complexity Score: 3/10**
- One LLM call per question.
- Lightweight compared to the generation and selection stages.

---

## Summary Table

| Operation | Significance (1-10) | Complexity (1-10) | Accuracy Contribution |
|---|---|---|---|
| Value Retrieval (LSH) | 7 | 4 | -2.92% when removed |
| Divide & Conquer CoT | 7 | 7 | -1.24% when removed; +8.02% over baseline as single generator |
| Query Plan CoT | 6 | 7 | -0.65% when removed; +7.76% over baseline as single generator |
| Online Synthetic Examples | 7 | 9 | -0.85% when removed; +10.27% over baseline as single generator |
| Query Fixer | 8 | 6 | -3.78% when removed; ~+2% per generator |
| Selection Agent | 10 | 8 | -4.17% vs self-consistency; -7.5% vs ranker agent |
| Column Selection | 4 | 3 | Not independently measured |

**Key Insight:** The framework's power comes from the synergy between diversity-maximizing generation and preference-optimized selection. The three generators contribute a combined upper bound of 82.79%, meaning the LLM's parametric knowledge is sufficient to solve most questions — the challenge is extracting and identifying the correct answer. The selection agent closes ~42% of the gap between self-consistency (68.84%) and the upper bound (82.79%), achieving the final 73.01%.
