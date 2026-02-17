# Agentar-Scale-SQL: Method Analysis

## 1. General Pipeline Overview

The Agentar-Scale-SQL framework processes a natural language question into a final SQL query through a two-phase architecture: an **offline preprocessing** phase and an **online inference** framework with three sequential stages.

### High-Level Data Flow

```
Raw Inputs: (Question Q_u, Evidence E_u, Database D)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  OFFLINE PREPROCESSING (one-time per database)                  │
│                                                                 │
│  Database D ──┬──► DDL Schema (for Reasoning Generator)         │
│               ├──► Light Schema / Markdown (for ICL Generator)  │
│               └──► Cell Value Vector Store (VD_cell)            │
│                                                                 │
│  Training Set ──► Example Vector Store (VD_example)             │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: TASK UNDERSTANDING                                     │
│                                                                 │
│  (Q_u, E_u) ──► Keyword Extraction ──► Cell Retrieval           │
│                                         from VD_cell            │
│  Q_u ──────► Skeleton Extraction ──► Example Retrieval          │
│                                       from VD_example           │
│                                                                 │
│  Output: Matched cell values + Retrieved few-shot examples      │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: SQL GENERATION SCALING                                 │
│                                                                 │
│  ┌──────────────────────┐   ┌──────────────────────┐            │
│  │  Reasoning Generator │   │    ICL Generator     │            │
│  │  (RL-tuned, DDL      │   │  (Gemini-2.5-Pro,    │            │
│  │   schema, 8 queries) │   │   GPT-5, light       │            │
│  │                      │   │   schema, 9 queries) │            │
│  └──────────┬───────────┘   └──────────┬───────────┘            │
│             └──────────┬───────────────┘                        │
│                        ▼                                        │
│              17 raw SQL candidates                              │
│                        │                                        │
│                        ▼                                        │
│  ┌─────────────────────────────────────────────┐                │
│  │         Iterative Refinement                 │                │
│  │  SQL Fixer (syntax) + SQL Revisor (semantic) │                │
│  └─────────────────────────────────────────────┘                │
│                        │                                        │
│              n refined SQL candidates (C)                        │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: SQL SELECTION SCALING                                  │
│                                                                 │
│  C ──► Group by execution results ──► Representative set C'     │
│                                                                 │
│  C' ──► Pairwise Round-Robin Tournament                         │
│         (RL-enhanced Reasoning Selector)                        │
│                                                                 │
│  Output: c_final = argmax(W_i) ──► Final SQL Query              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Operation Analysis

---

### Operation 0: Offline Preprocessing — Dual Schema Generation

#### 2.1 Input, Output, and Details

- **Input:** Raw database D (tables, columns, constraints, cell values)
- **Output:** Two schema representations:
  - **DDL Schema:** Standard SQL CREATE TABLE statements enriched with example cell values as inline comments. Used by the Reasoning Generator.
  - **Light Schema:** Markdown-based tabular format with column descriptions, types, value examples, primary keys, and foreign keys. Used by the ICL Generator.
- **Details:** The database metadata is parsed and reformatted into two complementary views. The DDL format is code-like and aligns with how code-specialized models were pre-trained. The Markdown light schema is more human-readable and suitable for in-context learning with general-purpose LLMs. This dual representation is a deliberate design choice to maximize diversity across generators.

#### 2.2 Role in the General Pipeline

This operation establishes the foundational context that all downstream generators consume. By creating two distinct schema views, it enables the Diverse Synthesis strategy: the Reasoning Generator can leverage code-oriented DDL for precise SQL construction, while the ICL Generator benefits from the structured readability of Markdown. Without this step, the two generators would operate on identical input formats, reducing candidate diversity.

#### 2.3 Performance Significance: **4/10**

The dual schema format is an enabler of diversity rather than a direct performance driver. Its contribution is indirect — it makes the Diverse Synthesis strategy possible. The ablation study does not isolate this component, but its effect is subsumed within the generator ablations. The format difference alone does not explain the large gains; rather, it is the combination of different models and prompting strategies that drives diversity.

#### 2.4 Complexity: **3/10**

This is a one-time preprocessing step per database. It involves parsing database metadata and reformatting it into two textual representations. The computation is deterministic string manipulation with no LLM calls. For BIRD's 95 databases, this is a lightweight batch operation.

---

### Operation 1: Offline Preprocessing — Cell Value Indexing

#### 2.1 Input, Output, and Details

- **Input:** All textual cell values from the database D
- **Output:** A vector store VD_cell indexed with embeddings (all-MiniLM-L6-v2) of cell values
- **Details:** Every cell value in every column of the database is embedded using the all-MiniLM-L6-v2 sentence transformer model and stored in a Chroma vector database. This creates a dense retrieval index that can be queried at inference time with keywords extracted from the user's question.

#### 2.2 Role in the General Pipeline

This vector store is consumed in Step 1 (Task Understanding) to retrieve relevant cell values that may appear in WHERE, HAVING, or other filtering clauses of the SQL. Without accurate cell value retrieval, the generated SQL would frequently reference incorrect or non-existent values, leading to empty or wrong results.

#### 2.3 Performance Significance: **3/10**

Cell value retrieval is a supporting operation. The ablation for "w/o Task Understanding" shows only a -0.45 EX drop, and cell retrieval is only one part of Task Understanding. The impact is modest but nonzero — it helps ground SQL generation in actual database values, which is particularly important for questions referencing specific entities.

#### 2.4 Complexity: **4/10**

Embedding all cell values for large databases can be moderately expensive. For BIRD's 95 databases with potentially millions of cells, the embedding computation is non-trivial but parallelizable. This is a one-time cost using a small (22M parameter) embedding model, so it is manageable.

---

### Operation 2: Offline Preprocessing — Example Indexing

#### 2.1 Input, Output, and Details

- **Input:** Training set question-SQL pairs from BIRD
- **Output:** A vector store VD_example indexed with embeddings of question skeletons
- **Details:** Each training example's question is processed to extract a skeleton (abstracting away specific entity names), then embedded and stored in Chroma. At inference time, the skeleton of the incoming question is used to retrieve similar examples for few-shot ICL.

#### 2.2 Role in the General Pipeline

Retrieved examples serve as few-shot demonstrations for the ICL Generator in Step 2. By providing structurally similar solved examples, the ICL Generator can leverage pattern recognition to produce SQL for new questions. The skeleton-based retrieval ensures that examples are matched on query structure rather than surface-level entity overlap.

#### 2.3 Performance Significance: **5/10**

Few-shot examples are a core enabler of the ICL Generator, which contributes -3.78 EX when removed. However, the example retrieval is just one ingredient — the ICL Generator also depends on prompt diversity, model capability, and temperature variation. The retrieval quality directly affects ICL output quality, making this moderately significant.

#### 2.4 Complexity: **2/10**

Embedding the BIRD training set (~9,428 dev+train examples) is a small-scale operation. The all-MiniLM-L6-v2 model processes these quickly, and this is a one-time offline cost.

---

### Operation 3: Task Understanding — Keyword Extraction + Cell Retrieval

#### 2.1 Input, Output, and Details

- **Input:** Question Q_u, Evidence E_u, Database Schema
- **Output:** A JSON containing:
  - `database_literals`: Specific values extracted from the question/evidence (e.g., "Japan", "1996")
  - `question_skeleton`: Abstracted question template with placeholders
- **Details:** Powered by Gemini-2.5-Flash (temperature 0.2). The LLM identifies database literals (specific values that would appear in WHERE clauses) and generates a question skeleton. The extracted keywords are then used to query VD_cell via embedding-based similarity search to retrieve matched cell values with their source table and column information.

#### 2.2 Role in the General Pipeline

This operation serves as the "grounding" step that connects the natural language question to actual database content. The matched cell values are fed into both generators as "Matched contents," providing concrete reference values. The question skeleton is used to retrieve few-shot examples from VD_example. This step resolves ambiguities in the question before generation begins.

#### 2.3 Performance Significance: **3/10**

The ablation shows removing Task Understanding causes only a -0.45 EX drop on the development set. While this is a measurable contribution, it is the smallest among all ablated components. The moderate effect suggests that the generators are already reasonably capable of inferring relevant values from the schema alone, and that Task Understanding primarily helps on edge cases with ambiguous value references.

#### 2.4 Complexity: **2/10**

This involves 1 LLM call (Gemini-2.5-Flash, a lightweight model) plus a vector similarity search. The Flash model is optimized for speed, and the retrieval is a standard nearest-neighbor lookup. Total latency is minimal relative to the generation and selection stages.

---

### Operation 4: SQL Generation — Intrinsic Reasoning Generator

#### 2.1 Input, Output, and Details

- **Input:** DDL Schema, Matched Contents (from Task Understanding), Question, Evidence
- **Output:** 8 SQL candidate queries
- **Details:** This generator is a fine-tuned Omni-SQL-32B model enhanced via GRPO reinforcement learning on BIRD training data. The RL training uses an execution-based reward:
  - R=1 if results match ground truth
  - R=0.1 if SQL is executable but incorrect
  - R=0 if SQL has errors

  The model performs extended chain-of-thought reasoning internally (internal scaling) before producing SQL. At inference, 8 candidates are sampled with varying temperatures. The model uses DDL schema format, which aligns with its code-specialized pre-training.

#### 2.2 Role in the General Pipeline

This is one of two parallel generators in the Diverse Synthesis strategy. It provides the "depth" component — producing SQL through careful step-by-step reasoning that excels at logical accuracy, particularly on simple and moderate questions. The RL training specifically optimizes for execution correctness, making it the most reliable single generator. It uniquely solves 12 samples that the ICL generator cannot.

#### 2.3 Performance Significance: **9/10**

Removing the Reasoning Generator causes the largest performance drop of any single component: **-4.89 EX** (from 74.90 to 70.01). The drop is substantial across all difficulty levels. This generator is the backbone of the system — it provides the highest per-candidate accuracy through RL-optimized reasoning. The internal scaling paradigm (RL-enhanced extended reasoning) is the paper's most impactful contribution to the field.

#### 2.4 Complexity: **7/10**

This requires running a 32B parameter model 8 times per question with extended chain-of-thought reasoning. Each inference involves significant GPU compute (the model was trained on 32x A100 80GB GPUs). The RL training phase itself is extremely expensive (GRPO with execution-based rewards requires running SQL queries against databases for each rollout), though this is a one-time offline cost. At inference, the 8 parallel forward passes on a 32B model dominate the compute budget of the generation stage.

---

### Operation 5: SQL Generation — ICL Generator (Diverse Synthesis)

#### 2.1 Input, Output, and Details

- **Input:** Light Schema (Markdown), Retrieved Few-Shot Examples, Matched Contents, Question, Evidence
- **Output:** 9 SQL candidate queries
- **Details:** Uses multiple large proprietary LLMs:
  - Gemini-2.5-Pro at temperature 0.5 and 1.8
  - GPT-5 with minimal reasoning effort

  Diversity is maximized through four dimensions:
  1. **Prompt variation:** Direct prompting, Chain-of-Thought, Problem decomposition
  2. **Example ordering:** Randomized order of few-shot examples
  3. **Model variation:** Multiple LLMs (Gemini-2.5-Pro, GPT-5)
  4. **Temperature variation:** 0.5 (focused) and 1.8 (creative)

#### 2.2 Role in the General Pipeline

This is the "breadth" component of Diverse Synthesis. While the Reasoning Generator goes deep with one fine-tuned model, the ICL Generator goes wide with multiple models, prompts, and temperatures. It achieves a higher upper bound accuracy (81.36% vs 75.88%) because its diverse sampling explores more of the solution space. It uniquely solves 47 samples the Reasoning Generator misses, and is particularly strong on challenging questions (85 vs 81 correct). The combination of both generators achieves an upper bound of 84.29%.

#### 2.3 Performance Significance: **8/10**

Removing the ICL Generator causes the second-largest drop: **-3.78 EX** (from 74.90 to 71.12). The impact on challenging questions is especially severe (64.14 to 55.86, a -8.28 drop). The ICL Generator's higher upper bound and complementary coverage with the Reasoning Generator make it critical. The combined upper bound (84.29%) is substantially higher than either alone (81.36% or 75.88%), confirming their synergy.

#### 2.4 Complexity: **6/10**

This involves 9 API calls to large proprietary models (Gemini-2.5-Pro, GPT-5). While the compute is offloaded to API providers, the monetary cost and latency are significant. Each call involves processing the full schema, examples, and question. The calls can be parallelized, reducing wall-clock time. Compared to running a local 32B model 8 times, API calls are faster in latency but add financial cost.

---

### Operation 6: Iterative Refinement (SQL Fixer + SQL Revisor)

#### 2.1 Input, Output, and Details

- **Input:** The n SQL candidates from the generation stage, their execution results against the database
- **Output:** Refined set of SQL candidates with syntax and semantic errors corrected
- **Details:** Two sub-components:
  - **SQL Fixer:** Conditionally activated when a candidate SQL produces an execution error or empty result. An LLM (Gemini-2.5-Pro) receives the original SQL, the error message, the schema, and the question, then outputs a corrected SQL.
  - **SQL Revisor:** An LLM agent that identifies and refines logical/semantic flaws in queries that execute successfully but may be incorrect. Candidates are first grouped by execution outcome, then one representative from each group is selected for revision.

  The number of LLM calls ranges from 1-51 depending on how many candidates need fixing/revising.

#### 2.2 Role in the General Pipeline

This is the **Sequential Scaling** component of the framework. It acts as a quality improvement layer between generation and selection. By correcting syntax errors (turning non-executable SQL into executable SQL) and refining semantic errors (correcting logical mistakes), it increases the proportion of valid and correct candidates in the pool. This reduces the burden on the selection stage by improving candidate quality.

#### 2.3 Performance Significance: **4/10**

The ablation shows a **-0.52 EX** drop when Iterative Refinement is removed. This is a relatively modest contribution compared to the generators (-4.89, -3.78) and selection (-1.82). The limited impact suggests that the generators already produce mostly syntactically valid SQL, and the semantic revisions have hit-or-miss success. However, the 0.52 points may matter at the top of the leaderboard where margins are thin.

#### 2.4 Complexity: **5/10**

The cost is variable: 1-51 LLM calls depending on the number of erroneous candidates. Each call requires Gemini-2.5-Pro to analyze the SQL, its error, and the schema. In the worst case (many broken candidates), this is expensive. In practice, with well-tuned generators, most candidates are valid, keeping the average cost moderate. The grouping strategy (one revision per execution-outcome group) is a smart optimization that limits redundant work.

---

### Operation 7: SQL Selection — Execution-Based Grouping

#### 2.1 Input, Output, and Details

- **Input:** All n SQL candidates C = {c_1, ..., c_n} and the database D
- **Output:** Deduplicated candidate set C' = {c_1, ..., c_m} where m <= n
- **Details:** Each candidate SQL is executed against database D. Candidates that produce identical execution results are grouped together, and one representative is selected from each group. This transforms a set of ~17 candidates into a smaller set of m distinct-result candidates. This is essentially a deduplication step based on functional equivalence.

#### 2.2 Role in the General Pipeline

This is a preprocessing step for Tournament Selection. By collapsing functionally identical candidates, it reduces the O(m^2) pairwise comparison cost and ensures the selection stage focuses on genuinely different solutions. Without this step, the tournament would waste LLM calls comparing candidates that are functionally equivalent.

#### 2.3 Performance Significance: **2/10**

This step has no direct impact on accuracy — it is a computational optimization. If all candidates were fed into the tournament without grouping, the same final answer would be selected (just with more redundant comparisons). Its value is purely in reducing the selection cost.

#### 2.4 Complexity: **2/10**

Executing ~17 SQL queries against a database is fast (milliseconds to seconds per query). The grouping is a simple comparison of result sets. This is negligible compared to the LLM-based steps.

---

### Operation 8: SQL Selection — Tournament Selection with Reasoning Selector

#### 2.1 Input, Output, and Details

- **Input:** Candidate set C' = {c_1, ..., c_m}, Light Schema, Matched Contents, Question, Evidence, and execution results of each candidate
- **Output:** Single final SQL query c_final
- **Details:** A pairwise round-robin tournament:
  1. For every pair (c_i, c_j) in C', the RL-enhanced Reasoning Selector (Qwen2.5-Coder-32B-Instruct fine-tuned with GRPO) evaluates which candidate better answers the question
  2. The winner of each comparison gets a point (W_i incremented)
  3. The candidate with the maximum total score is selected: c_final = argmax(W_i)

  The selector is trained with a binary reward: R_S = 1 if correct selection, 0 otherwise. The number of pairwise comparisons is C(m,2) = m(m-1)/2, which ranges from 1 to 561 LLM calls depending on m.

  The selector receives the full context: database schema, matched cell values, the question, evidence, and the actual execution results of both candidate SQL queries, enabling informed comparison.

#### 2.2 Role in the General Pipeline

This is the final decision-making step and implements **Parallel Scaling** at the selection level. It replaces naive majority voting (self-consistency) with an intelligent, RL-trained judge. The tournament format ensures every candidate is compared against every other candidate, and the RL training teaches the selector to reason about SQL correctness beyond simple output frequency. This is the "last mile" that converts a diverse candidate pool into a single high-quality answer.

#### 2.3 Performance Significance: **7/10**

The ablation shows **-1.82 EX** when replacing tournament selection with majority voting. This is the third-largest contribution. Critically, this operation can only be effective if the candidate pool already contains a correct answer (upper bound 84.29%), so its value is multiplicative with the generators. The selection model achieves 74.90% from a pool where voting achieves only 73.08%, demonstrating that it can identify correct but less frequent answers. The gap between the upper bound (84.29%) and the selection result (74.90%) shows significant room for improvement in this component.

#### 2.4 Complexity: **8/10**

This is the most computationally expensive step at inference time. With m distinct-result groups, the number of pairwise comparisons is O(m^2). In the worst case, this means up to 561 LLM calls to the 32B Reasoning Selector. Each call requires processing the full schema, question, and two candidates with their execution results. The paper reports total inference calls of 20-630, with selection dominating the upper end. The RL training cost is also significant (8.5k samples, GRPO on 32 A100 GPUs), though this is a one-time offline expense.

---

## 3. Summary Table

| # | Operation | Scaling Type | Performance Significance (1-10) | Computation Complexity (1-10) | Ablation Impact (EX) |
|---|-----------|-------------|---:|---:|---:|
| 0 | Dual Schema Generation | — (Offline) | 4 | 3 | (indirect) |
| 1 | Cell Value Indexing | — (Offline) | 3 | 4 | (part of -0.45) |
| 2 | Example Indexing | — (Offline) | 5 | 2 | (part of -0.45) |
| 3 | Task Understanding (Keyword + Skeleton) | — (Preprocessing) | 3 | 2 | -0.45 |
| 4 | Reasoning Generator (RL-tuned) | Internal + Parallel | 9 | 7 | -4.89 |
| 5 | ICL Generator (Multi-LLM) | Parallel | 8 | 6 | -3.78 |
| 6 | Iterative Refinement (Fixer + Revisor) | Sequential | 4 | 5 | -0.52 |
| 7 | Execution-Based Grouping | — (Optimization) | 2 | 2 | N/A |
| 8 | Tournament Selection (RL Selector) | Parallel + Internal | 7 | 8 | -1.82 |

---

## 4. Key Insights

### What Matters Most

The two generators (Operations 4 and 5) are by far the most important components, collectively responsible for ~8.67 EX points in the ablation study. The RL-enhanced Reasoning Generator alone accounts for 4.89 points — the largest single-component contribution. The core insight is that **candidate pool quality and diversity** are the primary drivers of final performance.

### Diminishing Returns of Refinement

Iterative Refinement (Operation 6) contributes only 0.52 EX points despite potentially requiring up to 51 LLM calls. This suggests that well-tuned generators already produce high-quality SQL, and post-hoc correction has limited marginal value. This is the weakest cost-to-benefit ratio in the pipeline.

### Selection as Multiplier

Tournament Selection (Operation 8) is the third most important component (+1.82 over majority voting) but has the highest inference-time complexity. Its value is bounded by the candidate pool quality — it cannot select a correct answer if none exists. The 9.39-point gap between the upper bound (84.29%) and the final result (74.90%) indicates that selection is the primary bottleneck for future improvement.

### The Scaling Hypothesis Validated

The paper's central thesis — that scaling computation at test time is more effective than designing complex heuristics — is supported by the data. The framework achieves +5.54 EX over the strongest single model (Gemini-SQL) by investing in 20-630 LLM calls per question instead of 1-7. The latency cost (10-30 seconds) is deemed acceptable for the target B2B enterprise use case.

### Total Inference Cost Profile

| Stage | LLM Calls | % of Max Total |
|-------|-----------|----------------|
| Task Understanding | 1 | 0.16% |
| Generation | 17 | 2.7% |
| Refinement | 1-51 | 0.16-8.1% |
| Selection | 1-561 | 0.16-89.0% |
| **Total** | **20-630** | **100%** |

Selection dominates the computational budget. The framework's total cost (20-630 calls) is significantly lower than Contextual-SQL (2048 calls) while achieving higher accuracy, demonstrating efficient orchestration.
