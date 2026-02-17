# XiYan-SQL: Method Analysis

---

## 1. General Pipeline

The XiYan-SQL framework follows a three-stage pipeline that transforms a natural language question into an optimal SQL query:

```
Input: (Question Q, Evidence E, Database Schema S, Database D)
  │
  ▼
┌─────────────────────────────────────────────┐
│  Stage 1: Schema Filter                     │
│  ┌───────────────────────────────────────┐   │
│  │ 1a. Keyword Extraction (LLM)         │   │
│  │ 1b. Multi-path Retrieval             │   │
│  │     (embedding similarity + LSH)     │   │
│  │ 1c. Iterative Column Selection (LLM) │   │
│  └───────────────────────────────────────┘   │
│  Output: Schema set S = {S₁, S₂}            │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│  Stage 2: Multiple SQL Generation           │
│  ┌───────────────────────────────────────┐   │
│  │ For each schema Sᵢ (i=1..2):         │   │
│  │   For each generator Mⱼ (j=1..5):    │   │
│  │     Generate SQL candidate lᵢⱼ       │   │
│  │     Execute lᵢⱼ on database D        │   │
│  │     If error → Self-refine (1 retry) │   │
│  └───────────────────────────────────────┘   │
│  Output: Candidate SQL list L (up to 10)     │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│  Stage 3: SQL Selection                     │
│  ┌───────────────────────────────────────┐   │
│  │ 3a. Execute all candidates on DB      │   │
│  │ 3b. Cluster by execution results      │   │
│  │ 3c. If all consistent → pick shortest │   │
│  │ 3d. Else → Candidate Reorganization   │   │
│  │     (inter/intra-group sorting)       │   │
│  │ 3e. Selection Model picks best SQL    │   │
│  └───────────────────────────────────────┘   │
│  Output: Final SQL query l*                  │
└─────────────────────────────────────────────┘
  │
  ▼
Output: Optimal SQL query l*
```

**Data flow summary:** Raw schema → filtered schemas (2 variants) → 10 candidate SQLs → 1 optimal SQL.

---

## 2. Detailed Operation Analysis

---

### 2.1 Operation 1: Keyword Extraction

**Input:** Natural language question Q, evidence E.

**Output:** A set of keywords K = {k₁, k₂, ...} representing entities, values, and concepts relevant to the query.

**Details:** An LLM (GPT-4o) is prompted to extract keywords from the concatenation of the question and any external evidence/hints. These keywords serve as search terms for the subsequent retrieval step, capturing column names, table names, and value mentions that the user implicitly or explicitly references.

**Role in pipeline:** This is the entry point to schema filtering. By distilling the question into focused keywords, the system avoids comparing the entire question against every schema element, enabling more targeted retrieval. It transforms unstructured natural language into discrete search anchors.

**Significance: 4/10**
Keyword extraction is a preparatory step that feeds into retrieval. Its quality affects downstream recall, but it is a relatively standard NLP operation. The paper does not isolate its individual contribution, and errors here can be partially compensated by the iterative column selection step. It is necessary but not a primary driver of the performance gains.

**Complexity: 2/10**
A single LLM call with a short prompt. Input/output tokens are minimal (~414/37 tokens on average). Latency is dominated by API call overhead rather than computation. Constant with respect to database size.

---

### 2.2 Operation 2: Multi-path Retrieval

**Input:** Keywords K, original database schema S (all tables, columns, values), embedding model.

**Output:** A pruned schema S_rtrv = {c_r1, c_r2, ...} containing the top-k most relevant columns and associated values.

**Details:** This operation has two sub-processes:

- **Table/Column retrieval:** For each keyword kᵢ and each column cⱼ, a score is computed as the product of (a) cosine similarity between the keyword embedding and the column metadata embedding, and (b) cosine similarity between the full question+evidence embedding and the table metadata embedding. The top-k columns per keyword are retained.
- **Value retrieval:** Edit distance identifies candidate values similar to keywords. A RoBERTa tokenizer is used for Locality Sensitive Hashing (LSH) to efficiently filter value-related text. An embedding cosine similarity threshold further refines results.

**Role in pipeline:** This is the coarse-grained pruning step that reduces potentially hundreds of columns/tables in large databases down to a manageable set. It ensures that the subsequent LLM-based column selection operates on a feasible input size and that relevant schema elements are not missed. The multi-path design (table-level + column-level + value-level) provides complementary retrieval signals.

**Significance: 5/10**
Schema filtering as a whole contributes ~1.24% EX improvement (Table III ablation). Multi-path retrieval is the foundation of schema filtering — without it, the LLM would need to process the entire schema, which is impractical for large databases. However, the retrieval step alone is a recall-oriented filter; the precision refinement happens in the next step. Its contribution is enabling rather than directly performance-boosting.

**Complexity: 5/10**
Requires computing embeddings for all columns, tables, and values in the database (can be pre-computed and cached). The pairwise similarity computation scales as O(|K| × |C|) for columns and O(|K| × |V|) for values where |V| can be very large. LSH mitigates value search cost. The retrieval phase accounts for ~11.8s of the total ~40.5s pipeline latency, making it the most time-consuming single step.

---

### 2.3 Operation 3: Iterative Column Selection

**Input:** Pruned schema S_rtrv, question Q, evidence E, maximum iterations p_s = 2.

**Output:** A set of filtered schemas S = {S₁, S₂} with different precision-recall trade-offs.

**Details:** An LLM (GPT-4o) is called iteratively to select columns relevant to the question from the pruned schema. In each iteration:
1. The LLM selects relevant columns from the remaining schema → S_slct_i.
2. Primary and foreign keys related to selected columns are identified → P_i.
3. A new schema is formed by unifying all previously selected columns with the current selection.
4. Selected columns (except essential keys) are removed from S_rtrv for the next iteration.

This produces S₁ (higher precision, lower recall) and S₂ (lower precision, higher recall, as it includes S₁ plus additional columns from the second iteration).

**Role in pipeline:** This step introduces schema diversity — a key design principle of XiYan-SQL. By producing two schemas with different precision-recall characteristics, it gives the downstream generators different "views" of the database, increasing the chance that at least one schema contains all necessary information. The iterative removal mechanism ensures the two schemas are meaningfully different.

**Significance: 5/10**
The Schema Filter module as a whole contributes ~1.24% EX improvement. Column selection is the precision-refinement component. Table V shows that schema S₁ alone yields +2.18% over full schema, while S₂ yields +1.40%. The diversity between S₁ and S₂ is essential for the multi-generator approach. The schema filter precision-recall balance is highlighted in the Discussion section as a key empirical insight.

**Complexity: 3/10**
Two LLM calls to GPT-4o with moderate prompt sizes (~3792/63 tokens). Latency is ~6.8s total. The PFKeyIdentifier is a simple graph traversal on the schema. Overall, this step is computationally light relative to its importance.

---

### 2.4 Operation 4: Multi-task Fine-tuning of SQL Generators

**Input:** Training data consisting of question-SQL pairs, augmented with three auxiliary tasks: (a) reverse question inference from SQL, (b) reverse evidence inference from SQL, (c) self-refine task (regenerate SQL from execution feedback).

**Output:** Four fine-tuned SQL generation models (SQLG1–SQLG4) based on Qwen2.5-Coder-32B, each with different generation characteristics.

**Details:** This is a training-time operation (not part of inference). The method involves:

- **Multi-task fine-tuning:** In addition to the standard text-to-SQL task, three auxiliary tasks are jointly trained:
  - *Reverse question inference:* Given SQL + schema, generate possible questions (forces the model to understand SQL-to-NL alignment).
  - *Reverse evidence inference:* Given SQL + question, identify relevant evidence from candidates (strengthens evidence utilization).
  - *Self-refine:* Given a failed SQL + execution error, regenerate corrected SQL (builds error recovery capability).
- **Multi-format SQL training:** Different models are fine-tuned with SQL formatted in different styles:
  - SQLG1: Standard format with multi-task data.
  - SQLG2: Complex/chunked writing patterns (more advanced query structures).
  - SQLG3: Standardized presentation styles.
  - SQLG4: Mixed format combining all variations.

**Role in pipeline:** This is the core innovation of XiYan-SQL. The multi-task approach enhances the fundamental SQL generation quality of each model (single model reaches 66.88% EX, surpassing GPT-4o's 62.65% few-shot). The multi-format approach creates generators with different "preferences" — ensuring the candidate pool has both quality and diversity. This directly enables the multi-generator ensemble strategy.

**Significance: 10/10**
This is the most impactful component of the framework. The ablation (Table III) shows that using only one SFT generator drops performance by 4.04% — the largest single ablation effect. Table VI demonstrates that multi-task fine-tuning improves performance by 2.2–4.5% over standard fine-tuning across all model sizes. The multi-format training creates the diversity needed for the ensemble to work. Without this operation, the entire multi-generator framework would not function effectively.

**Complexity: 8/10**
Fine-tuning Qwen2.5-Coder-32B requires ~45 GPU hours on A100 80G GPUs per model, consuming ~180M tokens. Four models are trained (SQLG1–SQLG4), totaling ~180 GPU hours. This is a one-time training cost but is the most computationally expensive component. However, this cost is amortized over all subsequent inference.

---

### 2.5 Operation 5: Multiple SQL Candidate Generation (Inference)

**Input:** Question Q, evidence E, filtered schema set S = {S₁, S₂}, five generators (SQLG1–SQLG5), database D.

**Output:** Candidate SQL list L = {l₁, ..., l₁₀} (up to 10 candidates).

**Details:** For each of the 2 schemas and each of the 5 generators (4 fine-tuned + 1 ICL-based GPT-4o), a SQL query is generated:
1. Generator produces SQL: l_ij = f_Mj(Q, E, Sᵢ).
2. SQL is executed on the database.
3. If execution produces a syntax error or anomalous values, the generator self-refines once using execution feedback.
4. The final SQL (original or refined) is added to the candidate list.

The five generators are:
- SQLG1: Multi-task trained, standard format (highest individual accuracy: 69.34%).
- SQLG2: Multi-task + complex writing patterns (66.05%).
- SQLG3: Multi-task + standardized styles (68.15%).
- SQLG4: Multi-task + mixed formats (68.50%).
- SQLG5: ICL-based GPT-4o (64.51%).

**Role in pipeline:** This is where the diversity-quality trade-off materializes. The combination of different schemas × different generators creates a 2D exploration of the solution space. Self-refine acts as a quality filter, catching obvious errors. The upper bound of the candidate pool reaches 82.2% at 10 candidates (Figure 4b), far exceeding any single generator. This step produces the raw material from which the selection stage must identify the best answer.

**Significance: 8/10**
The multi-generator ensemble is the defining feature of XiYan-SQL. Figure 4(a) shows that Multi-FT candidates achieve ~2% higher selection accuracy than ICL-Prompt candidates. The upper bound of 82.2% demonstrates significant headroom. Each generator contributes uniquely — even the weakest (SQLG5) produces 2.22% of uniquely correct results (Table XI, R₁). The candidate generation step is critical but depends on the fine-tuning quality (Operation 4) for its effectiveness.

**Complexity: 6/10**
10 sequential or parallel model inference calls. Fine-tuned model inference averages ~582.5 input / 47.8 output tokens with ~2.3s latency each. GPT-4o inference averages ~763.2/54.7 tokens with ~2.8s latency. With self-refine retries, the total generation phase uses ~10 model calls. If parallelized across generators, latency can be reduced significantly. Database execution per candidate adds minimal overhead.

---

### 2.6 Operation 6: Candidate Execution and Clustering

**Input:** Candidate SQL list L = {l₁, ..., l_n}, database D.

**Output:** Execution results R = {r₁, ..., r_n}, clusters C = {C₁, ..., C_m} grouped by result equivalence.

**Details:**
1. Each candidate SQL is executed on the database to obtain its result set.
2. Candidates with execution errors are filtered out.
3. Remaining candidates are grouped into clusters based on execution result equivalence — candidates producing identical results belong to the same cluster.

If all candidates fall into a single cluster (full consensus), the shortest SQL is returned immediately as the final answer — a fast-path optimization that avoids the selection model entirely (~45% of cases).

**Role in pipeline:** This step bridges generation and selection. Clustering by execution results reveals the consistency structure of the candidate pool. The fast-path for unanimous consensus leverages the insight that when all generators agree, the answer is almost certainly correct. For non-unanimous cases, the clustering provides crucial structural information for the reorganization strategy.

**Significance: 6/10**
The ~45% fast-path cases are handled efficiently and correctly here. For the remaining ~55%, clustering provides the foundation for the reorganization strategy that boosts selection performance by ~3% over majority voting (Table VIII). The execution-based grouping is more reliable than syntactic comparison since different SQL forms can produce identical results.

**Complexity: 2/10**
Executing SQL queries on a database is fast (milliseconds per query). Comparing result sets for equality is straightforward. Clustering is O(n²) in the number of candidates but n ≤ 10, so this is negligible. This step adds minimal overhead to the pipeline.

---

### 2.7 Operation 7: Candidate Reorganization

**Input:** Clusters C = {C₁, ..., C_m}, generator performance ordering O.

**Output:** Reorganized candidate list L' with optimized ordering.

**Details:** Two-level sorting is applied:
- **Inter-group sorting:** Clusters are sorted in descending order by size (largest cluster first). This places the most consistent results at the top, leveraging the prior that majority consensus is often correct.
- **Intra-group sorting:** Within each cluster, candidates are sorted by their generator's known performance ranking (best generator first).

Then, based on whether a majority consensus exists (|C₁| ≥ ⌈|L|/2⌉):
- **If majority exists:** All candidates from all clusters are presented in sorted order. The selection model sees the full picture but with the most likely correct answers first.
- **If no majority:** Only one representative candidate per cluster is presented (the shortest SQL from each cluster). This reduces noise and focuses the selection model's attention.

**Role in pipeline:** This step exploits the known positional bias of LLMs (tendency to prefer earlier options) as a feature rather than a bug. By placing the most likely correct candidates first, the reorganization nudges the selection model toward better decisions. The adaptive strategy (full list vs. representatives) manages the cognitive load on the selection model based on the difficulty of the case.

**Significance: 7/10**
Table VIII demonstrates the impact clearly: random ordering (s1) with 10 candidates drops to 68.19% (worse than majority voting at 70.21%), performance-based ordering (s2) achieves 71.84%, and the full reorganization strategy (s3) reaches 73.34%. The ~3% gap between s1 and s3 with 10 candidates is substantial. As candidate count grows, this operation becomes increasingly important — it is what makes scaling up candidates beneficial rather than harmful.

**Complexity: 1/10**
Pure sorting and list manipulation on at most 10 candidates. No model calls, no database access. Computational cost is negligible (microseconds). This is essentially free in terms of compute.

---

### 2.8 Operation 8: SQL Selection via Fine-tuned Model

**Input:** Question Q, unified schema S_un, evidence E, reorganized candidates L', fine-tuned selection model M_c (Qwen2.5-Coder-7B).

**Output:** The optimal SQL query l*.

**Details:** The selection model receives a prompt containing:
- The original question and evidence.
- The union of all filtered schemas (providing full context).
- The reorganized candidate list L'.

The model is trained on contrastive samples constructed from positive (correct SQL) and negative (incorrect SQL) pairs. Training data preparation includes:
- Generating candidates from diverse generators and grouping by execution results to identify correct/incorrect pairs.
- LLM-based controlled modifications of correct queries to create hard negatives.
- De-formalization of SQL to reduce style-based interference.
- Balanced distribution of positive/negative combinations, generator origins, and candidate ordering.

The model outputs a single selection (the index or the SQL itself), requiring only 1 output token on average.

**Role in pipeline:** This is the final decision-maker of the framework. It must distinguish between semantically different SQL candidates that may look syntactically similar. The fine-tuned model outperforms GPT-4o (69.56% vs. 67.47% in independent evaluation, Table IX) despite being much smaller (7B vs. estimated 200B+), demonstrating that task-specific fine-tuning is highly effective for this discrimination task.

**Significance: 8/10**
The SQL selection step contributes ~3.13% EX improvement over majority voting (Table III ablation). Combined with reorganization, it is the second most impactful component after multi-generator construction. The selection model bridges the gap between the candidate pool's upper bound (82.2%) and the final performance (73.34%). Without effective selection, adding more candidates actually hurts performance (as shown by the declining lower bound in Figure 4b).

**Complexity: 2/10**
A single inference call to a 7B model with ~1052 input tokens and 1 output token. Latency is ~0.8s. This is the cheapest model call in the entire pipeline. The small model size also means low deployment cost. Training cost is ~15 GPU hours (one-time).

---

## Summary Table

| # | Operation | EX Impact (%) | Significance (1-10) | Complexity (1-10) |
|---|-----------|:---:|:---:|:---:|
| 1 | Keyword Extraction | — (enabling) | 4 | 2 |
| 2 | Multi-path Retrieval | ~1.24% (shared) | 5 | 5 |
| 3 | Iterative Column Selection | ~1.24% (shared) | 5 | 3 |
| 4 | Multi-task Fine-tuning (training) | ~4.04% | 10 | 8 |
| 5 | Multiple SQL Generation (inference) | ~4.04% (shared w/ #4) | 8 | 6 |
| 6 | Candidate Execution & Clustering | ~3.13% (shared) | 6 | 2 |
| 7 | Candidate Reorganization | ~3.13% (shared w/ #8) | 7 | 1 |
| 8 | SQL Selection Model | ~3.13% (shared w/ #7) | 8 | 2 |

**Key insight:** The highest-impact operations (multi-task fine-tuning and multi-generator generation) are also the most computationally expensive, but their costs are either one-time (training) or parallelizable (inference). The cheapest operation (candidate reorganization) delivers outsized returns by exploiting LLM positional biases. The end-to-end pipeline latency of ~40.5 seconds is dominated by schema filtering (~18.6s) rather than by the novel multi-generator components.
