# Adaptive Hybrid NL2SQL: A Unified Framework for Text-to-SQL

## Design Rationale

This method is constructed by identifying and combining the highest-impact components from four state-of-the-art systems — Agentar-Scale-SQL, Automatic Metadata Extraction, CHASE-SQL, and XiYan-SQL — while eliminating redundancies and low-value operations. The key design principles are:

1. **Automated metadata beats human metadata** (from Automatic Metadata Extraction: profiling metadata alone 61.2% > human metadata 59.6%).
2. **RL-tuned generators are the single highest-impact component** (from Agentar: -4.89 EX ablation, the largest across all four methods).
3. **Multi-task fine-tuning with format diversity produces the best cost-effective generator ensemble** (from XiYan: -4.04% ablation, single model reaches 66.88% surpassing GPT-4o's 62.65%).
4. **Query fixing has outsized returns relative to complexity** (from CHASE-SQL: -3.78% ablation with bounded compute).
5. **Adaptive selection with fast-path consensus dramatically reduces average inference cost** (from XiYan: ~45% of cases resolved without selection model; from Agentar: RL tournament selection +1.82 over majority voting).
6. **Candidate reorganization is nearly free but significantly boosts selection** (from XiYan: +3% over random ordering at zero compute cost).

Components explicitly **excluded** due to low cost-benefit ratio:
- Iterative Refinement / SQL Revisor (Agentar: only -0.52 EX for up to 51 LLM calls — worst ratio in any method).
- Full pairwise tournament on all candidates (Agentar: up to 561 LLM calls — replaced with grouped tournament on representatives).
- Five-variant schema linking with retries (Automatic Metadata: up to 20 LLM calls for schema linking alone — replaced with task-aligned dual-schema approach).

---

## 1. General Pipeline

```
Raw Inputs: (Question Q, Evidence E, Database D)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  OFFLINE PREPROCESSING (one-time per database)                      │
│                                                                     │
│  Database D ──► Statistical Profiling ──► LLM Profile Summarization │
│       │              │                        │                     │
│       │              ▼                        ├──► Short Summaries  │
│       │        Profile Stats                  └──► Long Summaries   │
│       │              │                                  │           │
│       │              ▼                                  ▼           │
│       │     LSH Index on Cell Values          FAISS Semantic Index  │
│       │       (per field)                      (on long summaries)  │
│       │                                                             │
│       ├──► DDL Schema (enriched with summaries, for RL Generator)   │
│       └──► Markdown Schema (enriched with summaries, for FT/ICL)   │
│                                                                     │
│  Training Set ──► Skeleton Masking ──► Example Vector Store         │
│                                                                     │
│  ── Model Training (one-time) ──                                    │
│  RL-GRPO fine-tuning of Reasoning Generator (32B)                   │
│  Multi-task fine-tuning of 2 diverse SQL Generators (32B)           │
│  RL-GRPO fine-tuning of Selection Model (7-8B)                      │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: CONTEXT GROUNDING                                          │
│                                                                     │
│  (Q, E) ──► Keyword Extraction (LLM) ──► LSH Cell Value Retrieval  │
│  Q ──────► Skeleton Masking ──► Few-Shot Example Retrieval          │
│                                                                     │
│  Output: Matched cell values V_m + Retrieved examples X_fs          │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: ADAPTIVE SCHEMA LINKING                                    │
│                                                                     │
│  Q ──► FAISS Retrieval (top-k fields by semantic similarity)        │
│     ──► LLM Column Selection (iteration 1) ──► Schema S₁ (precise) │
│     ──► LLM Column Selection (iteration 2) ──► Schema S₂ (recall)  │
│                                                                     │
│  Output: Dual filtered schemas {S₁, S₂} with enriched metadata     │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: DIVERSE SQL GENERATION                                     │
│                                                                     │
│  ┌────────────────────────┐  ┌────────────────────────┐             │
│  │  RL Reasoning Generator│  │  Multi-task FT Gens    │             │
│  │  (DDL schema, extended │  │  (Markdown schema,     │             │
│  │   CoT, {S₁,S₂} × 2    │  │   {S₁,S₂} × 2 models │             │
│  │   samples = 4 queries) │  │   = 4 queries)         │             │
│  └──────────┬─────────────┘  └──────────┬─────────────┘             │
│             │                           │                           │
│  ┌──────────┴───────────────────────────┘                           │
│  │                                                                  │
│  │  ┌──────────────────────────────┐                                │
│  │  │  ICL Generator (GPT-5 /     │                                │
│  │  │  Gemini-2.5-Pro, S₂ schema, │                                │
│  │  │  few-shot examples, 2-3     │                                │
│  │  │  queries with temp/prompt   │                                │
│  │  │  variation)                 │                                │
│  │  └──────────┬───────────────────┘                                │
│  │             │                                                    │
│  └─────────────┘                                                    │
│             │                                                       │
│             ▼                                                       │
│        10-11 raw SQL candidates                                     │
│             │                                                       │
│             ▼                                                       │
│  ┌──────────────────────────────────────┐                           │
│  │  Query Fixer (up to β=2 iterations)  │                           │
│  │  Execute → if error/empty → LLM fix  │                           │
│  └──────────────────────────────────────┘                           │
│             │                                                       │
│        10-11 refined SQL candidates (C)                              │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: ADAPTIVE SQL SELECTION                                     │
│                                                                     │
│  C ──► Execute all on DB ──► Cluster by execution results           │
│                                                                     │
│  ┌─ Decision Branch ──────────────────────────────────────────────┐ │
│  │                                                                │ │
│  │  IF all candidates in 1 cluster (unanimous):                   │ │
│  │     → FAST PATH: return shortest SQL                           │ │
│  │                                                                │ │
│  │  ELSE:                                                         │ │
│  │     → Select 1 representative per cluster (shortest SQL)       │ │
│  │     → Reorganize: inter-group sort by cluster size (desc),     │ │
│  │       intra-group sort by generator performance ranking        │ │
│  │     → RL Reasoning Selector: pairwise round-robin tournament   │ │
│  │       on representative set C' (m representatives,             │ │
│  │       C(m,2) comparisons)                                      │ │
│  │     → c_final = argmax(W_i)                                    │ │
│  │                                                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  Output: Final SQL query c_final                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Inference Cost Profile (Expected)

| Stage | LLM Calls | Notes |
|-------|-----------|-------|
| Context Grounding | 1 | Keyword extraction (lightweight model) |
| Schema Linking | 2-3 | FAISS retrieval (no LLM) + 2 LLM column selection iterations |
| Generation | 10-11 | 4 RL + 4 multi-task FT + 2-3 ICL |
| Query Fixing | 0-22 | Conditional: only on errored candidates, β=2 max |
| Selection | 0-30 | Fast path: 0; Worst case with ~8 clusters: C(8,2)=28 |
| **Total** | **13-67** | |

Compared to: Agentar (20-630), CHASE-SQL (~480+), Automatic Metadata (~25-35), XiYan (~12-15).

This method targets a **3-5x reduction in average inference cost** compared to Agentar and CHASE-SQL while maintaining higher candidate diversity than XiYan and Automatic Metadata Extraction.

---

## 2. Detailed Operation Analysis

---

### Operation 0: Database Profiling + LLM Summarization (Offline)

#### 2.1 Input, Output, and Details

- **Input:** Raw database D (all tables, columns, and cell values).
- **Output:**
  - Per-field statistical profiles (NULL counts, distinct values, min/max, top-k values, minhash sketches).
  - Per-field short summaries (concise semantic description for schema linking).
  - Per-field long summaries (detailed description with value patterns, formats, statistics for SQL generation context).
  - Two enriched schema representations:
    - **DDL Schema:** CREATE TABLE statements with summary comments and example values (consumed by RL Reasoning Generator).
    - **Markdown Schema:** Tabular format with long summaries, types, primary/foreign keys (consumed by Multi-task FT and ICL generators).

- **Details:** This combines the profiling approach from Automatic Metadata Extraction with the dual schema representation from Agentar-Scale-SQL. First, a single-pass statistical profiling computes per-field metadata (distribution stats, top-k values, minhash sketches). Then, an LLM (e.g., GPT-4o) summarizes each field's profile into short and long natural language descriptions, leveraging world knowledge to infer field semantics (e.g., recognizing "CDS" as "County-District-School" codes). Finally, the summaries are injected into two schema formats optimized for different generator types.

  The key insight from Automatic Metadata Extraction is that LLM-generated summaries from profiling data **outperform human-supplied metadata** (61.2% vs 59.6%), and fusing both yields 63.2%. This operation ensures that all downstream generators operate with richer semantic context than any single prior method provides.

#### 2.2 Role in the General Pipeline

This is the **foundation layer** that elevates every downstream component. The enriched metadata provides:
- Better schema linking (fields have semantic descriptions, not just names).
- Better SQL generation (generators understand what columns mean and what values they contain).
- Better value matching (profile statistics reveal value formats and patterns).

The dual schema representation enables the Diverse Synthesis strategy: RL generators consume DDL (aligned with code pre-training), while FT/ICL generators consume Markdown (aligned with instruction tuning).

#### 2.3 Performance Significance: **8/10**

Automatic Metadata Extraction demonstrated that profiling + LLM summarization alone accounts for a ~11.4% absolute improvement over no metadata (49.8% → 61.2%) and outperforms expensive human annotation. When fused with human metadata, it reaches 63.2%. This is the single largest accuracy driver in that system. In our pipeline, the enriched metadata improves every generator's baseline quality, reducing the number of candidates needed and improving selection accuracy.

#### 2.4 Complexity: **4/10**

- **Profiling:** O(N) single pass per table. Uses standard algorithms (HyperLogLog, top-k heaps, minhash). Fast even for large databases.
- **LLM Summarization:** One LLM call per field (or batched). For BIRD's schemas (~hundreds of fields), this is ~100-500 calls to a lightweight model. This is a one-time offline cost.
- **Schema Formatting:** Deterministic string manipulation. Negligible compute.
- Total offline cost: minutes to low hours depending on database size. Amortized over all queries.

---

### Operation 1: Index Building — Cell Values + Examples + Semantic (Offline)

#### 2.1 Input, Output, and Details

- **Input:**
  - All cell values from database D.
  - Long summaries from Operation 0.
  - Training set question-SQL pairs.
- **Output:**
  - **LSH Index on cell values:** Per-field index using k-shingling + locality-sensitive hashing for approximate string matching. Given a keyword, retrieves fields containing similar values.
  - **FAISS Semantic Index:** Vector index over long field summaries for semantic field retrieval during schema linking.
  - **Example Vector Store:** Embeddings (all-MiniLM-L6-v2) of skeleton-masked training questions, enabling structural similarity retrieval of few-shot examples.

- **Details:** Three complementary indices are built:
  1. The LSH index (from Automatic Metadata / CHASE-SQL) enables robust value matching that handles typos, abbreviations, and partial matches — critical for grounding SQL WHERE clauses in actual database values.
  2. The FAISS index (from Automatic Metadata) enables semantic field retrieval: given a question, find fields whose descriptions are semantically related, even if no keyword overlap exists.
  3. The Example Store (from Agentar / XiYan) indexes training examples by structural skeleton (entity names replaced with placeholders), enabling retrieval of structurally similar solved examples regardless of domain.

#### 2.2 Role in the General Pipeline

These indices serve three distinct downstream consumers:
- LSH → Context Grounding (Step 1): matches question keywords to actual cell values.
- FAISS → Schema Linking (Step 2): retrieves semantically relevant fields as candidates for column selection.
- Example Store → SQL Generation (Step 3): provides structurally similar few-shot demonstrations to the ICL Generator.

Together, they bridge the gap between unstructured natural language and structured database content without requiring expensive LLM calls at inference time.

#### 2.3 Performance Significance: **5/10**

Each index contributes indirectly through the operations it enables:
- LSH-based value retrieval accounts for -2.92% in CHASE-SQL's ablation.
- Few-shot examples are a core enabler of ICL generation (part of -3.78 EX in Agentar).
- FAISS enables focused schema linking which contributes ~2% in Automatic Metadata.
No single index is a primary driver, but removing all three would cripple the pipeline's grounding capabilities.

#### 2.4 Complexity: **3/10**

- LSH index: O(N × F) for N sampled values × F fields. Uses small hash functions — fast batch computation.
- FAISS index: One embedding per field summary (~hundreds of embeddings). ANN index construction is sub-second.
- Example Store: ~10K training examples embedded with a 22M parameter model. Minutes on CPU.
- All are one-time offline costs with sub-linear query-time complexity.

---

### Operation 2: Model Training — RL Reasoning Generator (Offline)

#### 2.1 Input, Output, and Details

- **Input:** BIRD training set (question-SQL pairs), database instances for execution-based reward computation, base model (Qwen2.5-Coder-32B or equivalent).
- **Output:** An RL-enhanced Reasoning Generator capable of extended chain-of-thought SQL generation with execution-optimized reasoning.
- **Details:** The model is fine-tuned using Group Relative Policy Optimization (GRPO) with an execution-based reward signal:
  - R = 1.0 if generated SQL produces results matching ground truth.
  - R = 0.1 if SQL is executable but produces incorrect results.
  - R = 0.0 if SQL has syntax/runtime errors.

  This is adopted directly from Agentar-Scale-SQL, which demonstrated that RL-enhanced reasoning is the **single most impactful component** across all four methods (-4.89 EX ablation impact). The RL training teaches the model to perform extended internal reasoning (chain-of-thought) before producing SQL, significantly improving logical accuracy especially on simple and moderate queries. The model consumes DDL-formatted schemas, aligning with its code-specialized pre-training.

#### 2.2 Role in the General Pipeline

The RL Reasoning Generator is the **depth component** of the generation strategy. It provides the highest per-candidate accuracy through careful step-by-step reasoning. While the other generators explore breadth (multiple models, prompts, temperatures), this generator goes deep with one highly optimized model. In Agentar's analysis, it uniquely solved 12 samples that the ICL generator could not, demonstrating complementary coverage.

#### 2.3 Performance Significance: **10/10**

This is the single most impactful component in the entire pipeline, directly inherited from the strongest component of the strongest method. Agentar's ablation shows -4.89 EX when removed — the largest single-component impact across all four analyzed methods. The RL-enhanced extended reasoning paradigm represents the most significant technical contribution to the NL2SQL field from the analyzed methods.

#### 2.4 Complexity: **8/10**

- **Training:** Requires 32× A100 80GB GPUs for GRPO training. Each rollout requires executing generated SQL against databases for reward computation. This is extremely expensive but is a **one-time offline cost**.
- **Inference:** 4 forward passes through a 32B model (2 schemas × 2 temperature samples) with extended chain-of-thought reasoning. This is significant GPU compute but reduced from Agentar's 8 samples by leveraging the dual-schema strategy for diversity instead of pure temperature variation.

---

### Operation 3: Model Training — Multi-task Fine-tuned Generators (Offline)

#### 2.1 Input, Output, and Details

- **Input:** BIRD training data augmented with three auxiliary tasks, base model (Qwen2.5-Coder-32B).
- **Output:** Two fine-tuned SQL generators (SQLG-A, SQLG-B) with complementary generation characteristics.
- **Details:** Adopted from XiYan-SQL's multi-task training approach, which demonstrated that auxiliary tasks substantially improve SQL generation quality (+2.2–4.5% over standard fine-tuning). The auxiliary tasks are:
  1. **Reverse question inference:** Given SQL + schema, generate the corresponding natural language question. Forces bidirectional understanding of the NL-SQL mapping.
  2. **Reverse evidence inference:** Given SQL + question, identify relevant evidence from candidates. Strengthens evidence utilization.
  3. **Self-refine:** Given failed SQL + execution error, generate corrected SQL. Builds error recovery capability directly into the model.

  Two generators are trained with different SQL formatting styles:
  - **SQLG-A:** Standard SQL format with multi-task training (highest individual accuracy in XiYan: 69.34%).
  - **SQLG-B:** Complex/chunked writing patterns emphasizing advanced query structures (provides structural diversity).

  We reduce from XiYan's 4 generators to 2 because diminishing returns set in after 2-3 generators (XiYan's Figure 4b shows the upper bound curve flattening), and the RL Reasoning Generator already provides a fundamentally different generation paradigm.

#### 2.2 Role in the General Pipeline

These generators form the **breadth component** alongside the RL Generator. While the RL Generator excels through deep reasoning on a single model, the multi-task FT generators excel through superior training that builds bidirectional NL-SQL understanding. The format diversity between SQLG-A and SQLG-B ensures different query structures are explored. Combined with the RL generator, the three fine-tuned models provide a 3-way diversity axis (RL reasoning vs. multi-task standard vs. multi-task complex).

#### 2.3 Performance Significance: **9/10**

XiYan's ablation shows -4.04% when reducing to a single generator — the largest impact in that system. The multi-task training itself accounts for +2.2–4.5% over standard fine-tuning across all model sizes. A single multi-task model (66.88%) already surpasses GPT-4o few-shot (62.65%). In our pipeline, these generators complement the RL generator by providing candidates with different structural biases, increasing the candidate pool's upper bound.

#### 2.4 Complexity: **7/10**

- **Training:** ~45 GPU hours per model on A100 80GB (from XiYan). Two models = ~90 GPU hours total. Significant but less than the RL training and a one-time cost.
- **Inference:** 4 forward passes (2 schemas × 2 generators) through 32B models. Each call averages ~582 input / ~48 output tokens with ~2.3s latency. Parallelizable across generators.

---

### Operation 4: Model Training — RL Selection Model (Offline)

#### 2.1 Input, Output, and Details

- **Input:** Training data of candidate SQL pairs with correctness labels, base model (Qwen2.5-Coder-7B-Instruct).
- **Output:** An RL-enhanced binary selection model that can judge which of two SQL candidates better answers a question.
- **Details:** Combines insights from Agentar (RL-trained selector) and XiYan (contrastive training with hard negatives). The training process:
  1. Generate candidate pools for training questions using all generators.
  2. Execute candidates and identify correct/incorrect pairs.
  3. Construct contrastive training samples: (question, schema, candidate_A, candidate_B, execution_results_A, execution_results_B, label).
  4. Apply GRPO with binary reward: R = 1 if correct candidate selected, R = 0 otherwise.
  5. Include hard negatives via controlled SQL modifications (from XiYan) to teach fine-grained discrimination.
  6. Balance training data across generator origins, candidate orderings, and difficulty levels.

  We use a smaller model (7B) than Agentar's selector (32B) based on XiYan's finding that a fine-tuned 7B model **outperforms GPT-4o** on selection (69.56% vs. 67.47%). The RL training further enhances discrimination beyond what supervised fine-tuning alone achieves.

#### 2.2 Role in the General Pipeline

The Selection Model is the **final decision-maker** that converts candidate diversity into accuracy. It is only invoked when candidates disagree (non-unanimous cases, ~55% of questions per XiYan's data). The pairwise comparison design is specifically chosen because comparing 2 candidates is more tractable than ranking all simultaneously (CHASE-SQL showed that a ranker approach drops -7.5% vs. pairwise).

#### 2.3 Performance Significance: **7/10**

Agentar's ablation shows -1.82 EX when replacing RL tournament with majority voting. CHASE-SQL shows -4.17% when replacing pairwise selection with self-consistency. XiYan shows ~3.13% improvement from selection over majority voting. The selection model's value is multiplicative with generator quality — it can only select a correct answer if one exists in the pool. With our high-quality generator ensemble, the selection model's impact should be at the upper end of these ranges.

#### 2.4 Complexity: **5/10**

- **Training:** ~15-20 GPU hours on A100 for a 7B model (extrapolated from XiYan's 15 hours). Much cheaper than the generator training. One-time cost.
- **Inference:** Each pairwise comparison is a single forward pass through a 7B model. The model produces ~1 output token per comparison. With representative-only tournament (typically 3-8 representatives), this means 3-28 comparisons — far fewer than Agentar's worst case of 561.

---

### Operation 5: Context Grounding (Online — Step 1)

#### 2.1 Input, Output, and Details

- **Input:** Question Q, Evidence E, LSH cell value index, Example Vector Store.
- **Output:**
  - **Matched cell values V_m:** Specific database values matching keywords in the question, with their source table/column information.
  - **Few-shot examples X_fs:** 6-8 structurally similar solved question-SQL pairs from the training set.

- **Details:** This step performs two parallel retrieval operations:
  1. **Keyword Extraction + Cell Retrieval:** A lightweight LLM (e.g., Gemini-2.5-Flash) extracts database literals from Q and E (specific values like "Japan", "1996"). Each keyword queries the LSH index to find matching cell values, returning the top matches with their table.column locations. This grounds the question in actual database content.
  2. **Skeleton Extraction + Example Retrieval:** The question is abstracted into a skeleton (replacing entity names with placeholders). The skeleton embedding queries the Example Vector Store for the 6-8 most structurally similar training examples. These serve as few-shot demonstrations for the ICL Generator.

  This combines Agentar's Task Understanding with CHASE-SQL's value retrieval approach, using a single lightweight LLM call for keyword extraction plus two vector lookups.

#### 2.2 Role in the General Pipeline

Context Grounding connects the abstract question to concrete database content before any SQL generation occurs. The matched cell values prevent generators from hallucinating non-existent values in WHERE/HAVING clauses. The few-shot examples provide the ICL Generator with structural templates for similar query patterns. This step is fast and provides high-value context to all downstream generators.

#### 2.3 Performance Significance: **4/10**

Agentar's ablation shows only -0.45 EX for removing Task Understanding — the smallest component impact. CHASE-SQL shows -2.92% for removing LSH value retrieval. The modest impact suggests that well-tuned generators can often infer correct values from schema context alone, but grounding helps on edge cases with ambiguous value references or typos. The few-shot examples have a larger implicit contribution through their role in ICL generation.

#### 2.4 Complexity: **2/10**

- 1 LLM call to a lightweight model (Gemini-2.5-Flash or similar) for keyword/skeleton extraction.
- 2 vector similarity lookups (sub-millisecond each).
- Total latency: <1 second. This is the cheapest online step.

---

### Operation 6: Adaptive Schema Linking (Online — Step 2)

#### 2.1 Input, Output, and Details

- **Input:** Question Q, Evidence E, FAISS Semantic Index, enriched full schema, LLM.
- **Output:** Two filtered schemas with different precision-recall trade-offs:
  - **S₁ (Precise):** High-precision schema containing only the most confidently relevant fields. Used by generators for focused SQL generation.
  - **S₂ (Recall):** Higher-recall schema containing S₁ plus additional potentially relevant fields. Used as a safety net to ensure no critical fields are missed.

  Both schemas include enriched metadata (long summaries from Operation 0) for included fields.

- **Details:** This combines XiYan's iterative column selection with Automatic Metadata's semantic retrieval:
  1. **Semantic Pre-filtering:** The FAISS index retrieves the top-k fields whose long summaries are most semantically similar to the question. This produces a candidate field set that is manageable for LLM processing. (No LLM call — pure vector retrieval.)
  2. **Iteration 1 — Precise Selection:** An LLM (GPT-4o) receives the candidate fields with their short summaries and the question. It selects the most relevant fields. Primary and foreign keys related to selected fields are automatically included. This produces S₁.
  3. **Iteration 2 — Recall Expansion:** The selected fields from iteration 1 are removed from the candidate set. The LLM processes the remaining fields and selects additional potentially relevant ones. S₂ = S₁ ∪ newly selected fields ∪ associated keys.

  This dual-schema approach (from XiYan) creates structured diversity in the generation stage. S₁ gives generators a clean, focused view; S₂ provides a broader context that catches fields the first iteration missed.

#### 2.2 Role in the General Pipeline

Schema linking is the **critical bottleneck** between metadata and generation. Without it, large schemas overwhelm LLMs (the "lost in the middle" problem). Automatic Metadata Extraction showed that perfect schema linking would yield 69.0% vs. 63.2% with their algorithm — a 5.8% gap indicating massive room for improvement. Our approach uses semantic pre-filtering (no LLM cost) to narrow the candidate set before applying LLM-based selection, reducing the total LLM calls from Automatic Metadata's 5-20 down to 2-3. The dual output feeds into the generation stage's diversity strategy.

#### 2.3 Performance Significance: **6/10**

XiYan's ablation shows ~1.24% from schema filtering. Automatic Metadata shows ~2% from schema linking (61.2% → 63.2%). The gap to perfect linking (5.8% in Automatic Metadata) indicates that schema linking is underexploited across all methods. Our approach combines the best elements: semantic pre-filtering for recall, iterative LLM selection for precision, and dual output for diversity. The enriched metadata (from Operation 0) should improve selection quality beyond what any prior method achieves, as the LLM has richer descriptions to judge relevance.

#### 2.4 Complexity: **3/10**

- 1 FAISS vector retrieval (sub-millisecond).
- 2 LLM calls for column selection (moderate prompt sizes: ~3,800 input / ~63 output tokens each based on XiYan's measurements).
- 1 optional LLM call if initial retrieval is insufficient.
- Total latency: ~7-10 seconds. Significantly cheaper than Automatic Metadata's 5-variant approach (~20 calls) or CHASE-SQL's column selection.

---

### Operation 7: Diverse SQL Generation (Online — Step 3)

#### 2.1 Input, Output, and Details

- **Input:** Question Q, Evidence E, Matched cell values V_m, Few-shot examples X_fs, Dual schemas {S₁, S₂}, three generator types.
- **Output:** 10-11 candidate SQL queries.
- **Details:** Three generator types produce candidates in parallel:

  **Generator A — RL Reasoning Generator (4 candidates):**
  - Consumes DDL-formatted schemas enriched with metadata summaries.
  - Generates 2 candidates per schema (S₁ and S₂) with temperature variation.
  - Each generation involves extended chain-of-thought reasoning before SQL output.
  - Total: 4 candidates emphasizing logical depth and reasoning accuracy.

  **Generator B — Multi-task Fine-tuned Generators (4 candidates):**
  - SQLG-A generates 1 candidate on S₁ and 1 on S₂ (standard format).
  - SQLG-B generates 1 candidate on S₁ and 1 on S₂ (complex format).
  - Consumes Markdown-formatted schemas with long summaries.
  - Total: 4 candidates emphasizing format diversity and multi-task understanding.

  **Generator C — ICL Generator (2-3 candidates):**
  - Uses a frontier proprietary model (GPT-5 or Gemini-2.5-Pro).
  - Consumes Markdown schema (S₂ for maximum recall) with few-shot examples X_fs.
  - Generates 2-3 candidates with prompt variation (direct prompting vs. chain-of-thought) and temperature variation.
  - Provides a fundamentally different model family for maximum candidate diversity.

  **Diversity is achieved across five dimensions:**
  1. Model architecture (RL-tuned 32B vs. Multi-task FT 32B vs. proprietary frontier model).
  2. Schema representation (DDL vs. Markdown).
  3. Schema scope (S₁ precise vs. S₂ recall).
  4. Prompting strategy (extended reasoning vs. multi-task vs. few-shot ICL).
  5. Temperature variation (within each generator type).

#### 2.2 Role in the General Pipeline

Generation is where the question is actually translated to SQL. All prior operations (profiling, indexing, grounding, schema linking) serve to prepare optimal context for this step. The multi-generator strategy is the defining feature of all four analyzed methods — the key insight is that **candidate pool quality and diversity** are the primary drivers of final performance. By combining the highest-impact generator types from the strongest methods (RL reasoning from Agentar, multi-task FT from XiYan, ICL diversity from Agentar/CHASE-SQL), we target an upper bound exceeding any individual method.

Expected upper bounds (extrapolated from individual methods):
- Agentar: 84.29% from 17 candidates
- XiYan: 82.2% from 10 candidates
- CHASE-SQL: 82.79% from 21 candidates
- **Our target: ~85%+ from 10-11 candidates** (due to combining fundamentally different generator paradigms rather than variations within one paradigm).

#### 2.3 Performance Significance: **10/10**

The generators collectively represent the most critical pipeline component across all four methods:
- Agentar: RL Generator (-4.89 EX) + ICL Generator (-3.78 EX) = -8.67 EX total.
- XiYan: Multi-generator (-4.04 EX).
- CHASE-SQL: Combined generators form the 82.79% upper bound.

Our approach uniquely combines the two strongest generator paradigms (RL reasoning and multi-task FT) which have never been used together. The complementary strengths (RL excels at logical reasoning on simple/moderate queries; multi-task FT excels at schema understanding; ICL excels at creative/challenging queries) should produce a higher upper bound with fewer total candidates.

#### 2.4 Complexity: **7/10**

- 4 forward passes through a 32B RL model with extended reasoning (most expensive per-call).
- 4 forward passes through 32B multi-task FT models (moderate cost per-call).
- 2-3 API calls to frontier models (fast latency but monetary cost).
- Total: 10-11 LLM calls, parallelizable into 3 batches (one per generator type).
- Wall-clock latency: ~5-10 seconds if parallelized (dominated by the RL generator's extended reasoning).
- This is more efficient than Agentar (17 generation calls) and CHASE-SQL (~24 generation calls including synthetic example generation), while maintaining comparable diversity.

---

### Operation 8: Query Fixer (Online — Step 3, Post-generation)

#### 2.1 Input, Output, and Details

- **Input:** Each of the 10-11 candidate SQL queries, their execution results (success, error, or empty result), database schema, original question, evidence.
- **Output:** Corrected SQL candidates (replacing those that had errors).
- **Details:** Adopted from CHASE-SQL's query fixer, which showed the second-largest single-component impact (-3.78% when removed):
  1. Each candidate is executed against the database.
  2. If execution produces a **syntax error** or **empty result set**, the candidate enters a fixing loop.
  3. An LLM (Gemini-2.5-Flash for speed) receives: the original SQL, the error message or "empty result" flag, the schema, question, and evidence.
  4. The LLM performs self-reflection: analyzes why the query failed and produces a corrected version.
  5. Maximum β=2 iterations (reduced from CHASE-SQL's β=3 based on diminishing returns — most fixes succeed on the first attempt).

  Candidates that execute successfully with non-empty results pass through unchanged.

#### 2.2 Role in the General Pipeline

The Query Fixer is a **quality amplifier** positioned between generation and selection. Its primary value is converting non-executable candidates into executable ones — a candidate with a minor syntax error contains the right logic but would be discarded without fixing. By rescuing these candidates, it increases the effective size and quality of the candidate pool presented to selection. CHASE-SQL demonstrated that fixing consistently improves each generator by ~2% across all types.

#### 2.3 Performance Significance: **6/10**

CHASE-SQL's ablation shows -3.78% when removing the fixer — the second-largest impact. In contrast, Agentar's more complex Iterative Refinement (which includes both a fixer and a semantic revisor) only contributes -0.52 EX. The difference is explained by scope: CHASE-SQL's fixer operates on all 21 candidates with simple error correction, while Agentar's refinement attempts deeper semantic revision on a subset. Our approach follows CHASE-SQL's effective-but-bounded strategy. With our high-quality generators (RL-tuned and multi-task FT), fewer candidates should need fixing, but the fixer still provides valuable rescue for edge cases.

#### 2.4 Complexity: **4/10**

- Conditional LLM calls: only for candidates with errors or empty results.
- Best case: 0 LLM calls (all candidates execute successfully).
- Worst case: 10-11 candidates × 2 iterations = 22 LLM calls (unlikely with well-tuned generators).
- Expected average: 2-5 LLM calls per question (based on typical error rates of ~20-30% for individual candidates).
- Uses a lightweight model (Gemini-2.5-Flash) for speed.
- Database execution per candidate adds negligible overhead (milliseconds).

---

### Operation 9: Adaptive SQL Selection (Online — Step 4)

#### 2.1 Input, Output, and Details

- **Input:** 10-11 refined SQL candidates C, database D, RL Selection Model, generator performance rankings.
- **Output:** Single final SQL query c_final.
- **Details:** This is a multi-phase selection process combining XiYan's adaptive strategy with Agentar's RL tournament:

  **Phase 1 — Execution and Clustering:**
  Execute all candidates on D. Group by execution result equivalence. Let m = number of distinct result groups.

  **Phase 2 — Decision Branch:**

  - **If m = 1 (unanimous agreement, ~45% of cases per XiYan):**
    → **Fast Path:** Return the shortest SQL from the single cluster. No selection model invoked. Rationale: when all generators using different schemas, models, and prompting strategies agree, the answer is almost certainly correct. This saves 100% of selection compute for nearly half of all questions.

  - **If m ≥ 2 (disagreement, ~55% of cases):**
    1. **Representative Selection:** Pick one representative from each cluster (shortest SQL within each group). This reduces the candidate set from 10-11 to m representatives (typically 3-8).
    2. **Candidate Reorganization (from XiYan):** Sort representatives in two levels:
       - Inter-group: by cluster size descending (largest cluster first — majority signal).
       - Intra-group: by generator performance ranking (best generator first).
       This exploits LLM positional bias as a feature (XiYan showed +3% from reorganization at zero compute cost).
    3. **RL Tournament Selection (from Agentar):** Pairwise round-robin tournament on the representative set. For each pair (c_i, c_j):
       - The RL Selection Model evaluates which candidate better answers Q, given the schema, matched values, and both candidates' execution results.
       - The winner receives 1 point (W_i incremented).
       - Reorganization order is used as tiebreaker in the RL model's prompt (presenting the higher-ranked candidate first).
    4. **Final Selection:** c_final = argmax(W_i). Ties broken by cluster size, then by generator performance ranking.

  **Phase 3 — Output:**
  Return c_final as the final SQL query.

#### 2.2 Role in the General Pipeline

Selection is the **conversion mechanism** that transforms candidate diversity into accuracy. Without effective selection, more candidates can actually **hurt** performance (XiYan Figure 4b: lower bound decreases with more candidates). Our adaptive approach addresses the core tension: spending compute only when it matters.

The fast path handles the easy ~45% of questions with zero selection cost. For the hard ~55%, the representative-based tournament reduces Agentar's worst-case 561 comparisons to at most C(8,2) = 28, while the reorganization bias provides a "free" accuracy boost that reduces the burden on the RL selector.

This operation closes the gap between the candidate pool's upper bound (~85%) and the final accuracy. Agentar showed that tournament selection achieves +1.82 over majority voting, and our approach augments this with reorganization (+3% from XiYan) and the fast path (eliminating unnecessary compute).

#### 2.3 Performance Significance: **8/10**

This is the second most critical online component after generation. Combined evidence:
- Agentar: Tournament selection +1.82 EX over majority voting.
- CHASE-SQL: Pairwise selection +4.17% over self-consistency.
- XiYan: Selection + reorganization +3.13% over majority voting.

Our adaptive approach should capture the best of all three: the RL training quality of Agentar's selector, the reorganization bias of XiYan, and the computational efficiency of the fast path. The key insight is that selection quality is **bounded by the candidate pool** — with our higher-quality pool (from combining RL + multi-task FT + ICL generators), the selection model has more correct candidates to identify.

#### 2.4 Complexity: **5/10**

- **Fast path (~45% of questions):** 0 LLM calls. Just SQL execution + result comparison. Negligible compute.
- **Tournament path (~55% of questions):**
  - SQL execution for all candidates: milliseconds.
  - Clustering and reorganization: negligible compute.
  - RL tournament on m representatives: C(m,2) pairwise comparisons.
    - Best case (m=2): 1 comparison.
    - Typical case (m=4-5): 6-10 comparisons.
    - Worst case (m=8): 28 comparisons.
  - Each comparison: 1 forward pass through a 7B model (~1 output token). Fast inference.
- **Expected average across all questions:** ~0.55 × 8 ≈ 4-5 LLM calls to the 7B model.
- This is dramatically cheaper than Agentar (1-561 calls to a 32B model) and CHASE-SQL (420 calls to Gemini Flash).

---

## Summary Table

| # | Operation | Phase | Significance (1-10) | Complexity (1-10) | Key Source Methods |
|---|-----------|-------|:---:|:---:|---|
| 0 | DB Profiling + LLM Summarization | Offline | 8 | 4 | Automatic Metadata, Agentar |
| 1 | Index Building (LSH + FAISS + Examples) | Offline | 5 | 3 | Automatic Metadata, CHASE-SQL, Agentar |
| 2 | RL Reasoning Generator Training | Offline | 10 | 8 | Agentar |
| 3 | Multi-task FT Generator Training | Offline | 9 | 7 | XiYan |
| 4 | RL Selection Model Training | Offline | 7 | 5 | Agentar, XiYan |
| 5 | Context Grounding | Online | 4 | 2 | Agentar, CHASE-SQL |
| 6 | Adaptive Schema Linking | Online | 6 | 3 | XiYan, Automatic Metadata |
| 7 | Diverse SQL Generation | Online | 10 | 7 | Agentar, XiYan, CHASE-SQL |
| 8 | Query Fixer | Online | 6 | 4 | CHASE-SQL |
| 9 | Adaptive SQL Selection | Online | 8 | 5 | Agentar, XiYan, CHASE-SQL |

## Expected Performance Analysis

| Metric | Agentar | Auto Metadata | CHASE-SQL | XiYan | **Ours (Expected)** |
|--------|:---:|:---:|:---:|:---:|:---:|
| Candidate Upper Bound | 84.29% | N/A | 82.79% | 82.2% | **~85-87%** |
| Final EX Accuracy | 74.90% | 63.2%* | 73.01% | 73.34% | **~76-78%** |
| Avg. Online LLM Calls | ~170 | ~25 | ~480 | ~12 | **~20-35** |
| Max Online LLM Calls | 630 | ~35 | ~500+ | ~15 | **~67** |
| Generators Count | 2 types | 1 | 3 types | 5 models | **3 types (6 models)** |
| Selection Method | RL Tournament (full) | Majority Vote | Pairwise (all) | FT Model + Reorg | **Adaptive RL Tournament** |

*Auto Metadata accuracy is on MiniDev (500 questions) with GPT-4o; others are on BIRD dev (1534 questions) with specialized models. Not directly comparable but included for context.

### Why This Method Should Outperform Each Individual Method

1. **vs. Agentar:** We inherit its strongest component (RL Reasoning Generator) but add multi-task FT generators (which Agentar lacks), enriched metadata (Agentar uses raw schemas), and adaptive selection (reducing 20-630 calls to 13-67 while maintaining quality).

2. **vs. Automatic Metadata Extraction:** We inherit its strongest innovation (profiling + LLM summarization) and combine it with far superior generation (RL + multi-task FT vs. single GPT-4o) and selection (RL tournament vs. simple majority voting).

3. **vs. CHASE-SQL:** We inherit its query fixer and pairwise selection design but replace its generation strategy (3 prompting variants of one model) with fundamentally different model architectures (RL + multi-task FT + ICL), producing higher-quality and more diverse candidates. We also reduce selection cost by operating on cluster representatives.

4. **vs. XiYan:** We inherit its multi-task training and reorganization strategy but add RL-enhanced generation and selection (which XiYan lacks), enriched metadata (which XiYan lacks), and a more powerful selection mechanism (RL tournament vs. single-pass selection model).
