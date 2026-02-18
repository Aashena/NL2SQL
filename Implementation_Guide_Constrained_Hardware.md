# Implementation Guide: Adaptive Hybrid NL2SQL on Constrained Hardware

## System Specifications

| Resource | Available |
|----------|-----------|
| GPU | Apple M2 (20GB available VRAM, unified memory) |
| Closed-Source LLM Access | Gemini (Flash/Pro), Claude (Haiku/Sonnet/Opus) |
| Local Model Capability | Up to ~8B parameter models (quantized) via MLX/llama.cpp |
| Training Capability | QLoRA fine-tuning of ≤8B models; no full fine-tuning of 32B models |

## Key Constraints Summary

The M2 with 20GB VRAM fundamentally limits local model inference to ~8B parameters (4-bit quantized) and rules out training 32B models locally. This forces a **hybrid strategy**: use API-based models (Gemini, Claude) for generation and use locally fine-tuned small models or API calls where the plan specifies fine-tuned 32B models.

---

## Operation 0: Database Profiling + LLM Summarization (Offline)

### Ideal Resources
- Any CPU for statistical profiling (trivially parallelizable).
- A moderate LLM (GPT-4o-class) for field summarization: ~100–500 API calls per database.
- Storage for profiles and summaries.

### Implementation on Our System
This operation is **fully feasible** with no compromises.

- **Statistical Profiling:** Run entirely on CPU using Python (pandas, datasketch for minhash). The M2 CPU is more than sufficient for single-pass profiling of BIRD-scale databases.
- **LLM Summarization:** Use **Gemini-2.5-Flash** or **Claude Haiku** via API. Both are cost-effective and fast for field-level summarization. At ~100–500 calls for BIRD schemas, the API cost is negligible (<$1).
- **Schema Formatting:** Pure string manipulation — no GPU needed.

### Conditions & Adaptations
- Batch field summaries (5–10 fields per API call) to reduce total calls from ~500 to ~50–100.
- Cache all summaries to disk as JSON for reuse across experiments.
- Use structured output (JSON mode) in API calls to ensure parseable summaries.

### Performance Impact: **None (0% compromise)**
This operation runs identically to the ideal setup. API-based summarization matches or exceeds what a locally-hosted model would produce. The one-time offline cost is trivial.

### Verdict: **Fully beneficial — implement as designed.**

---

## Operation 1: Index Building — Cell Values + Examples + Semantic (Offline)

### Ideal Resources
- CPU + RAM for LSH index construction (datasketch library).
- A small embedding model (all-MiniLM-L6-v2, 22M params) for example and summary embeddings.
- FAISS library for ANN index construction.
- Disk storage for indices.

### Implementation on Our System
This operation is **fully feasible** with no compromises.

- **LSH Index:** Built entirely on CPU using `datasketch`. BIRD databases are small enough to index in minutes.
- **FAISS Semantic Index:** Use `sentence-transformers` with `all-MiniLM-L6-v2` (22M params — runs instantly on M2). Build a FAISS flat or IVF index over field summary embeddings. With ~hundreds of embeddings, even a flat index provides sub-millisecond retrieval.
- **Example Vector Store:** Embed ~10K training examples with the same model. Use FAISS or a simple cosine similarity search. This takes minutes on CPU.

### Conditions & Adaptations
- Use `faiss-cpu` (no GPU FAISS needed for this scale).
- Store indices as serialized files (`.faiss`, `.pkl`) for fast loading at inference time.
- Skeleton masking can be implemented with simple regex/NLP (spaCy NER to mask entities).

### Performance Impact: **None (0% compromise)**
All indices are identical to what would be built on a high-end system. The embedding model and index structures are deterministic.

### Verdict: **Fully beneficial — implement as designed.**

---

## Operation 2: RL Reasoning Generator Training (Offline)

### Ideal Resources
- **32× A100 80GB GPUs** for GRPO training of a 32B parameter model.
- Database instances for execution-based reward computation during RL rollouts.
- Weeks of training time.
- This is the most compute-intensive operation in the entire pipeline.

### Implementation on Our System
**Direct implementation is impossible.** A 32B model cannot fit in 20GB VRAM even at 4-bit quantization (~18GB for weights alone, with no room for activations/optimizer states). GRPO training requires full forward+backward passes plus multiple rollouts.

**Adaptation strategies (choose one or combine):**

1. **Replace with API-based reasoning model (Recommended Primary):**
   - Use **Claude Sonnet/Opus** or **Gemini-2.5-Pro** with extended thinking/chain-of-thought prompting as the "reasoning generator."
   - These models already have strong reasoning capabilities without RL fine-tuning.
   - Generate 4 candidates (2 schemas × 2 temperature settings) via API calls.
   - Cost: ~$0.02–0.10 per question for 4 calls.

2. **Fine-tune a smaller model locally with QLoRA (Supplementary):**
   - Use **Qwen2.5-Coder-7B** (4-bit quantized, ~4GB VRAM) with QLoRA fine-tuning via `mlx-lm` or `unsloth`.
   - Train on BIRD training data with standard supervised fine-tuning (SFT) on question→CoT→SQL pairs.
   - Skip RL/GRPO (requires too much compute for rollouts + reward). Instead, use **rejection sampling fine-tuning (RFT):** generate multiple outputs, keep only those that execute correctly, and fine-tune on those.
   - QLoRA on 7B with MLX: feasible on M2 with batch_size=1, ~2–4 hours for a few epochs.

3. **Use a cloud-hosted fine-tuned model:**
   - Fine-tune via **Gemini fine-tuning API** or **OpenAI fine-tuning API** on your training data.
   - This preserves the fine-tuning benefit without local compute, but you lose RL-based training.

### Conditions & Adaptations
- The API-based approach requires crafting strong system prompts that replicate extended reasoning behavior (step-by-step schema analysis, join path reasoning, value matching).
- If using a local 7B model, expect lower per-candidate accuracy. Compensate by generating more candidates (6 instead of 4) to maintain pool diversity.
- Include matched cell values and evidence directly in the prompt to compensate for the model not having learned to extract them through RL.

### Performance Impact: **Significant (estimated 3–5% EX reduction)**
- The RL Reasoning Generator is the single most impactful component (-4.89 EX in ablation).
- API models (Claude Opus, Gemini-2.5-Pro) are strong reasoners but lack the RL-trained execution optimization. Expected gap: ~2–3% per-candidate accuracy.
- A local 7B SFT model would have a larger gap (~5–7% per-candidate), but this is partially recovered through more candidates and the selection mechanism.
- The multi-generator ensemble design means that even a weaker "reasoning generator" still contributes unique candidates that improve the pool's upper bound.

### Verdict: **Still highly beneficial.** Even degraded, this generator slot provides critical candidate diversity. Use API models as primary, optionally supplement with a local 7B SFT/RFT model.

---

## Operation 3: Multi-task Fine-tuned Generators Training (Offline)

### Ideal Resources
- **~90 GPU hours on A100 80GB** (45 hours per model × 2 models).
- 32B parameter base model (Qwen2.5-Coder-32B).
- Multi-task training data (reverse question inference, reverse evidence inference, self-refine).

### Implementation on Our System
**Direct 32B training is impossible** (same VRAM limitation as Operation 2).

**Adaptation strategies:**

1. **Fine-tune two 7B models locally with QLoRA (Recommended):**
   - Use **Qwen2.5-Coder-7B** as the base model.
   - Implement the same multi-task training approach but at 7B scale:
     - Task 1: NL→SQL (primary)
     - Task 2: SQL→NL (reverse question inference)
     - Task 3: SQL+NL→Evidence (reverse evidence inference)
     - Task 4: Failed SQL+Error→Corrected SQL (self-refine)
   - Train two variants:
     - **SQLG-A:** Standard SQL formatting, multi-task trained.
     - **SQLG-B:** Complex SQL patterns (CTEs, window functions), multi-task trained.
   - QLoRA fine-tuning of 7B on M2: ~3–6 hours per model with MLX, feasible at batch_size=1–2.

2. **Use API models with specialized prompts:**
   - Use **Gemini-2.5-Flash** (cost-effective, fast) with two different system prompts:
     - Prompt A: Standard SQL generation with schema focus.
     - Prompt B: Complex SQL patterns with emphasis on advanced constructs.
   - This loses the multi-task training benefit but gains from the model's broader pre-training.

3. **Hybrid approach (Best balance):**
   - Train one local 7B multi-task model (SQLG-A) for speed and cost efficiency.
   - Use one API model (Gemini-2.5-Flash with specialized prompt) as SQLG-B for quality.

### Conditions & Adaptations
- For local training, prepare multi-task data in a chat format compatible with the base model's template.
- Use `mlx-lm` for M2-optimized training and inference.
- Quantize trained models to 4-bit for inference (~4GB VRAM per model).
- If running both local models, load them sequentially (not in parallel) to stay within VRAM limits.

### Performance Impact: **Moderate (estimated 2–3% EX reduction)**
- XiYan showed that multi-task training on a 7B model still achieves competitive results (their smallest model with multi-task training outperformed larger models with standard training).
- The 7B→32B scale reduction loses ~2–3% per-candidate accuracy based on XiYan's scaling analysis.
- Multi-task training at 7B still provides the bidirectional NL-SQL understanding benefit (+2–4% over standard SFT even at small scale).
- The hybrid approach (one local + one API) provides genuine architectural diversity.

### Verdict: **Still beneficial.** Multi-task training at 7B scale retains most of the methodology's value. The hybrid local+API approach is recommended.

---

## Operation 4: RL Selection Model Training (Offline)

### Ideal Resources
- **~15–20 GPU hours on A100** for a 7B model with GRPO.
- Candidate pools with correctness labels for training.
- Contrastive training samples with hard negatives.

### Implementation on Our System
**Partially feasible.** The 7B model fits in 20GB VRAM, but RL (GRPO) training is extremely memory-intensive due to multiple rollouts and KL divergence computation.

**Adaptation strategies:**

1. **Supervised fine-tuning (SFT) instead of RL (Recommended):**
   - Fine-tune **Qwen2.5-Coder-7B** with QLoRA on binary classification: given (question, schema, SQL_A, SQL_B, results_A, results_B), predict which is correct.
   - Generate training data by running your generators on the training set, executing results, and labeling.
   - Include hard negatives by making small SQL modifications (column swaps, condition changes) to correct queries.
   - QLoRA SFT on 7B: ~4–8 hours on M2.

2. **API-based selection (Fallback):**
   - Use **Claude Sonnet** or **Gemini-2.5-Pro** for pairwise comparison via API calls.
   - These models are strong at comparative reasoning without fine-tuning.
   - Higher per-query cost but zero training cost.

3. **Hybrid approach:**
   - Train a local 7B SFT selector for the pairwise tournament.
   - Use an API model as a tiebreaker when the local model's confidence is low.

### Conditions & Adaptations
- For SFT training, create balanced datasets: equal correct/incorrect pairs, varied candidate orderings, diverse question types.
- Include execution results in the selection prompt (this was shown to significantly help in all methods).
- If using API-based selection, the per-question cost increases by ~$0.01–0.05 for the tournament phase. This is acceptable for research/development but may matter at scale.

### Performance Impact: **Minor (estimated 1–2% EX reduction)**
- SFT vs. RL for selection: XiYan showed that a supervised fine-tuned 7B selector already outperforms GPT-4o (69.56% vs. 67.47%). The RL enhancement in Agentar adds +1.82 EX over majority voting, but much of this comes from the pairwise format itself, not the RL training.
- API-based selection may actually outperform a locally trained selector for difficult cases, partially offsetting the RL loss.
- The adaptive strategy (fast path for unanimous cases) still eliminates ~45% of selection compute regardless of selector quality.

### Verdict: **Still beneficial.** SFT-trained 7B selector captures most of the value. API-based selection is a strong fallback.

---

## Operation 5: Context Grounding (Online — Step 1)

### Ideal Resources
- 1 lightweight LLM call (Gemini-2.5-Flash or similar).
- LSH index (pre-built, in-memory).
- Example Vector Store (pre-built, in-memory).

### Implementation on Our System
**Fully feasible** with no compromises.

- **Keyword Extraction:** Use **Gemini-2.5-Flash** or **Claude Haiku** via API. Single call, <1 second.
- **LSH Cell Retrieval:** Pure CPU computation against the pre-built index. Sub-millisecond.
- **Skeleton Masking:** Local NLP processing (spaCy or simple regex). Negligible compute.
- **Example Retrieval:** Vector similarity search against pre-built FAISS index. Sub-millisecond.

### Conditions & Adaptations
- None required. This operation is inherently lightweight.
- Optionally, keyword extraction could be done locally with a small model (e.g., Phi-3-mini) to save API costs during development/debugging.

### Performance Impact: **None (0% compromise)**
Identical to ideal implementation.

### Verdict: **Fully beneficial — implement as designed.**

---

## Operation 6: Adaptive Schema Linking (Online — Step 2)

### Ideal Resources
- FAISS index (pre-built, in-memory).
- 2–3 LLM calls to a strong model (GPT-4o-class) for column selection.

### Implementation on Our System
**Fully feasible** with no compromises.

- **FAISS Retrieval:** CPU-based vector search. Sub-millisecond.
- **LLM Column Selection:** Use **Claude Sonnet** or **Gemini-2.5-Pro** for the two column selection iterations. These models are strong at structured reasoning over tabular metadata.
- Alternative: Use **Gemini-2.5-Flash** or **Claude Haiku** to reduce cost (column selection is a relatively straightforward task that lighter models handle well).

### Conditions & Adaptations
- Use structured output (JSON mode) to ensure clean column selection results.
- Provide enriched field summaries (from Operation 0) in the prompt to maximize selection quality.
- Consider caching schema linking results for repeated/similar questions during development.

### Performance Impact: **None to marginal (0–0.5% compromise)**
Claude Sonnet and Gemini-2.5-Pro are comparable to GPT-4o for this task. If using lighter models (Haiku/Flash), there may be a marginal quality reduction on ambiguous column selections, but the dual-schema strategy (precise + recall) provides a safety net.

### Verdict: **Fully beneficial — implement as designed.**

---

## Operation 7: Diverse SQL Generation (Online — Step 3)

### Ideal Resources
- 32B RL Reasoning Generator (4 candidates) — GPU with ≥40GB VRAM.
- 32B Multi-task FT Generators ×2 (4 candidates) — GPU with ≥40GB VRAM.
- Frontier API model (2–3 candidates) — API access.
- Total: 10–11 candidates from 3 generator types.

### Implementation on Our System
**Requires significant adaptation** to the generator lineup.

**Adapted Generator Configuration:**

| Generator | Original Plan | Adapted Plan | Candidates |
|-----------|--------------|--------------|:---:|
| A: Reasoning | 32B RL-tuned local | Claude Opus/Sonnet (extended thinking) | 4 |
| B1: Multi-task FT | 32B multi-task local | Local 7B multi-task QLoRA model | 2 |
| B2: Multi-task FT | 32B multi-task local | Gemini-2.5-Flash (specialized prompt) | 2 |
| C: ICL | GPT-5 / Gemini-2.5-Pro | Gemini-2.5-Pro (few-shot) | 2–3 |
| **Total** | | | **10–11** |

**Detailed adaptation:**

1. **Generator A — Reasoning (API-based):**
   - Use **Claude Sonnet with extended thinking** or **Gemini-2.5-Pro** with chain-of-thought prompting.
   - Generate 4 candidates: 2 schemas × 2 temperature settings (or 2 schemas × {with/without extended thinking}).
   - DDL schema format in prompt to encourage code-style reasoning.

2. **Generator B1 — Multi-task FT (Local 7B):**
   - Load the locally fine-tuned 7B multi-task model (from Operation 3) via MLX.
   - Generate 2 candidates: 1 per schema (S₁ and S₂).
   - Markdown schema format.
   - Inference: ~2–4 seconds per candidate on M2.

3. **Generator B2 — Diverse FT Substitute (API-based):**
   - Use **Gemini-2.5-Flash** with a specialized prompt emphasizing complex SQL patterns (CTEs, window functions, subqueries).
   - Generate 2 candidates: 1 per schema.
   - This replaces the second local multi-task model with an API call, providing genuine model diversity.

4. **Generator C — ICL (API-based):**
   - Use **Gemini-2.5-Pro** with few-shot examples from the Example Store.
   - Generate 2–3 candidates with prompt variation.

### Conditions & Adaptations
- Run API calls in parallel (asyncio/concurrent.futures) to minimize wall-clock latency.
- Run local 7B inference sequentially (one model at a time to stay within VRAM).
- Total API cost per question: ~$0.05–0.15 (dominated by Claude Sonnet/Opus calls).
- Include matched cell values, evidence, and enriched metadata in all prompts to compensate for the lack of RL-trained value extraction.

### Performance Impact: **Moderate (estimated 2–4% reduction in candidate pool upper bound)**
- The candidate pool upper bound drops from ~85–87% to ~82–84% due to:
  - API models lack RL-trained execution optimization (~1–2% per-candidate accuracy gap).
  - Local 7B model has lower individual accuracy than 32B (~2–3% gap).
- However, the **diversity across genuinely different model families** (Claude, Gemini, local Qwen) may partially compensate — different model families make different errors.
- The 5-dimension diversity strategy is preserved: model architecture (3 families), schema format (DDL/Markdown), schema scope (S₁/S₂), prompting (reasoning/standard/ICL), temperature.

### Verdict: **Still highly beneficial.** The multi-generator strategy remains the core driver of performance. API models are strong baselines, and the diversity across model families is arguably greater than the original plan's diversity across fine-tuned variants of one architecture.

---

## Operation 8: Query Fixer (Online — Step 3, Post-generation)

### Ideal Resources
- A lightweight LLM (Gemini-2.5-Flash) for fix attempts.
- Database access for execution verification.
- 0–22 conditional LLM calls.

### Implementation on Our System
**Fully feasible** with no compromises.

- **SQL Execution:** Run against SQLite databases locally. Negligible compute.
- **Fix LLM Calls:** Use **Gemini-2.5-Flash** or **Claude Haiku** for error analysis and SQL correction.
- Maximum 22 calls (rare worst case), expected 2–5 calls per question.

### Conditions & Adaptations
- None required. This operation is inherently lightweight and API-friendly.
- Optionally, use the local 7B model for fixing (since it was trained with the self-refine task in multi-task training). This saves API costs and leverages the model's specific training for error correction.

### Performance Impact: **None (0% compromise)**
API-based fixing is identical to or better than what the plan specifies. The lightweight model requirement is easily met by Gemini Flash or Claude Haiku.

### Verdict: **Fully beneficial — implement as designed.**

---

## Operation 9: Adaptive SQL Selection (Online — Step 4)

### Ideal Resources
- SQL execution engine (SQLite).
- RL-trained 7B selection model (for tournament) — GPU with ~10GB VRAM.
- Generator performance rankings (pre-computed).

### Implementation on Our System
**Largely feasible** with minor adaptations.

**Phase 1 — Execution and Clustering:** Fully feasible. SQLite execution is CPU-based and instant.

**Phase 2 — Decision Branch:**

- **Fast Path (unanimous, ~45% of cases):** Fully feasible. No model needed — just return shortest SQL. Zero compromise.

- **Tournament Path (~55% of cases):**
  - **Option A — Local SFT 7B Selector (Recommended):**
    - Load the fine-tuned 7B selector (from Operation 4) via MLX.
    - Run pairwise comparisons on cluster representatives.
    - 7B model at 4-bit quantization: ~4GB VRAM, ~0.5–1 second per comparison.
    - Typical case: 6–10 comparisons = 3–10 seconds.
  - **Option B — API-based Selection:**
    - Use **Claude Haiku** or **Gemini-2.5-Flash** for pairwise comparison.
    - Cost: ~$0.001–0.005 per comparison, ~$0.01–0.05 per question.
    - Latency: ~1–2 seconds per comparison (parallelizable in pairs).

### Conditions & Adaptations
- Candidate reorganization is a pure algorithm (sorting by cluster size and generator ranking) — no adaptation needed.
- If using local 7B selector: ensure it is unloaded when generators need VRAM (sequential model loading).
- The fast path optimization is critical on our system — it saves both compute and API costs for ~45% of questions.

### Performance Impact: **Minor (estimated 0.5–1.5% EX reduction)**
- SFT selector vs. RL selector: ~0.5–1% gap (most of the selection value comes from the pairwise format and the tournament structure, not the RL training specifically).
- API-based selection may actually be comparable or better for difficult disambiguation cases.
- The adaptive strategy (fast path + representative tournament + reorganization) is fully preserved.

### Verdict: **Still highly beneficial — implement with local SFT selector or API fallback.**

---

## Overall Impact Assessment

### Cumulative Performance Estimate

| Component | Original Expected EX | Adapted Expected EX | Degradation |
|-----------|:---:|:---:|:---:|
| Candidate Pool Upper Bound | ~85–87% | ~82–84% | -3% |
| Selection Efficiency | ~90% of upper bound | ~88% of upper bound | -2% |
| **Final EX Accuracy** | **~76–78%** | **~72–74%** | **~3–5%** |

For reference, the current SOTA methods achieve:
- Agentar: 74.90% (requires 32× A100s)
- CHASE-SQL: 73.01% (requires massive API budget)
- XiYan: 73.34% (requires multi-GPU training)

**Our adapted system at ~72–74% would be competitive with SOTA methods** while running on a single M2 Mac with API access — a dramatically more accessible setup.

### Cost Profile (Adapted)

| Stage | API Calls | Estimated Cost/Question |
|-------|:---------:|:-----------------------:|
| Context Grounding | 1 (Haiku/Flash) | ~$0.001 |
| Schema Linking | 2–3 (Sonnet/Pro) | ~$0.01–0.02 |
| Generation (API) | 8–9 (mixed models) | ~$0.05–0.12 |
| Generation (Local 7B) | 2 (local) | $0 |
| Query Fixing | 0–5 (Haiku/Flash) | ~$0.001–0.005 |
| Selection (Local or API) | 0–10 | ~$0–0.01 |
| **Total** | **~13–30** | **~$0.06–0.16** |

At ~$0.10 average per question, running the full BIRD dev set (1534 questions) costs ~$150.

### Implementation Priority Order

Given constrained resources, implement operations in this order to maximize incremental value:

| Priority | Operation | Reason |
|:--------:|-----------|--------|
| 1 | Op 0: DB Profiling + Summarization | Foundation for everything; zero hardware constraint |
| 2 | Op 1: Index Building | Enables grounding and retrieval; zero hardware constraint |
| 3 | Op 7: Diverse SQL Generation (API-based) | Core value driver; API-only, immediate results |
| 4 | Op 5: Context Grounding | Cheap, improves generation quality |
| 5 | Op 6: Schema Linking | Moderate cost, significant quality improvement |
| 6 | Op 8: Query Fixer | Low cost, rescues broken candidates |
| 7 | Op 9: Selection (API-based first) | Start with API selection, then train local model |
| 8 | Op 3: Multi-task FT Training (7B) | Local training, improves generation diversity |
| 9 | Op 4: Selection Model Training (7B) | Local training, reduces ongoing API costs |
| 10 | Op 2: RL Generator Training (7B RFT) | Most complex local training; optional enhancement |

### Recommended Implementation Phases

**Phase 1 — API-Only Baseline (1–2 weeks):**
Implement Operations 0, 1, 5, 6, 7 (API generators only), 8, 9 (API selection). This gives a working end-to-end pipeline with zero local model training. Expected accuracy: ~70–72% EX.

**Phase 2 — Local Model Enhancement (2–4 weeks):**
Train local 7B multi-task generator (Op 3) and 7B selector (Op 4). Integrate into the pipeline as additional generator and local selector. Expected accuracy: ~72–74% EX.

**Phase 3 — Optimization (2–4 weeks):**
Experiment with RFT training for reasoning generator (Op 2). Tune prompts, temperatures, and candidate counts. Ablation studies. Expected accuracy: ~73–75% EX.

---

## Technical Stack Recommendations

| Component | Recommended Tool |
|-----------|-----------------|
| LLM API Clients | `litellm` (unified interface for Gemini + Claude) |
| Local Model Inference | `mlx-lm` (M2-optimized) |
| Local Model Training | `mlx-lm` fine-tuning or `unsloth` (if compatible) |
| Embeddings | `sentence-transformers` with `all-MiniLM-L6-v2` |
| Vector Search | `faiss-cpu` |
| LSH Indexing | `datasketch` |
| NER/Skeleton Masking | `spaCy` (en_core_web_sm) |
| Database Execution | `sqlite3` (Python stdlib) |
| Async API Calls | `asyncio` + `aiohttp` or `httpx` |
| Experiment Tracking | `wandb` or simple JSON logging |
| Configuration | `hydra` or `pydantic-settings` |
