# NL2SQL Project — Claude Code Memory

## Project Goal
Build an Adaptive Hybrid NL2SQL system based on the method plan in `New_NL2SQL_method_plan.md`.
Phase 1 (current) implements an API-only baseline using the Claude API — no local model training.
Target: ≥68% Execution Accuracy on the BIRD dev set (1,534 questions).

## Key Documents
- `New_NL2SQL_method_plan.md` — Full method design (9 operations, rationale, ablation data)
- `Implementation_Guide_Constrained_Hardware.md` — Hardware adaptation notes (M2 Mac, API-only)
- `Phase1_implementation_details.md` — **Primary implementation reference** (the spec to follow)

## Architecture: Two-Phase Pipeline
**Offline** (one-time per database, results cached to disk):
- Op 0: Statistical profiling + LLM summarization → DDL + Markdown schemas
- Op 1: LSH index (cell values) + FAISS index (field semantics) + Example vector store

**Online** (per question at inference time):
- Op 5: Context grounding (keyword extraction → LSH lookup + few-shot retrieval)
- Op 6: Adaptive schema linking (FAISS pre-filter → 2 LLM iterations → S₁, S₂ schemas)
- Op 7: Diverse SQL generation (3 generator types, 10–11 candidates total)
- Op 8: Query fixer (execute → fix errors/empty results, β=2 iterations)
- Op 9: Adaptive selection (fast path if unanimous; otherwise pairwise tournament)

## Claude Model Assignment
| Task | Model |
|------|-------|
| Keyword extraction, query fixing, pairwise selection, field summarization | `claude-haiku-4-5-20251001` |
| Schema linking, Generator B2 (complex SQL), Generator C (ICL), Generator B1 standard | `claude-sonnet-4-6` |
| Generator A (reasoning) | `claude-sonnet-4-6` + extended thinking |

## Code Conventions
- Python 3.11+, async/await throughout (asyncio)
- Pydantic v2 for all data models and settings
- `pydantic-settings` for config (reads from `.env`)
- All LLM calls use `anthropic` SDK with `tenacity` retry (exponential backoff)
- Tool-use (JSON schema) for ALL structured LLM outputs — never free-text parsing
- Prompt caching (`cache_control: {"type": "ephemeral"}`) on schema/example blocks
- Disk cache for LLM responses (keyed by SHA256 of model+prompt), toggled via `CACHE_LLM_RESPONSES`
- Tests: pytest + pytest-asyncio + pytest-mock; mark live API tests with `@pytest.mark.live`

## Project Structure (src/)
```
config/settings.py        data/bird_loader.py       data/database.py
preprocessing/profiler.py preprocessing/summarizer.py preprocessing/schema_formatter.py
indexing/lsh_index.py     indexing/faiss_index.py   indexing/example_store.py
grounding/context_grounder.py
schema_linking/schema_linker.py
generation/base_generator.py  generation/reasoning_generator.py
generation/standard_generator.py  generation/icl_generator.py
fixing/query_fixer.py
selection/adaptive_selector.py
pipeline/offline_pipeline.py  pipeline/online_pipeline.py
cache/cache_manager.py
evaluation/evaluator.py  evaluation/metrics.py
```

## Generator Configuration (Phase 1)
```
Generator A  (Reasoning, Sonnet + extended thinking): 4 candidates (S₁×2 prompts + S₂×2 prompts)
Generator B1 (Standard, Haiku):                       2 candidates (S₁, S₂)
Generator B2 (Complex SQL, Sonnet):                   2 candidates (S₁, S₂)
Generator C  (ICL, Sonnet + few-shot):                2–3 candidates (direct, CoT, step-back)
Total: 10–11 candidates
```

## Key Design Decisions Made
1. Extended thinking prompt variation (not temperature) used for Generator A diversity (API forces temp=1 with thinking)
2. Prompt caching on the full field-summary block in schema linking saves ~60% of schema-linking costs
3. Confidence scoring: +1.0 success + 0.5 plausibility bonus - 0.5 per fix iteration needed
4. Selection fast path: if all candidates cluster to same result, return shortest SQL, 0 API calls
5. Representatives in tournament = 1 per cluster (shortest SQL by confidence), not all candidates
6. Example store excludes same-db_id examples (prevent schema leakage into few-shot)

## Dataset
- BIRD dev: 1,534 questions, 11 SQLite databases (in `data/bird/dev/`)
- BIRD train: ~9,428 questions, 84 databases (in `data/bird/train/`) — used for example store only
- BIRD mini-dev: 500 questions (in `data/bird/mini_dev/`) — used for smoke tests
- Metric: Execution Accuracy (EX) — compare result sets after sorting and type normalization

## What to Update Here
After each implementation prompt, update this file with:
- Any new architectural decisions made during implementation
- Any deviations from Phase1_implementation_details.md and why
- Performance observations from live tests
- Patterns that should be followed in subsequent components
