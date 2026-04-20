# Evaluation guide (Phase 5 — future)

Not implemented in the MVP. This note previews the Phase 5 contract so Phase 3's plans stay forward-compatible.

## Three metric families

### Retrieval quality (needs qrels)

- `recall@k` — fraction of relevant docs that appear in top-k (primary)
- `MRR@10` — mean reciprocal rank of the first relevant hit
- `NDCG@10` — graded relevance, discounted by rank

If the user has no qrels, Phase 5 skips these and reports latency only (no LLM-as-judge).

### Latency

- `p50 / p95 / p99` under a configurable concurrency level
- Always measured against Milvus Standalone or Zilliz Cloud — **never against Milvus Lite** (not in scope here anyway)

### RAG quality (ragas)

- `faithfulness` — does the answer stay grounded in retrieved chunks
- `answer_relevance` — does the answer match the question
- Requires an LLM judge (same API key as your generator)

## Comparison mode

Phase 5 will be able to re-run a fixed query set against multiple plan variants and produce a decision table:

| variant | recall@10 | p95 (ms) | faithfulness | cost / query |
| ---     | ---       | ---      | ---          | --- |

Variants are produced by swapping one axis: embedding model, index params, hybrid vs dense, reranker on/off.

## Output

`runs/<timestamp>/eval_report.json` + `eval_report.md` with the decision table and recommendations.
