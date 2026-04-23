# Evaluation guide (Phase 5)

`uv run python scripts/zilliz_ops.py evaluate` runs a fixed query set against a completed Execute run and writes `eval_report.json` + `eval_report.md` in the run dir.

## Three metric families

### Retrieval quality (needs qrels)

Enable by passing `--qrels <path>`; each JSONL line must be `{"query": str, "relevant_ids": [str, ...], "grade": int?}`.

- `recall@k` — fraction of relevant docs that appear in top-k (k defaults to 10)
- `MRR@10` — mean reciprocal rank of the first relevant hit
- `NDCG@10` — graded relevance, discounted by rank

If you have no qrels, Phase 5 degrades gracefully:

- `--queries <path>` (one query per line) → latency only (no retrieval math, since there are no labels)
- no flags → **derived mode**: the evaluator samples 25 docs from the corpus, takes the first sentence of each as the query, and treats the source doc as the (single) relevant id. The report is tagged `derived: true` so you know it's a smoke test not a real eval.

### Latency

- `p50 / p95 / p99` in milliseconds, wall-clock, against the live collection
- `--concurrency <N>` (default 1; max 64) controls worker-pool size
- Measured against Milvus Standalone or Zilliz Cloud — **refused against Milvus Lite** (`backend_unsupported`), since latency numbers against Lite are meaningless

### RAG quality (ragas, opt-in)

Enable by passing `--judge-llm <provider>:<model>` (e.g. `openai:gpt-4o-mini`). Requires the corresponding API key in the env (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `COHERE_API_KEY`); missing key emits `judge_unavailable`.

- `faithfulness` — does the answer stay grounded in retrieved chunks
- `answer_relevance` — does the answer match the question

Phase 5 uses the top retrieved chunk as the stand-in answer; plug in a real generator in a follow-up run if you want to score actual LLM output. Pinned to `ragas>=0.2,<0.3`.

## Comparison mode

`--compare variants.yaml` re-runs the same query set against alternative plan variants and produces a decision table. Requires `--qrels` — comparison without labels is refused with `qrels_missing`.

```yaml
variants:
  - name: small-m
    overrides:
      index:
        params: { M: 8 }
  - name: voyage
    overrides:
      embedding:
        model: voyage-3
  - name: no-hybrid
    overrides:
      hybrid: false
  - name: no-rerank
    overrides:
      reranker: null
```

Allowed override axes: `embedding.{model,provider,dim}`, `index.{type,metric,params}`, `hybrid` (bool), `reranker` (string or null). Cap: 6 variants by default (override with `--allow-large`).

The decision table lands in `eval_report.md` with fixed columns:

| variant | recall@10 | p95 (ms) | faithfulness | cost/query |
| --- | --- | --- | --- | --- |

`cost/query` is currently a placeholder (`—`) until the evaluator learns to price per-provider calls.

## Output

`runs/<ts>/eval_report.json` — machine-readable, stable key order. Top-level keys:
- `derived` (bool), `query_count`, `retrieval_metrics`, `latency_metrics`,
- `rag_metrics` (null if no judge), `variants` (empty array for single-plan runs),
- `run_id`, `timestamp`.

`runs/<ts>/eval_report.md` — human-readable, contains the decision table and a "Note" block when `derived: true`.
