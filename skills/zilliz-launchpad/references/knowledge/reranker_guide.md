# Rerankers

A reranker takes the top `k*3` raw candidates from vector search and reorders them using a cross-encoder model. Costs more per query, typically moves recall@10 up by a few points.

## Supported in MVP

| Adapter id                 | Backend             | Cost surface |
| ---                        | ---                 | --- |
| `cohere-rerank-3`          | Cohere Rerank 3     | Paid per 1k docs; 1k req/min free tier. |
| `bge-reranker-v2-m3`       | Zilliz BYOM endpoint | Self-hosted; you control cost. |

## When to enable

- User-facing search / RAG where a few extra points of quality matter
- Small top_k (≤ 10) — reranking 30 candidates is cheap; reranking 300 is not
- Mixed-intent queries where dense alone confuses semantic vs lexical matches

## When to skip

- Analytics / bulk batch search (cost blows up)
- Very large top_k
- Latency budget < 100ms per query (rerank adds 50-200ms depending on size)

## Opt-in by default

The planner's default is `reranker=null`. Set `configure.reranker_preference` to `"cohere"` or `"bge"` to turn it on. Auto-mode will not enable rerank without explicit consent.
