# Index tuning

Milvus index types supported in the MVP:

| Index      | Best for                                      | Backend    | Build time | Query latency |
| ---        | ---                                           | ---        | ---        | --- |
| `FLAT`     | Sub-10k vectors, exact recall                 | any        | instant    | linear in N |
| `IVF_FLAT` | Small-medium (50k–1M), tunable recall         | any        | fast       | good |
| `HNSW`     | Medium (100k–10M), low-latency, high recall   | any        | medium     | excellent |
| `DISKANN`  | Large (>10M), memory-constrained              | Zilliz Cloud | slow     | good |

## Decision tree used by the planner

```
dataset_size <= 100_000         → HNSW M=16  efConstruction=200   (Standalone-ok)
dataset_size <= 1_000_000       → HNSW M=24  efConstruction=256   (Standalone-ok)
dataset_size  > 1_000_000 AND deployment in {serverless,dedicated,byoc}
                                 → DISKANN search_list_size=100   (Cloud-only)
dataset_size  > 1_000_000 AND deployment=local-standalone
                                 → HNSW M=32  efConstruction=256   + warning
```

## Metrics

- **`COSINE`** — default for text embeddings (normalized)
- **`IP`** — normalized dot product; equivalent to cosine when vectors are L2-normalized
- **`L2`** — Euclidean; use when your embedding provider documents L2 specifically

Text embedding providers supported in the MVP all return L2-normalized outputs — `COSINE` is safe as the default.

## Quantization (SQ8 / PQ)

**Not wired into the planner in this MVP.** HNSW + SQ8 or PQ cuts memory 4-8× with modest recall loss; when you hit scale where memory matters, add `"quantization": "SQ8"` to `plan.json` and rerun `execute`. This is a follow-up change.

## `ef` / `nprobe` at query time

- **HNSW** `ef` — query-time tunable; higher = better recall / slower. Default 64 is a good starting point.
- **IVF_*** `nprobe` — higher = better recall / slower. Default 10.

The MVP doesn't expose these via the sidecar; add them in a follow-up change if users need recall/latency trade-offs live.
