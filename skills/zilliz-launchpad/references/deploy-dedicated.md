# Deploying to Zilliz Cloud Dedicated

**Status**: reference only; Deploy phase is future work. You can still connect from Phase 4.

## When to pick Dedicated

- Steady, high-QPS production workloads
- Strict latency SLOs (< 50ms p95)
- Need for GPU-backed indices (`GPU_IVF_FLAT`, `GPU_IVF_PQ`, `GPU_CAGRA`)

## Sizing (rule of thumb)

| Dataset size | Suggested plan |
| --- | --- |
| < 10M vectors | S-tier CPU |
| 10M–100M     | M-tier CPU or S-tier GPU |
| > 100M       | L-tier GPU |

Consult <https://docs.zilliz.com> for up-to-date tier specs.

## Constraints / notes

- `DISKANN` and `HNSW` CPU paths both available
- GPU indexes require matching GPU-tier CUs
- TLS is on by default; always use the `https://` endpoint with a token

TODO (Phase 6): automate tier selection based on dataset size + latency target.
