# Milvus metrics

Phase 6 records observability pointers in `runs/<ts>/deploy.json` and appends a deploy snapshot to `runs/<ts>/observability.json`. This note describes the contract.

## Scrape URL contract

`deploy.json.observability` carries the two URL pointers:

| field | set on | value |
| --- | --- | --- |
| `prometheus_url` | local Standalone | `http://localhost:9091/metrics` |
| `grafana_dashboard` | Zilliz Cloud | URL from `zilliz cluster describe` (field name varies by CLI version: `grafana_dashboard`, `grafanaDashboard`, `dashboard_url`) |
| `query_log_sample_path` | always | `observability.json` (relative to the run dir) |

Exactly one of `prometheus_url` / `grafana_dashboard` is populated per deploy.

## Key metrics to watch

Milvus Standalone exposes Prometheus metrics on `:9091/metrics` by default:

- `milvus_proxy_search_latency_bucket` — per-request latency histogram
- `milvus_proxy_search_req_count` — QPS
- `milvus_rootcoord_collection_num` — total collections
- `milvus_datanode_flushed_size` — flush backpressure signal
- `milvus_querynode_num_entities` — in-memory entity count per segment

When a slowdown appears in the latency histogram, correlate with:

- `milvus_querynode_cgo_time` — CGO overhead spike
- `milvus_datanode_flushed_size` — flush / compaction pressure

## Integrations

- Scrape with any Prometheus-compatible agent (local or Cloud with egress configured)
- Zilliz Cloud surfaces a pre-baked Grafana dashboard — the URL lands in `deploy.json`

## Image collections — different baselines

Image-search runs (use_case `image-search`, embedding modality `image`) have noticeably different observability shapes than the text default — calibrate dashboards accordingly:

- **Ingest throughput is encoder-bound.** Local CLIP ViT-B/32 is roughly 100 imgs/sec on M-series CPU and 1000+ on MPS/CUDA, so `milvus_datanode_flushed_size` will look idle and `milvus_proxy_insert_req_count` will appear bursty in batches of `IMAGE_BATCH_SIZE` (default 16).
- **Vector dim matters.** CLIP is 512 dim, Voyage multimodal is 1024 dim — `milvus_querynode_num_entities` should match `processed_files[]` from `execute.json`, but per-segment memory will be smaller than text collections (which default to 1536).
- **Search latency baseline is lower** because there's no BM25 fan-out (image collections always have `sparse_enabled=false`); a p95 above what the text path reports usually means the encoder, not Milvus.
- **No RAG-quality metrics** — eval reports for image collections always have `rag_metrics: null`. Watch `recall@10` on a labelled image qrels file or the derived eval (`--judge-llm openai:gpt-4o-mini`) instead.

## What Phase 6 writes to observability.json

```jsonc
{
  "pointers": {
    "prometheus_url": null,
    "grafana_dashboard": "https://grafana.zilliz.com/d/abc",
    "query_log_sample_path": "observability.json"
  },
  "deploy_snapshots": [
    { "cluster_id": "...", "collection": "...", "post_ingest_row_count": 12345,
      "ingest_mode": "bulk", "timestamp": "..." }
  ],
  "latency_samples": [
    { "source": "evaluate", "run_id": "...", "p50_ms": 12.3, "p95_ms": 42.0, ... }
  ]
}
```

Evaluate (Phase 5) appends to `latency_samples`; Deploy (Phase 6) appends to `deploy_snapshots`.
