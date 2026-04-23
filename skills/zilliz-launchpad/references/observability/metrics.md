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
