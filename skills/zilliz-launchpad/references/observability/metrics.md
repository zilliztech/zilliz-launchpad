# Milvus metrics (Phase 6 territory)

Not wired into the MVP. Milvus Standalone exposes Prometheus metrics on `:9091/metrics` by default.

## Key metrics to watch

- `milvus_proxy_search_latency_bucket` — per-request latency histogram
- `milvus_proxy_search_req_count` — QPS
- `milvus_rootcoord_collection_num` — total collections
- `milvus_datanode_flushed_size` — flush backpressure signal
- `milvus_querynode_num_entities` — in-memory entity count per segment

## Integrations

- Scrape with any Prometheus-compatible agent
- Zilliz Cloud surfaces a pre-baked Grafana dashboard in the console; no setup needed

## Opening issues

When a slowdown appears in the latency histogram, correlate with:

- `milvus_querynode_cgo_time` — CGO overhead spike
- `milvus_datanode_flushed_size` — flush/compaction pressure

A proper observability story belongs to Phase 6. This note is a stub.
