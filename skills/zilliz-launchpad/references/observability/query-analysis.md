# Query analysis

Where slow-query data comes from, how the launchpad samples it into `observability.json`, and common diagnoses.

## Surfaces

- **Local Standalone** — `docker compose logs milvus | grep -i slow` (threshold: > 100 ms by default). No sampling into `observability.json` on the local path.
- **Zilliz Cloud** — Query Log in the console with per-query latency breakdown. Deploy surfaces the Grafana dashboard URL in `deploy.json.observability.grafana_dashboard`; follow that for live data.

## Query-log sampling

Phase 5 (`evaluate`) and Phase 6 (`deploy`) both append to `observability.json` in the run dir when it exists:

- **Evaluate** adds one entry per run to `latency_samples` with the measured p50/p95/p99 (see `references/observability/metrics.md` for the schema).
- **Deploy** adds one entry per run to `deploy_snapshots` with cluster id, collection, post-ingest row count, and ingest mode.

A single run dir accumulates entries over time, so reruns give you a timeline without extra tooling.

## Common patterns

- **Hot segment** — a single segment receives disproportionate load. Mitigate with more query nodes (Dedicated/BYOC) or by repartitioning.
- **Expression filter mis-ordering** — put selective filters before expensive ones.
- **`ef` / `nprobe` too low** — recall suffers, users complain. Bump it, accept the latency hit. Phase 5's comparison mode is the right tool for picking a new value: re-run `evaluate --compare variants.yaml` with one variant per candidate `ef`.
- **Cold segments after ingest** — first search after a large ingest warms the index. If p99 in `observability.json.latency_samples` looks elevated right after a deploy, rerun `evaluate` once the cluster has been warm for a few minutes.
