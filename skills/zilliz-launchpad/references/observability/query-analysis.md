# Query analysis (Phase 6 territory)

Not wired into the MVP. Notes for future work.

## Surfaces

- Milvus logs: `docker compose logs milvus | grep -i slow` (> 100ms by default)
- Zilliz Cloud: Query Log in the console, with per-query latency breakdown

## Common patterns

- **Hot segment** — single segment receiving disproportionate load. Mitigate with more query nodes (Dedicated/BYOC) or by repartitioning.
- **Expression filter mis-ordering** — put selective filters before expensive ones.
- **`ef`/`nprobe` too low** — recall suffers, users complain. Bump it, accept latency hit.

## TODO (Phase 6)

- Integrate query log sampling into `runs/<timestamp>/observability.json`
- Surface top-N slow queries + their filters in a follow-up UI tab
