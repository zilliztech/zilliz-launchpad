# Troubleshooting

Common issues and fixes.

## `ERROR: Docker is not installed or not on PATH`

Install Docker Desktop: <https://docs.docker.com/get-docker/>. After install, launch the app and wait for the daemon to start (tray icon goes green). Retry `start_milvus.sh up`.

## `ERROR: Port 19530 is already in use`

A previous Milvus is still running.

```bash
docker ps | grep milvus
./skills/zilliz-launchpad/scripts/start_milvus.sh down
# or: docker stop milvus-standalone milvus-etcd milvus-minio
```

If no container matches, another process has the port:

```bash
lsof -iTCP:19530 -sTCP:LISTEN
```

Reset data entirely (nukes local collections):

```bash
./skills/zilliz-launchpad/scripts/start_milvus.sh clean
```

## `{"code": "missing_credential", "env_var": "OPENAI_API_KEY", ...}`

The requested phase needs an API key that isn't set. Follow the `export_hint` in the error payload, then re-run.

```bash
export OPENAI_API_KEY=sk-...
```

## `{"code": "schema_conflict", ...}`

You're trying to create a collection with a different schema than one that already exists. Either:

- Pick a new `collection_name` in `plan.json` and re-run `execute`, or
- Drop the existing collection: `python -c "from lib.client import MilvusClient; MilvusClient('http://localhost:19530').drop_collection('launchpad_collection')"`

## `{"code": "sparse_unavailable", ...}`

You asked for sparse/hybrid search but the collection was built with `sparse_enabled=false`. Change `configure.hybrid_preference` to `"hybrid"` or `"sparse"`, re-plan, and re-execute (drop collection first).

## No sample selected / Phase 1 fails

Run with `--sample movies` or `--sample beir-scifact-mini`, or pass `--input path/to/your.jsonl`.

## UI shows no results

1. Check the sidecar is up: `curl http://127.0.0.1:8000/health`
2. Check `runs/<timestamp>/execute.json` — the smoke query result should be non-empty
3. In the UI, try the `dense` mode first — it works even when sparse is disabled

## API embedding rate limits

Batch size is 64 by default. Shrink if you're hitting rate limits:

```bash
# Edit plan.json — future versions will expose this via configure
```

(A `--batch-size` flag is a small follow-up change.)

## Phase 4 smoke query returns zero hits

Almost always means ingestion finished before indexing completed. Wait a few seconds and query again; or re-run `execute` (idempotent).

## CI failing on UI secret leak check

Check no library under `lib/milvus-client.ts` or `app/**` reads `process.env.OPENAI_API_KEY` directly. Only `NEXT_PUBLIC_SIDECAR_URL` is safe to inline.
