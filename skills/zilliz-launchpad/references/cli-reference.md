# CLI / pymilvus quick reference

## `zilliz_ops.py` subcommands

```
collect    [--sample NAME | --input PATH] [--run-dir DIR]
configure  [--from-json FILE] [--run-dir DIR] [--use-case X] [--dataset-size N] [--deployment X]
plan       [--run-dir DIR]
execute    [--run-dir DIR] [--sample NAME | --input PATH] [--ui-port N] [--no-ui]
```

Each subcommand writes its artifact into the run directory and echoes a short summary to stdout. Errors go to stderr as single-line JSON envelopes (see `SKILL.md` § Error envelopes).

## `pymilvus` patterns used in `lib/`

```python
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType

client = MilvusClient(uri="http://localhost:19530")   # local
# client = MilvusClient(uri="https://...zillizcloud.com", token=os.environ["ZILLIZ_TOKEN"])

schema = CollectionSchema(fields=[
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
])
client.create_collection("docs", schema=schema)

idx = client.prepare_index_params()
idx.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE",
              params={"M": 16, "efConstruction": 200})
client.create_index("docs", idx)

client.upsert("docs", data=[{"id": "1", "text": "...", "embedding": [...]}])
client.load_collection("docs")

hits = client.search("docs", data=[vec], anns_field="embedding", limit=10)
```

## Useful filter expressions

```
year >= 2020
year >= 2020 and genre == "sci-fi"
id in ["m001","m002"]
```

## Optional: `zilliz` CLI integration

The optional [`zilliz` CLI](https://github.com/zilliztech/zilliz-cli) (≥ 0.3.0) unlocks Cloud auto-discovery, pre-flight, and bulk import. Detection is automatic via `lib.zilliz_cli.is_available()`.

### With the CLI on PATH (Cloud)

```bash
# one-time
zilliz auth login
zilliz cluster list --output json

# Phase 2 auto-picks a cluster from `zilliz cluster list`
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py configure \
    --use-case rag --deployment zilliz-serverless --dataset-size 500000

# Phase 4 runs `zilliz cluster describe` as a pre-flight and
# routes ingestion through `zilliz import create` when rows > 100k
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py execute --input big.jsonl
```

### Without the CLI (or for local Milvus)

```bash
# Phase 2 prompts for URI + token as before
export ZILLIZ_TOKEN=<api-key>
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py configure \
    --use-case rag --deployment zilliz-serverless --dataset-size 500000

# Phase 4 skips pre-flight and always uses the client-side upsert path
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py execute --input big.jsonl
```

Errors the CLI integration can emit (see SKILL.md for the full table):
`zilliz_cli_missing`, `zilliz_cli_auth`, `cluster_not_ready`.
