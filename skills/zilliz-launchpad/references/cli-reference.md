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
