# Schema design

## Baseline schema produced by the planner

```
id         VARCHAR(128)  PK        # deterministic SHA-256 hash (or user-provided)
text       VARCHAR(65535)          # chunk content
embedding  FLOAT_VECTOR(dim)       # dense
[sparse    SPARSE_FLOAT_VECTOR]    # when sparse_enabled
<extra>                            # scalar fields from collect.json
```

## Primary key

- Default: deterministic hash of `source_id + chunk_idx`. Enables idempotent re-ingest.
- If the user insists on a natural key (e.g., `doc_id`), set `configure.suggested_primary_key` and ensure uniqueness across chunks yourself (prefix with `doc_id::chunk_idx`).

## Scalar fields

- String → `VARCHAR` with `max_length=256` default. Bump to 65535 for free-form text.
- Int / float / bool → native types; used for filters (`year >= 2020`, `score > 0.5`).
- Avoid very wide string fields (> 2KB) as non-PK — store as part of `text` instead.

## Partition key (future work)

Milvus supports a partition key field for tenant-scoped filtering. Not wired into the planner yet; add to `plan.json` manually if you need it (e.g., `tenant_id`).

## Anti-patterns

- Storing full documents in a scalar field *and* chunking them: duplicates data. Just chunk.
- One collection per document: collections are heavyweight. One collection for the whole corpus; use filters or partitions for scoping.
