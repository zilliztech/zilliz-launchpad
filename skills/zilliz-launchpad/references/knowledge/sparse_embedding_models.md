# Sparse vectors

Milvus 2.4+ supports a native `SPARSE_FLOAT_VECTOR` field for BM25-compatible, SPLADE, and BGE-M3-sparse representations.

## Common options

- **BM25-style (Milvus built-in)** — the default when you enable `sparse` in the plan. No external embedding call needed; Milvus tokenizes + weights at ingest.
- **SPLADE / BGE-M3 sparse** — requires generating sparse vectors server-side or via a BYOM endpoint. Higher quality than BM25 on many benchmarks.

## When to enable

Enable sparse when the use case is:

- RAG on mixed content (acronyms, product codes, rare named entities)
- Multilingual or code search with strong lexical overlap
- Anywhere hybrid (dense + sparse) would help

Leave it off when:

- Pure semantic search on short, natural-language queries
- Cost / latency is tight and dense alone performs adequately

## Impact on plan

`sparse_enabled: true` adds a `sparse` field to the collection schema, increases storage ~15-25%, and enables the `search_sparse` and `search_hybrid` code paths.
