# RAG / search templates

Three starting shapes the planner supports via `configure.use_case`:

## `rag`  (default)

- Long-form documents, natural-language queries, retrieval feeds an LLM
- Planner picks: dense + sparse (auto), chunking 512/64, HNSW
- Post-MVP: reranker for quality bumps; evaluation via `faithfulness` / `answer relevance`

## `semantic-search`

- Short-to-medium docs; user sees retrieval results directly (no LLM)
- Planner picks: dense + sparse (auto), chunking 512/64, HNSW
- Recall@10 + manual quality review are the right metrics

## `recommendations`

- Item catalog, user query is context (history / profile embedding) rather than text
- Planner picks: dense only (sparse off by default), chunking disabled (one record per item)
- Scalar filters (genre, category, stock status) become critical — use Milvus filter expressions

## Choosing between them

Use the most restrictive template that fits. When unclear, default to `rag` — it's the most flexible.
