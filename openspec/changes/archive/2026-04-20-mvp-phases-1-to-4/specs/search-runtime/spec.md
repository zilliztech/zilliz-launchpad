## ADDED Requirements

### Requirement: Dense vector search

The library SHALL expose `search_dense(collection, query_text, top_k, filter=None)` which embeds the query via the same provider used at ingest time and returns ranked hits with id, score, and selected scalar fields.

#### Scenario: Basic dense query

- **WHEN** `search_dense("docs", "action movies", top_k=5)` is called on a populated collection
- **THEN** 5 hits are returned in descending score order

#### Scenario: Filter is applied before ranking

- **WHEN** `search_dense("docs", "space opera", top_k=5, filter='year >= 2010')` is called
- **THEN** every hit's `year` field is `>= 2010`

### Requirement: Sparse vector search

The library SHALL expose `search_sparse(collection, query_text, top_k)` which uses a BM25-compatible sparse representation stored alongside the dense vector. The feature MAY be unavailable if the collection was not ingested with sparse vectors enabled; in that case the call MUST raise a clear `SparseUnavailable` error.

#### Scenario: Sparse hit on lexical match

- **WHEN** the collection contains the phrase "quantum entanglement" in a document
- **AND** `search_sparse("docs", "quantum entanglement", top_k=3)` runs
- **THEN** that document appears in the top 3

#### Scenario: Sparse disabled at ingest

- **WHEN** the collection was ingested with `sparse.enabled=false`
- **AND** `search_sparse(...)` is called
- **THEN** `SparseUnavailable` is raised with a remediation message

### Requirement: Hybrid search

The library SHALL expose `search_hybrid(collection, query_text, top_k, fusion="rrf")` which runs dense and sparse searches and fuses results. `fusion` MUST accept `"rrf"` (default) and `"weighted"` (with optional `weights=(α, β)`).

#### Scenario: RRF fusion combines results

- **WHEN** dense returns `[A, B, C]` and sparse returns `[B, D, E]` for a query
- **AND** `search_hybrid(..., fusion="rrf", top_k=3)` runs
- **THEN** `B` is ranked higher than singleton hits from either list (because it appears in both)

### Requirement: Optional reranker

`search_dense`, `search_sparse`, and `search_hybrid` SHALL accept an optional `rerank` argument naming a supported reranker (`"cohere-rerank-3"`, `"bge-reranker-v2-m3"`). When provided, the top `top_k * 3` raw hits are reranked and truncated to `top_k`.

#### Scenario: Cohere reranker is applied

- **WHEN** `search_dense("docs", q, top_k=5, rerank="cohere-rerank-3")` runs
- **AND** `COHERE_API_KEY` is set
- **THEN** 5 hits are returned
- **AND** their order is Cohere's rerank order (not the raw vector score order)

#### Scenario: Reranker API key missing

- **WHEN** `rerank="cohere-rerank-3"` is requested and `COHERE_API_KEY` is unset
- **THEN** the call raises `MissingCredentialError` naming the variable
