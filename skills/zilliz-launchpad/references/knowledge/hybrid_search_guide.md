# Hybrid search (dense + sparse)

The launchpad's `search_hybrid` runs dense and sparse searches in parallel and fuses the ranked lists.

## Fusion strategies

- **RRF (Reciprocal Rank Fusion)** — default. Parameter-free, robust across corpora. Score: `Σ 1/(k + rank_i)` with `k=60`.
- **Weighted** — `α·dense_norm + β·sparse_norm` with `α + β = 1` (typical `0.5/0.5`, tune empirically).

Pick RRF unless you have a reason. Weighted requires per-corpus tuning.

## When to use hybrid

Hybrid reliably beats dense-only on:

- Queries mixing natural language with rare tokens (product codes, species, acronyms)
- Short keyword queries (dense models under-index on lexical overlap)
- Code search
- Multilingual corpora where exact-match matters

## Configuration

The launchpad enables sparse and hybrid when:

- `configure.hybrid_preference ∈ {"sparse", "hybrid"}`, or
- `configure.hybrid_preference == "auto"` **and** `use_case ∈ {"rag", "semantic-search"}`

To compare dense vs hybrid on the same collection: run Phase 4 once, then call `search_dense` and `search_hybrid` against the sidecar from the UI.
