## 1. Project scaffolding

- [x] 1.1 Initialize `pyproject.toml` with `uv` and declare Python 3.11+ and the MVP runtime deps (`pymilvus`, `typer`, `httpx`, `pydantic`, `fastapi`, `uvicorn`, `openai`, `cohere`, `voyageai`)
- [x] 1.2 Initialize dev deps (`pytest`, `pytest-asyncio`, `ruff`, `mypy`) and add lint/format/type-check commands
- [x] 1.3 Create directory skeleton: `skills/zilliz-launchpad/{references,scripts,scripts/lib,scripts/sample_data,scripts/ui,scripts/runs}`, `tests/{fixtures}`, `mcp/` (stub with a README)
- [x] 1.4 Add `.gitignore` entries for `__pycache__`, `.venv`, `node_modules`, `scripts/runs/*`, `scripts/ui/.next`
- [x] 1.5 Wire a simple CI workflow (lint + type-check + unit tests)

## 2. `lib/` capability primitives (milvus-client)

- [x] 2.1 Implement `lib/client.py`: URI-aware factory returning a configured `pymilvus.MilvusClient` with host-pattern-based token resolution
- [x] 2.2 Implement `lib/operations.py`: `create_collection` (idempotent + schema-conflict detection), `load_collection`, `drop_collection`, `collection_exists`
- [x] 2.3 Implement index ops in `lib/operations.py`: `create_index` (idempotent or drop-rebuild), `describe_index`, `drop_index`; support FLAT / IVF_FLAT / HNSW / DISKANN
- [x] 2.4 Write unit tests for client URI detection and schema-conflict detection (mock pymilvus)

## 3. `lib/` capability primitives (ingestion-pipeline)

- [x] 3.1 Implement `lib/chunking.py`: recursive character splitter with configurable size / overlap
- [x] 3.2 Implement `lib/embeddings.py`: provider strategies for `openai`, `voyage`, `cohere`, `zilliz-byom`; selected by string key
- [x] 3.3 Implement `lib/ingest.py`: chunk → embed → batched insert with exponential-backoff retry; deterministic SHA-256 primary keys
- [x] 3.4 Add tests for chunking boundaries, retry behavior (mocked transport), and idempotent PKs

## 4. `lib/` capability primitives (search-runtime)

- [x] 4.1 Implement `lib/search.py`: `search_dense`, `search_sparse`, `search_hybrid` (RRF + weighted fusion), optional reranker hook
- [x] 4.2 Add reranker adapters for Cohere and BGE (BGE via Zilliz BYOM endpoint)
- [x] 4.3 Raise `SparseUnavailable` when collection lacks sparse field
- [x] 4.4 Unit tests for fusion logic and filter translation

## 5. Credential & error primitives

- [x] 5.1 Implement `lib/credentials.py` with `resolve(key, prompt_if_missing=True)` returning the value or raising `MissingCredentialError`
- [x] 5.2 Implement structured error classes: `MissingCredentialError`, `SchemaConflictError`, `SparseUnavailable`, with a common serialization for the CLI

## 6. Sample datasets

- [x] 6.1 Add `sample_data/movies.jsonl` (≤5k rows) and `sample_data/movies.PROVENANCE.md` with source + license (**synthetic — 20 rows**)
- [x] 6.2 Add `sample_data/beir_scifact_mini.jsonl` + `beir_scifact_qrels.tsv` + `sample_data/beir_scifact.PROVENANCE.md` (**synthetic — 15 docs + 8 queries**)
- [x] 6.3 Implement `lib/samples.py`: `list_datasets()` and `load(name)`

## 7. CLI (`zilliz_ops.py`) & Phase 1–4 logic

- [x] 7.1 Scaffold Typer app in `scripts/zilliz_ops.py` with subcommands `collect`, `configure`, `plan`, `execute`
- [x] 7.2 Implement `collect`: sample analysis producing `collect.json` (field types, lengths, suggested PK)
- [x] 7.3 Implement `configure`: structured dialogue (stdin if non-agent, stdout structured requests if agent) producing `configure.json`
- [x] 7.4 Publish `references/requirement-profile.schema.json` and validate merged Phase 1+2 output against it
- [x] 7.5 Implement `plan`: deterministic decision tree producing `plan.json` + `plan.md` under `scripts/runs/<utc-iso>/`
- [x] 7.6 Implement `execute`: apply plan (collection, index, ingest, start UI sidecar) idempotently
- [x] 7.7 Implement post-execute smoke query and exit-code semantics

## 8. Local UI sidecar

- [x] 8.1 Implement `lib/ui.py`: FastAPI app exposing `POST /search` → `search_dense|sparse|hybrid`
- [x] 8.2 `execute` starts the sidecar on a configurable port and writes the PID to the run directory

## 9. Next.js demo UI

- [x] 9.1 Scaffold `scripts/ui/` with `create-next-app --ts` (App Router) — **scaffolded manually; run `npm install` on first use**
- [x] 9.2 Implement root `/` page: query input, mode selector (Dense / Sparse / Hybrid), top-k input, results cards
- [x] 9.3 Implement `lib/milvus-client.ts`: typed fetch wrapper calling the sidecar
- [x] 9.4 Enforce no secrets in the browser bundle (grep check added to CI)

## 10. Skill content

- [x] 10.1 Author `skills/zilliz-launchpad/SKILL.md` — ≤500 lines, Phase 1–4 flow, reference lazy-load table
- [x] 10.2 Add per-phase reference notes (`references/deploy-*.md`, `cli-reference.md`) — stubs with TODO markers are acceptable for non-MVP phases
- [x] 10.3 Author `references/knowledge/{dense,sparse}_embedding_models.md`, `hybrid_search_guide.md`, `reranker_guide.md`, `document_processing.md`, `index_tuning.md`, `schema_design.md`, `rag_templates.md`
- [x] 10.4 Add `references/observability/{metrics,query-analysis}.md` as forward-looking notes (Phase 6 territory)

## 11. docker-compose & bootstrapping

- [x] 11.1 Write `scripts/start_milvus.sh`: pre-check Docker, detect port conflicts, start/stop Milvus Standalone via docker-compose
- [x] 11.2 Include the docker-compose file and a brief troubleshooting block

## 12. Tests

- [x] 12.1 `tests/test_plan.py`: golden-file tests over the plan decision tree (at least 6 profiles covering dataset-size buckets × deployment targets)
- [x] 12.2 `tests/test_execute.py`: E2E against Milvus Standalone using `movies` fixture; asserts collection row count, index presence, smoke query hit
- [x] 12.3 Mark Zilliz-Cloud integration tests with `@pytest.mark.cloud` and keep them out of default CI
- [x] 12.4 Add a CI lane that grep-checks the built UI bundle for secret names

## 13. Packaging & distribution

- [x] 13.1 Produce a `skill.json` (or equivalent) describing the skill for `npx skills add`
- [x] 13.2 Write top-level `README.md` with Quick Start (4 commands: install skill, start Milvus, run phases, open UI)
- [ ] 13.3 **TODO (external)** — Verify `npx skills add zilliztech/zilliz-launchpad -a claude-code` works against a dry-run registry. Requires coordinating with `zilliztech` org admins to publish the skill. Cannot be done from this repo alone.

## 14. Documentation

- [x] 14.1 Troubleshooting doc (Docker missing, port 19530 in use, API key invalid, no sample selected)
- [x] 14.2 Contributing doc (how to add an embedding provider, how to add a sample dataset)
- [ ] 14.3 **TODO (post-merge)** — Archive this OpenSpec change upon merge. Run `openspec archive mvp-phases-1-to-4` after the PR lands.
