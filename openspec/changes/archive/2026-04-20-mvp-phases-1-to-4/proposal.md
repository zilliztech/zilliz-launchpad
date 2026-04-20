## Why

Developers who want to build a search/RAG app on Milvus or Zilliz Cloud today face the same cliff documented by OpenSearch Launchpad: dozens of decisions (embedding model, index type + quantization, schema, chunking, hybrid vs dense, reranker) gate a working prototype, and there is no UI to validate results until all of them are made. `zilliz-launchpad` collapses that cliff into a guided, agent-driven workflow — from a sample document to a running local search UI in minutes — while staying open-source and Zilliz-Cloud-compatible.

This change delivers the **MVP: Phases 1–4** (Collect → Configure → Plan → Execute). Evaluate (Phase 5) and Deploy (Phase 6) are deferred to later changes.

## What Changes

- Introduce a new agent skill `skills/zilliz-launchpad/` installable via `npx skills add zilliztech/zilliz-launchpad`.
- Ship a Python 3.11+ library (`scripts/lib/`) of capability primitives: Milvus/Zilliz Cloud client, CRUD operations, ingestion pipeline, search (dense / sparse / hybrid / reranker).
- Ship a CLI entry `zilliz_ops.py` that the skill calls to execute each phase.
- Ship `start_milvus.sh` to launch Milvus Standalone via docker-compose (Milvus Lite is **not** supported in MVP).
- Ship a Next.js TypeScript demo UI under `scripts/ui/` for immediate result inspection.
- Ship a phase-scoped reference knowledge base under `references/knowledge/` (dense/sparse embedding matrix, hybrid guide, reranker guide, chunking, index tuning with quantization, schema design, RAG templates).
- Ship sample data (movies, BEIR-scifact mini) so the workflow works without user input.
- Embedding is **API-only** (OpenAI / Voyage / Cohere / Zilliz BYOM). No torch, no local models.
- Credentials are read from environment variables first, prompted via agent dialogue if missing.
- Plan and (future) Evaluate outputs land in `scripts/runs/<timestamp>/` as structured JSON + Markdown.
- **BREAKING**: N/A — this is the initial implementation.

## Capabilities

### New Capabilities

- `skill-orchestration`: the `SKILL.md` entry, six-phase flow contract, `zilliz_ops.py` CLI, `runs/<timestamp>/` output convention, credential resolution (env → agent prompt), reference lazy-loading rules.
- `milvus-client`: URI-aware connection layer for local Milvus Standalone and Zilliz Cloud; schema/collection/index lifecycle (create / load / drop).
- `ingestion-pipeline`: document chunking, API-based embedding invocation, batched insert with retry.
- `search-runtime`: dense, sparse, hybrid (RRF / weighted) queries with optional reranker and scalar filter.
- `requirements-gathering`: Phase 1 (Collect) + Phase 2 (Configure) conversational logic — data-shape detection from a sample, structured requirement capture (query pattern, deployment target, constraints).
- `plan-and-execute`: Phase 3 (Plan) decision tree producing a structured plan (schema + index + embedding + pipeline) and Phase 4 (Execute) orchestration that applies the plan to the target backend.
- `local-demo-ui`: Next.js TypeScript app that talks to the backend via a typed client and renders dense/hybrid results side-by-side.
- `sample-datasets`: bundled movies + BEIR-scifact-mini loaders so the full flow runs with zero user-supplied data.

### Modified Capabilities

_None — this is a greenfield project._

## Impact

- **New directories**: `skills/zilliz-launchpad/`, `tests/`, `mcp/` (stub for future MCP path).
- **New Python deps** (tracked in `pyproject.toml` via `uv`): `pymilvus`, `openai` (or generic HTTP client), `httpx`, `pydantic`, `typer` (CLI), `pytest`.
- **New JS deps** (under `scripts/ui/`): `next`, `react`, `typescript`, project standard lint/type configs.
- **Runtime requirement**: Docker (for Milvus Standalone). `start_milvus.sh` pre-checks and surfaces actionable errors.
- **External services** (optional, per-run): OpenAI / Voyage / Cohere APIs, Zilliz Cloud.
- **IDE surface**: first-class support for Claude Code only. Cursor / Codex targeted by later changes.
- **Out of scope for this change**: Milvus Lite support, MCP server implementation, Evaluate phase, Deploy phase, local embedding models, LLM-as-judge qrels synthesis, multi-modal data.
