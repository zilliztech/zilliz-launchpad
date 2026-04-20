## Context

`zilliz-launchpad` is a greenfield agent skill. No prior implementation exists in this repo. The parallel project [`opensearch-project/opensearch-launchpad`](https://github.com/opensearch-project/opensearch-launchpad) is the closest prior art; its design (skill-first, phase-scoped references <500 lines, MCP as optional backward-compat) is adopted where it fits. Divergence from OpenSearch Launchpad is driven by Milvus/Zilliz-specific decisions — vector-native workflow, index + quantization decision tree, and Zilliz Cloud as the deployment target.

The MVP (Phases 1–4) must produce a working local search app from a sample document, with no paid services required except an embedding API key. Phase 5 (Evaluate) and Phase 6 (Deploy) are out of scope for this change but their hooks are respected (e.g., `runs/<timestamp>/` output directory is introduced now so later changes can write reports there).

## Goals / Non-Goals

**Goals:**

- A user with a sample JSON/text file can go from `npx skills add zilliztech/zilliz-launchpad` to a searchable local UI in under 10 minutes on a laptop with Docker.
- Skill content (`SKILL.md`) stays under 500 lines; per-phase reference material is loaded lazily to protect the agent's context budget.
- Phase boundaries are enforced by the skill's flow, but the underlying Python `lib/` is organized by capability primitive (client / ingest / search), so future phases (5, 6) compose the same primitives.
- Milvus Standalone (local) and Zilliz Cloud are both targets of the same code path — only the URI differs.
- Credentials come from environment variables first; the skill prompts the user in dialogue only when a required variable is missing.
- The demo UI is a standard Next.js TS app so contributors can extend it without learning a bespoke framework.

**Non-Goals:**

- No Milvus Lite support in this change. (Deferred; reasons captured in `.plans/opensearch-launchpad-research.md` §G.1.)
- No MCP server implementation. `mcp/` stays a stub directory.
- No local / on-device embedding models. No `torch` in dependencies.
- No multi-modal inputs (images, audio). Text only.
- No authentication/authorization for the local UI — it binds to localhost only.
- No telemetry or usage reporting in this change.
- No packaging for IDEs other than Claude Code.

## Decisions

### D1. Skill-first, CLI-backed, MCP deferred

The skill's `SKILL.md` orchestrates phases but delegates every imperative operation to `zilliz_ops.py` (a Typer CLI). The CLI is also usable by humans and by a future MCP server without change. The MCP path is not built now but the CLI's shape is designed so a thin MCP wrapper can expose each subcommand as a tool later.

_Alternative considered:_ Put logic inline in skill scripts (what OpenSearch Launchpad does for its Agent Skill Path). Rejected — a CLI is more testable, easier to invoke outside an agent, and keeps the skill content short.

### D2. `lib/` is organized by capability, not by phase

`client.py`, `operations.py`, `ingest.py`, `search.py`, `samples.py` are reusable primitives. Phases compose them. This matters because Phase 5 (Evaluate) will reuse `search.py` heavily and Phase 6 (Deploy) will reuse `client.py` — phase-shaped modules would force duplication later.

_Alternative considered:_ One file per phase (`phase1_collect.py`, `phase2_configure.py`, …). Rejected as above.

### D3. URI-first connection abstraction

`MilvusClient` factory takes a single `uri` argument and detects target (local Standalone vs Zilliz Cloud vs anything else) from URL scheme/host. Token is resolved from env (`ZILLIZ_TOKEN` / `MILVUS_TOKEN`) based on the URI's host pattern. This keeps business logic free of "am I local or cloud" branches.

_Alternative considered:_ Separate `LocalClient` / `CloudClient` classes. Rejected — user-visible surface should be one.

### D4. Embedding is API-only

Only HTTP-API providers (OpenAI, Voyage, Cohere, Zilliz BYOM endpoint) are supported. `lib/ingest.py` has a small strategy pattern keyed on provider name. No `sentence-transformers` or `torch` in deps.

_Alternative considered:_ Support local BGE via `sentence-transformers`. Rejected for MVP — doubles install size, needs GPU for decent throughput, and conflicts with the "laptop-friendly" positioning. Re-visitable post-MVP.

### D5. Credential resolution order: env → agent dialogue

`lib/credentials.py` exposes a single `resolve(key: str, *, prompt_if_missing=True) -> str`. It:
1. Reads `os.environ`.
2. If unset and the skill is running in an agent context, raises a `MissingCredentialError` with a structured payload; the skill catches this and asks the user in dialogue, then re-invokes with the value piped via env.

This keeps CLI scriptable outside an agent (no stdin prompts blocking automation).

### D6. Plan and run artifacts under `scripts/runs/<timestamp>/`

Each Plan invocation writes `plan.json` (machine) + `plan.md` (human). Phase 4 Execute reads `plan.json`. Phase 5 Evaluate (future) appends `eval_report.{json,md}` into the same timestamp directory. This creates a per-run audit trail without requiring a database or an opinionated spec framework.

### D7. Phase 4 Execute is idempotent where possible

- Collection creation: skip if exists with compatible schema, fail loudly if schema differs.
- Data insert: deterministic primary keys (hash of source line) so reruns are no-ops.
- Index build: check if present with same params, else drop + rebuild.

Users will rerun Execute during iteration, so this matters for UX.

### D8. Demo UI is a standalone Next.js app

`scripts/ui/` is a complete Next.js project (not a script that emits HTML). It consumes a small FastAPI sidecar (`lib/ui.py` → `uvicorn` on a local port) that wraps `lib/search.py`. Separation means a user can swap the UI out without touching search logic.

_Alternative considered:_ Serve HTML directly from Python (Jinja + htmx). Rejected — user chose Next.js TS explicitly to align with the wider Zilliz Cloud front-end stack.

### D9. Reference knowledge layout mirrors Claude Code's skill conventions

```
references/
  deploy-*.md               # per-target deployment notes
  cli-reference.md          # pymilvus + zilliz-cli cheat sheet
  knowledge/
    dense_embedding_models.md
    sparse_embedding_models.md
    hybrid_search_guide.md
    reranker_guide.md
    document_processing.md
    index_tuning.md
    schema_design.md
    rag_templates.md
    evaluation_guide.md
  observability/            # Phase 6 / Ops territory, included as reference
    metrics.md
    query-analysis.md
```

Phase → reference mapping is documented in `SKILL.md`. The agent loads only the references for the phase it's in.

### D10. Testing strategy

- `tests/test_plan.py`: golden-file tests over the Plan decision tree — given a requirement profile, assert the produced `plan.json`.
- `tests/test_execute.py`: E2E against Milvus Standalone in a throwaway docker-compose stack; fixture loads `sample_data/movies.jsonl` end-to-end.
- No integration tests against Zilliz Cloud in CI (would leak credentials / cost money). Contributors run them locally with `pytest -m cloud`.

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| Embedding API costs during CI → pricey to run full E2E tests per PR | Mock embedding in unit tests; use a tiny fixture (50 docs) in the single E2E suite; mark Zilliz Cloud tests with `@pytest.mark.cloud` and exclude from CI. |
| Docker requirement excludes users without local Docker | Acceptable for MVP. Lite support later. `start_milvus.sh` gives a clear "install Docker" error if missing. |
| `SKILL.md` creeping past 500 lines as phases grow | Enforce in CI with a simple line-count check. Push detail into `references/`. |
| Plan decision tree producing weird combinations (e.g., DiskANN on 500 rows) | Constraint layer in `lib/plan.py` with hard rules; golden tests guard regressions. |
| Next.js UI adds a JS toolchain to a Python project | Isolated under `scripts/ui/`; CI installs Node only for the UI lane. |
| Credential prompting in agent dialogue could be confusing | Error payload includes the exact `export` command to set the variable, so the agent can coach the user precisely. |
| API embedding vendor lock-in | Strategy-pattern registry in `lib/ingest.py`; adding a provider is ~30 lines. |
| `runs/<timestamp>/` directory growing unbounded | Simple `--keep-last N` flag on the CLI; not enforced by default but available. |

## Migration Plan

Not applicable — greenfield. Rollback is `git revert`.

## Open Questions

1. **Sample dataset licensing**: `movies.jsonl` and BEIR-scifact-mini — confirm both are redistributable under Apache-2.0. If not, pick alternatives.
2. **Default chunking strategy**: fixed 512-token? Recursive with separators? Needs a reasonable default in `document_processing.md`.
3. **Reranker cost surface**: Cohere rerank is paid per 1k docs — should Plan recommend it by default for small datasets, or require opt-in? Leaning opt-in.
4. **Concurrency backend for latency tests** (relevant for Phase 5, but `lib/search.py` shape decided now): asyncio + `httpx.AsyncClient`, or subprocess out to `locust`? Leaning asyncio native.
5. **Skill distribution on npm**: registering the `zilliztech` org scope and publishing the skill package — coordinate with org admins.
