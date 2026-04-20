---
name: zilliz-launchpad
description: Turn a sample document into a running Milvus / Zilliz Cloud search app in minutes. Guide the user through Collect → Configure → Plan → Execute and start a local search UI.
---

# zilliz-launchpad

You are running the **zilliz-launchpad** skill. Your job is to take a user from *"I have some documents I want to search"* to *"a working local search UI on Milvus / Zilliz Cloud"* with as little friction as possible.

The skill has **four phases for the MVP** (Phases 5 and 6 are future work):

1. **Collect** — analyze a sample file, detect its shape, suggest primary-key and text fields
2. **Configure** — capture the user's intent (use case, dataset size, deployment target, preferences)
3. **Plan** — deterministically design collection schema, embedding, index, and pipeline
4. **Execute** — create the collection, ingest data, start a Next.js demo UI

Every imperative action goes through the CLI `scripts/zilliz_ops.py`. Phases write outputs into `scripts/runs/<utc-iso>/`.

## Phase flow contract

Do not start a phase before the previous one has produced its artifact. If a phase exits non-zero, surface the error and stop. Each phase can be rerun — the CLI is idempotent.

| Phase | CLI subcommand | Required input | Output file |
| ---   | ---            | ---            | ---         |
| 1 Collect    | `collect`    | `--sample <name>` or `--input <path>` | `collect.json` |
| 2 Configure  | `configure`  | `--from-json <file>` or flags          | `configure.json` |
| 3 Plan       | `plan`       | run dir with collect + configure       | `plan.json`, `plan.md` |
| 4 Execute    | `execute`    | run dir with plan                      | `execute.json` (+ live Milvus + running sidecar) |

## Before anything else

Check the basics:

- Docker is running (or the user has a Zilliz Cloud URI they want to use instead)
- The user has (or can provide) an embedding API key for one of: OpenAI, Voyage, Cohere, Zilliz BYOM

If the user is going local for the first time, bring Milvus up:

```bash
cd skills/zilliz-launchpad/scripts
./start_milvus.sh up
```

## Credentials

Resolve secrets from env **first**. Only if a variable is missing and needed by the current phase, prompt the user in dialogue and re-invoke with it exported. Typical variables:

- `OPENAI_API_KEY` — OpenAI embeddings / reranker
- `VOYAGE_API_KEY` — Voyage embeddings
- `COHERE_API_KEY` — Cohere embeddings / rerank
- `ZILLIZ_TOKEN` — Zilliz Cloud authentication
- `ZILLIZ_BYOM_URL`, `ZILLIZ_BYOM_KEY` — Zilliz BYOM endpoint

When the CLI fails with `{"code":"missing_credential",...}`, the payload includes the variable name and an `export_hint`. Use that hint verbatim when asking the user.

## Reference lazy-loading

Load only the references you need for the current phase. Do not prefetch all of `references/` — that wastes context.

| Phase | Load from `references/` |
| ---   | --- |
| 1 Collect    | `knowledge/document_processing.md` |
| 2 Configure  | `knowledge/rag_templates.md`, `deploy-*.md` (match the user's target) |
| 3 Plan       | `knowledge/dense_embedding_models.md`, `knowledge/sparse_embedding_models.md`, `knowledge/index_tuning.md`, `knowledge/schema_design.md`, `knowledge/hybrid_search_guide.md`, `knowledge/reranker_guide.md` |
| 4 Execute    | `cli-reference.md` |

`observability/` is Phase 6 (Deploy) territory — do not load in MVP.

## Invoking phases

Assume the working directory is the repo root.

```bash
# Phase 1 — sample data (or --input your.jsonl)
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py collect --sample movies

# Phase 2 — take defaults with overrides
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py configure \
    --use-case rag --dataset-size 20 --deployment local-standalone

# Phase 3
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py plan

# Phase 4
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py execute --sample movies
```

On success of Phase 4, start the demo UI:

```bash
cd skills/zilliz-launchpad/scripts/ui
npm install
npm run dev
# open http://localhost:3000
```

## Error envelopes

Every CLI error arrives on stderr as a single-line JSON object:

```json
{"code": "missing_credential", "message": "...", "env_var": "OPENAI_API_KEY", "export_hint": "export OPENAI_API_KEY=..."}
```

Known codes and how to react:

| code | action |
| --- | --- |
| `missing_credential`  | prompt the user for the value; ask them to `export` it; retry |
| `schema_conflict`     | a collection exists with a different schema; offer to drop or to rename the plan's `collection_name` |
| `sparse_unavailable`  | the user picked hybrid/sparse but the collection was built without sparse; offer to rebuild |
| `invalid_profile`     | Phase 3 got a malformed input; go back to Configure |
| `backend_unsupported` | requested index (e.g. DiskANN) isn't available on the target; pick a fallback from `index_tuning.md` |

## What this MVP does *not* do

- No Milvus Lite (only Standalone + Zilliz Cloud)
- No MCP server (the CLI is designed so one can be added later without touching `lib/`)
- No on-device embedding — API providers only
- No Evaluate or Deploy phase (Phases 5–6 are future work)
- No multi-modal data — text only
- No IDE integration beyond Claude Code

If the user asks for any of the above, politely say it's on the roadmap and keep scope tight.

## Tone

Be concrete. When the user asks an open question (e.g., "should I use HNSW or DiskANN?"), consult the relevant reference in `knowledge/`, state a recommendation with one-line reasoning, and proceed. Don't pile up options without a pick.
