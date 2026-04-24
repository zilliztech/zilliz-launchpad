---
name: zilliz-launchpad
description: Turn a sample document into a running Milvus / Zilliz Cloud search app in minutes. Guide the user through Collect → Configure → Plan → Execute and start a local search UI.
---

# zilliz-launchpad

You are running the **zilliz-launchpad** skill. Your job is to take a user from *"I have some documents I want to search"* to *"a working search app on Milvus / Zilliz Cloud, scored for quality, running in production"* with as little friction as possible.

The skill has **six phases**:

1. **Collect** — analyze a sample file, detect its shape, suggest primary-key and text fields
2. **Configure** — capture the user's intent (use case, dataset size, deployment target, preferences)
3. **Plan** — deterministically design collection schema, embedding, index, and pipeline
4. **Execute** — create the collection, ingest data, start a Next.js demo UI
5. **Evaluate** — score retrieval / latency / RAG quality; compare plan variants side-by-side
6. **Deploy** — promote the local Execute run to Zilliz Cloud with observability wired

Every imperative action goes through the CLI `scripts/zilliz_ops.py`. Phases write outputs into `scripts/runs/<utc-iso>/`.

## Phase flow contract

Do not start a phase before the previous one has produced its artifact. If a phase exits non-zero, surface the error and stop. Each phase can be rerun — the CLI is idempotent.

| Phase | CLI subcommand | Required input | Output file |
| ---   | ---            | ---            | ---         |
| 1 Collect    | `collect`    | `--sample <name>` or `--input <path>` | `collect.json` |
| 2 Configure  | `configure`  | `--from-json <file>` or flags          | `configure.json` |
| 3 Plan       | `plan`       | run dir with collect + configure       | `plan.json`, `plan.md` |
| 4 Execute    | `execute`    | run dir with plan                      | `execute.json` (+ live Milvus + running sidecar) |
| 5 Evaluate   | `evaluate`   | run dir with execute                   | `eval_report.json`, `eval_report.md` |
| 6 Deploy     | `deploy`     | run dir with execute + Cloud credentials | `deploy.json` (+ `observability.json`) |

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

## Optional: `zilliz` CLI

Install the [zilliz CLI](https://github.com/zilliztech/zilliz-cli) (≥ 0.3.0) to unlock three Cloud-only enhancements. The CLI is **never required** for local Milvus.

| Phase | With `zilliz` on PATH | Without |
| ---   | ---                   | ---     |
| 2 Configure | `zilliz cluster list` → pick a cluster; writes `cluster_id` to `configure.json` | prompt for URI + token |
| 4 Execute pre-flight | `zilliz cluster describe` gates on `RUNNING`; fails fast with `cluster_not_ready` | skipped |
| 4 Execute bulk | `zilliz import create` for corpora above `bulk_import_threshold` (default 100k) | client-side upsert |
| 2/4 Token fallback | `zilliz auth whoami` returns a scoped token when `ZILLIZ_TOKEN` is unset | `missing_credential` |
| 6 Deploy create | `zilliz cluster create` provisions a new cluster when `--create --confirm` is passed | prompt for `--cluster-id` of an existing cluster |
| 6 Deploy bulk | `zilliz import create` for corpora above the plan's threshold | client-side upsert |

## Reference lazy-loading

Load only the references you need for the current phase. Do not prefetch all of `references/` — that wastes context.

| Phase | Load from `references/` |
| ---   | --- |
| 1 Collect    | `knowledge/document_processing.md` |
| 2 Configure  | `knowledge/rag_templates.md`, `deploy-*.md` (match the user's target) |
| 3 Plan       | `knowledge/dense_embedding_models.md`, `knowledge/sparse_embedding_models.md`, `knowledge/index_tuning.md`, `knowledge/schema_design.md`, `knowledge/hybrid_search_guide.md`, `knowledge/reranker_guide.md`, `knowledge/image_embedding_models.md` (image-search only) |
| 4 Execute    | `cli-reference.md` |
| 5 Evaluate   | `knowledge/evaluation_guide.md` |
| 6 Deploy     | `observability/metrics.md`, `observability/query-analysis.md`, `deploy-*.md` (match the target) |
| Ops (any)    | `ops-attu.md` — when the user wants to inspect, debug, or administer a Milvus cluster beyond what the CLI covers |

## Optional: Attu ops UI

Attu is an opt-in admin UI for developers and ops, not a replacement for the Next.js demo UI. Bring it up with `./start_milvus.sh attu up` (serves on `http://localhost:8000`, bound to loopback) when the user needs to verify ingest, drill into eval bad-cases, or run Cloud operations Attu covers but the CLI does not. See `references/ops-attu.md` for playbooks and Cloud connection instructions.

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

# Phase 5 — derived smoke eval, or with --qrels / --compare for a real eval
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py evaluate

# Phase 6 — promote to an existing Cloud cluster, or --create --confirm to provision
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py deploy --cluster-id <id>
```

On success of Phase 4, start the demo UI:

```bash
cd skills/zilliz-launchpad/scripts/ui
pnpm install
pnpm dev
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
| `missing_dependency`  | optional extra not installed (e.g. image-search needs `[multimodal]`); surface `install_hint` verbatim and ask the user to install |
| `schema_conflict`     | a collection exists with a different schema; offer to drop or to rename the plan's `collection_name` |
| `sparse_unavailable`  | the user picked hybrid/sparse but the collection was built without sparse; offer to rebuild |
| `invalid_profile`     | Phase 3 got a malformed input; go back to Configure |
| `backend_unsupported` | requested index (e.g. DiskANN) isn't available on the target; pick a fallback from `index_tuning.md` |
| `zilliz_cli_missing`  | a Cloud-only feature needs `zilliz`; point at `install_url` in the payload |
| `zilliz_cli_auth`     | CLI present but not logged in; tell the user to run `zilliz auth login` |
| `cluster_not_ready`   | pre-flight found a non-RUNNING cluster; surface `state` + `remediation` verbatim |
| `qrels_missing`       | comparison mode needs labels; ask the user for `--qrels <path>` |
| `judge_unavailable`   | `--judge-llm` requested but the provider's API key env var is unset |
| `cluster_create_failed` | surface `stderr` and `exit_code` verbatim; often a quota or region issue |
| `bulk_import_failed`  | include `job_id` when telling the user to check their Cloud console |
| `destructive_without_confirm` | `deploy --create` needs `--confirm`; summarise projected cost before re-running |

## What this skill does *not* do

- No Milvus Lite (only Standalone + Zilliz Cloud)
- No MCP server (the CLI is designed so one can be added later without touching `lib/`)
- No on-device embedding — API providers only
- No video — image-to-image supported (CLI `--query-image`, UI upload/drag-and-drop), frame-sampling not yet
- No IDE integration beyond Claude Code

If the user asks for any of the above, politely say it's on the roadmap and keep scope tight.

## Tone

Be concrete. When the user asks an open question (e.g., "should I use HNSW or DiskANN?"), consult the relevant reference in `knowledge/`, state a recommendation with one-line reasoning, and proceed. Don't pile up options without a pick.
