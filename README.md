# zilliz-launchpad

> **From a sample document to a running search app on Milvus / Zilliz Cloud — in minutes.**
> An opinionated, AI-guided scaffold delivered as an agent skill.

If you've never touched Milvus, the launchpad picks every default for you (schema, embedding model, index, hybrid setup) and hands you a working Next.js search UI. If you're already a Milvus user but tired of writing the same boilerplate for every new collection, it collapses the busywork into four idempotent CLI steps.

## What you get

After running the four phases against a sample file, you have:

- A **collection** in Milvus or Zilliz Cloud, schema designed from your data
- **Vector + sparse fields** wired up, with a chosen embedding model (OpenAI / Voyage / Cohere / BYOM)
- A **deterministic plan artifact** (`plan.md`) explaining every decision — useful for review and reproducibility
- A **Next.js demo UI** at `http://localhost:3000` that runs hybrid search against your data

## Requirements

- **An agent that supports skills** — e.g., Claude Code, Copilot CLI, Gemini CLI, or any agent compatible with the [`skills`](https://github.com/superagent-ai/skills) installer
- **Python ≥ 3.11** and [`uv`](https://github.com/astral-sh/uv)
- **Node.js ≥ 18** and **pnpm** (for the demo UI)
- **Docker** (for local Milvus Standalone) **or** a Zilliz Cloud account
- An API key from one of: **OpenAI** · **Voyage** · **Cohere** · **Zilliz BYOM**
- Optional: [`zilliz` CLI](https://github.com/zilliztech/zilliz-cli) — only needed for Cloud auto-discovery and bulk import

## Install

The launchpad ships as an agent skill. Install it with the [`skills`](https://github.com/superagent-ai/skills) CLI, which discovers skills under `skills/` and symlinks them into your agent's skill directory (Claude Code, Copilot CLI, Gemini CLI, Cursor, OpenCode, Codex):

```bash
npx skills add zilliztech/zilliz-launchpad
```

Once installed, open your agent in this repo and say something like *"use zilliz-launchpad to index this file"* — the skill drives the four phases end-to-end, installing Python deps, bringing up Milvus, and prompting for any missing API keys as it goes.

### Install options

```bash
# Install to a specific agent
npx skills add zilliztech/zilliz-launchpad -a claude-code

# Install globally (available across all projects)
npx skills add zilliztech/zilliz-launchpad -g

# Install to all detected agents
npx skills add zilliztech/zilliz-launchpad --all

# List available skills before installing
npx skills add zilliztech/zilliz-launchpad --list
```

Other agent flags include `-a copilot-cli`, `-a gemini-cli`, `-a cursor`, `-a opencode`, and `-a codex`.

### Preflight (optional)

The skill drives these on demand, but you can do them ahead of time to skip a few conversational round-trips:

```bash
# Install Python deps
uv sync

# Bring up local Milvus Standalone (skip if using Zilliz Cloud)
./skills/zilliz-launchpad/scripts/start_milvus.sh up

# Export at least one embedding key
export OPENAI_API_KEY=<your-key>
```

If you plan to drive the CLI directly without an agent, run all three — the Walkthrough below assumes they're done.

## Walkthrough — the four phases

The launchpad is organized as four CLI subcommands. Each writes a single artifact to `skills/zilliz-launchpad/scripts/runs/<utc-timestamp>/`. You can rerun any phase; nothing is destructive until Phase 4 actually touches Milvus.

```
collect ──▶ configure ──▶ plan ──▶ execute ──▶ demo UI
                                       │
                                       └─ Milvus collection + ingested data
```

We'll use the bundled `movies` sample (20 short fictional plot summaries) throughout.

---

### Phase 1 — Collect: analyze your data

Looks at your file, infers field types, picks a candidate primary key and text field, and writes `collect.json`.

```bash
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py collect --sample movies
# or: --input path/to/your.jsonl
```

Output (`collect.json`, abbreviated):

```json
{
  "data_shape": "jsonl",
  "record_count_estimate": 20,
  "fields": [
    { "name": "id",    "type": "string", "avg_length": 4,   "sample_value": "m001" },
    { "name": "title", "type": "string", "avg_length": 18,  "sample_value": "The Quantum Gardener" },
    { "name": "body",  "type": "string", "avg_length": 126, "sample_value": "An astrophysicist..." },
    { "name": "year",  "type": "int",    "sample_value": 2023 },
    { "name": "genre", "type": "string", "sample_value": "sci-fi" }
  ]
}
```

> **For Milvus veterans:** this is where you'd normally hand-write a `CollectionSchema`. Skip it.

---

### Phase 2 — Configure: capture your intent

Three knobs decide everything downstream: **use case**, **dataset size**, **deployment target**.

```bash
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py configure \
    --use-case rag \
    --dataset-size 20 \
    --deployment local-standalone
```

| Flag | Common values |
| --- | --- |
| `--use-case` | `rag`, `semantic-search`, `hybrid-search`, `recommendation` |
| `--dataset-size` | row-count estimate, drives index choice |
| `--deployment` | `local-standalone`, `zilliz-serverless`, `zilliz-dedicated`, `zilliz-byoc` |

Output (`configure.json`): a normalized requirement profile used by the downstream phases.

---

### Phase 3 — Plan: deterministic decisions, no LLM

Reads `collect.json` + `configure.json` and writes both a machine plan (`plan.json`) and a human-readable explanation (`plan.md`).

```bash
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py plan
```

Example `plan.md`:

```markdown
# Launchpad Plan

- Collection: `launchpad_collection`
- Target URI: `http://localhost:19530`
- Deployment: `local-standalone`

## Schema
- Primary key: `id`
- Text field: `body`
- Vector field: `embedding` (dim 1536)
- Sparse field: `sparse`
- Extra fields: title, year, genre

## Embedding
- Provider: `openai`
- Model: `text-embedding-3-small`
- Dim: 1536

## Index
- Type: HNSW   Metric: COSINE
- Params: { "M": 16, "efConstruction": 200 }

## Rationale
- Dataset size 20 → HNSW with M=16, ef=200
- Use case 'rag' + hybrid preference 'auto' → sparse=True
- Embedding provider 'openai' model 'text-embedding-3-small' (dim 1536)
```

Read the rationale, tweak `configure.json` if you disagree, rerun `plan`. Nothing has touched Milvus yet.

---

### Phase 4 — Execute: create the collection, ingest, search

```bash
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py execute --sample movies
# → connecting to http://localhost:19530
# → creating collection 'launchpad_collection' (HNSW, COSINE, dim=1536)
# → embedding 20 rows with openai/text-embedding-3-small
# → ingested 20 / 20
# → smoke test: query "movie about parallel universes"
# → ✓ Top-1: m001 'The Quantum Gardener' score=0.87
```

What it does:

1. Connects to the URI from `plan.json` (local Milvus or Zilliz Cloud)
2. Creates the collection + index per the plan (idempotent: skips if it matches; errors with `schema_conflict` if not)
3. Embeds and upserts your data (client-side for ≤100k rows, `zilliz import` for larger corpora when the CLI is installed)
4. Runs a smoke query and prints the top-1 result

---

### Demo UI

```bash
cd skills/zilliz-launchpad/scripts/ui
pnpm install
pnpm dev
# → http://localhost:3000
```

A minimal Next.js app that calls a local `/api/search` route, which in turn hits your Milvus collection. Uses the latest run directory automatically. Hot-reload friendly — restyle it however you like.

## Example prompts

The CLI is the seam, but day-to-day you'll talk to the skill in natural language inside your agent. Here are concrete prompts that map cleanly to the four phases — each one shows what the skill runs under the hood.

### 1. First time — just run the bundled sample

> *"Use zilliz-launchpad with the bundled `movies` sample so I can see the whole flow end-to-end."*

Skill runs all four phases against `sample_data/movies.jsonl` (`--use-case rag --dataset-size 20 --deployment local-standalone`) and starts the demo UI. **Best first invocation** — proves your environment is wired before you point it at real data.

### 2. RAG over your own JSONL

> *"I have `~/data/support_tickets.jsonl` (~80k rows, fields: `ticket_id`, `subject`, `body`, `priority`). Index it for RAG on local Milvus."*

- `collect --input ~/data/support_tickets.jsonl` — infers fields, picks `body` as the text field
- `configure --use-case rag --dataset-size 80000 --deployment local-standalone`
- `plan` — sparse field on by default for RAG
- `execute` — embed + upsert + smoke query

### 3. Hybrid search over a product catalog

> *"I want hybrid search over `products.jsonl` — keyword should match SKUs and brand names exactly, semantic should match descriptions."*

Skill writes `--use-case hybrid-search` into `configure.json`. Phase 3 produces dense (HNSW) + sparse (BM25) indexes; Phase 4 embeds the description field while keeping SKU/brand as scalar fields you can filter on in the UI. **Useful when you've used sparse before but don't want to wire BM25 + dense by hand.**

### 4. Skip local Milvus, go straight to Zilliz Cloud

> *"Skip local — set up a serverless cluster on Zilliz Cloud and ingest `corpus.jsonl` (~1.2M rows) into it."*

Assumes `zilliz auth login` already done. Skill will:

- Run `zilliz cluster list`, let you pick (or auto-select your most recent)
- `configure --deployment zilliz-serverless --dataset-size 1200000`
- Pre-flight cluster state via `zilliz cluster describe`
- Detect >100k rows and route Phase 4 through `zilliz import create` instead of client-side upsert

### 5. Tweak the plan before touching Milvus

> *"The current plan picked `text-embedding-3-small`. Switch to Voyage's `voyage-3` and rerun the plan."*

Skill edits the embedding section of `configure.json` in the active run dir, reruns `plan`, and diffs the new `plan.md` against the previous one. Phase 3 is deterministic and never touches Milvus, so you can iterate freely until you're happy, then run `execute`.

### 6. Append new data to an existing collection

> *"I already ran the launchpad on `docs_v1.jsonl` last week. Now I have `docs_v2.jsonl` — append it to the same collection."*

Skill reuses the previous run dir's plan (preserving schema) and runs only `execute --input docs_v2.jsonl`. If the new file's fields don't match the planned schema, it stops with `schema_conflict` and tells you how to resolve it.

### 7. Recover from `schema_conflict`

> *"Phase 4 just failed with `schema_conflict`. What do I do?"*

Skill parses the JSON error envelope on stderr and offers two paths: drop the existing collection (destructive — confirms first), or change `collection_name` in `plan.json` and rerun. **The general pattern**: every CLI error code in [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) maps to a remediation the skill knows how to drive — so when something goes red, just paste the error back into the conversation.

## Using Zilliz Cloud instead of local Milvus

Two changes vs. the local flow:

1. Install the [`zilliz` CLI](https://github.com/zilliztech/zilliz-cli), run `zilliz auth login` once. Without it the Cloud path still works — you just paste URI + token manually.
2. Pass `--deployment zilliz-serverless` (or `zilliz-dedicated` / `zilliz-byoc`) in Phase 2:

```bash
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py configure \
    --use-case rag --dataset-size 500000 --deployment zilliz-serverless
```

With the CLI on PATH, the launchpad will:

- Discover your clusters (`zilliz cluster list`) and write `cluster_id` into `configure.json`
- Pre-flight the cluster state (`zilliz cluster describe`) before Phase 4 ingests anything
- Route ingestion through `zilliz import create` for corpora above 100k rows

Without the CLI, export `ZILLIZ_TOKEN` directly and Phase 4 falls back to client-side upsert.

## Troubleshooting

See [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) for common errors (missing credentials, schema conflicts, Cloud cluster states) and their remediations. Every CLI error is a single-line JSON envelope on stderr — the `code` field maps to a row in that doc.

## License

Apache-2.0. Copyright 2026 Zilliz.
