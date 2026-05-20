# zilliz-launchpad

From a sample document to a production-deployed search app on Milvus / Zilliz Cloud — in minutes.
An AI-guided scaffold delivered as an agent skill. 

Whether you're new to Milvus or just tired of boilerplate — pick a file, run six steps, ship.

## What you get

| Artifact | Description |
|----------|-------------|
| Milvus Collection | Schema auto-designed from your data |
| Vector + sparse fields | OpenAI / Voyage / Cohere / BYOM (text); CLIP / Voyage multimodal (image) |
| `plan.md` | Every decision recorded — reviewable and reproducible |
| Next.js Demo UI | Hybrid search list **or** thumbnail gallery (auto-detected) at `localhost:3000` |
| `eval_report.md` | recall@10, p50/p95/p99 latency, optional RAG quality metrics |
| `deploy.json` | Zilliz Cloud deployment record, resumable on rerun |

## Requirements

- **Agent** — Claude Code, Copilot CLI, Gemini CLI, or any [`skills`](https://github.com/superagent-ai/skills)-compatible agent
- **Python ≥ 3.11** + [`uv`](https://github.com/astral-sh/uv)
- **Node.js ≥ 18** + pnpm
- **Docker** (local Standalone) or a Zilliz Cloud account
- **API key** — OpenAI, Voyage, Cohere, or Zilliz BYOM
- **Optional** — [`zilliz` CLI](https://github.com/zilliztech/zilliz-cli) for Cloud auto-discovery and bulk import
- **Optional** — `ffmpeg` on `PATH` (needed for video-search frame sampling; scene-change sampling shells out to it directly)

## Install

The launchpad ships as an agent skill. Install it with the [`skills`](https://github.com/superagent-ai/skills) CLI, which discovers skills under `skills/` and symlinks them into your agent's skill directory (Claude Code, Copilot CLI, Gemini CLI, Cursor, OpenCode, Codex):

```bash
npx skills add zilliztech/zilliz-launchpad
```

Once installed, open your agent in this repo and say something like *"use zilliz-launchpad to index this file"* — the skill drives the six phases end-to-end (stopping at Phase 4 by default; Phases 5 and 6 kick in when you ask), installing Python deps, bringing up Milvus, and prompting for any missing API keys as it goes.

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

### MCP install path

For non-skill-aware hosts (Cursor, Claude Desktop, generic MCP clients), the launchpad
also ships an MCP server that exposes the same six phases as MCP tools. Install the
`mcp` extra and launch the stdio server:

```bash
uv sync --extra mcp
uv run python -m launchpad_mcp.server
```

See [`mcp/README.md`](mcp/README.md) for the full tool catalog, the structured error
envelope, and a host-registration snippet.

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

## Walkthrough — the six phases

The launchpad is organized as six CLI subcommands. Each writes a single artifact to `skills/zilliz-launchpad/scripts/runs/<utc-timestamp>/`. You can rerun any phase; nothing is destructive until Phase 4 touches Milvus, and Phase 6 is gated behind an explicit `--confirm` before it creates any Cloud resources.

```
collect ──▶ configure ──▶ plan ──▶ execute ──▶ evaluate ──▶ deploy
                                       │           │           │
                                       │           │           └─ Zilliz Cloud + deploy.json
                                       │           └─ eval_report.{json,md}
                                       └─ Milvus collection + demo UI
```

We'll use the bundled `movies` sample (20 short fictional plot summaries) throughout.

---

### Phase 1 — Collect: analyze your data

Looks at your file, infers field types, picks a candidate primary key and text field, and writes `collect.json`.

```bash
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py collect --sample movies
# or: --input path/to/your.jsonl
# or: --input path/to/your.pdf      # one record per page (requires `.[documents]` extra)
# or: --input path/to/notes.md      # whole file; add --split-markdown-headings for `## ` sections
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

> **Image search?** Point `--input` at a directory of images (`.jpg/.png/.webp/.gif`):
> ```bash
> uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py collect --input ./photos/
> uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py configure \
>     --use-case image-search --dataset-size 64000 --deployment local-standalone
> ```
> Phase 1 walks the directory, reads EXIF, encodes a thumbnail per image. Phase 3 then picks `clip-local` (open-source ViT-B/32, runs on CPU/MPS/CUDA, no API key) by default — install the optional extra first: `uv pip install -e '.[multimodal]'`. The Next.js UI auto-switches to a thumbnail gallery. See [issue #14](https://github.com/zilliztech/zilliz-launchpad/issues/14) for the MVP scope.
>
> **Search by example (image → image).** After Phase 4 Execute builds the collection you can query with another image instead of a text phrase. In the demo UI (`pnpm dev` from `scripts/ui/`) click **Search by image…** next to the text box — or drop an image anywhere on the page — to find visually similar images in your collection. Uploads are capped at 10 MB per request. For a CLI smoke:
> ```bash
> uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py evaluate \
>     --query-image ./query/my_dog.jpg
> ```
> This prints the top-10 ranked primary keys with scores and is the fastest way to confirm image-to-image is wired. For a labelled eval, mix image-to-image rows into your qrels file (one row per line):
> ```json
> {"query_image_path": "query/sunset.jpg", "expected_image_ids": ["photos/sky1.jpg", "photos/beach2.jpg"]}
> ```
> and pass the file to `evaluate --qrels path/to/qrels.jsonl` to get recall / MRR / NDCG against your ground truth. Queries against a Voyage-multimodal-backed collection (`embedding_preference: voyage-multimodal-3`) call the Voyage API per query and bill to `VOYAGE_API_KEY`; CLIP-local stays free. See [issue #15](https://github.com/zilliztech/zilliz-launchpad/issues/15).
>
> **Video search?** Point `--input` at a directory of `.mp4 / .mov / .mkv / .webm` clips:
> ```bash
> uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py collect --input ./clips/
> uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py configure \
>     --use-case video-search --dataset-size 25 --deployment local-standalone
> ```
> Phase 1 samples a frame every 2 seconds (default) using PyAV, capped at 600 frames per video, and writes JPEGs under `<run-dir>/frames/`. Phase 3 reuses the same CLIP path image-search uses (or `voyage-multimodal-3` on override) and adds two scalar fields (`video_path`, `t_seconds`) so the UI can deep-link playback. Phase 4 fans out per video, embedding frames in batches with resumable per-video tracking (`processed_videos[]`). The demo UI renders each unique video as a card with the top-scoring frame as the primary thumbnail plus clustered secondary frames under it; clicking mounts an inline `<video>` element seeked to the matched timestamp. Sidecar static-serves videos under `/videos/…`; set `LAUNCHPAD_VIDEO_STATIC_ROOT` to point at your source tree if it lives outside the run-directory parent. Tune sampling density with `--frame-interval-seconds`, `--max-frames-per-video`, and `--sampling-strategy {every_n_seconds, scene_change}` (scene-change needs `ffmpeg` on `PATH`). Evaluation reports both `recall@k (frame)` and `recall@k (video)` so you can see whether the right moment (frame) or the right clip (video) is being retrieved. See [issue #16](https://github.com/zilliztech/zilliz-launchpad/issues/16). Voyage multimodal ingest bills per-frame; for videos default to CLIP-local unless you explicitly opt in. The sidecar upload cap remains 10 MB per request (inherited from image-search).

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
| `--use-case` | `rag`, `semantic-search`, `hybrid-search`, `recommendation`, `image-search`, `video-search` |
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

---

### Phase 5 — Evaluate: score retrieval, latency, RAG quality

Runs a query set against the live collection and writes `eval_report.{json,md}`. Three query-set tiers — pick whichever matches the labels you have:

```bash
# Derived smoke eval: no labels required, samples 25 docs from your corpus
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py evaluate

# Labelled eval: recall@10, MRR@10, NDCG@10
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py evaluate --qrels qrels.jsonl

# Opt-in RAG quality via ragas (needs a generator-model API key)
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py evaluate \
    --qrels qrels.jsonl --judge-llm openai:gpt-4o-mini
```

Example `eval_report.md`:

```markdown
# Evaluation Report — 2026-04-23T10-15-02Z-execute

- Query count: 25
- Derived query set: **true**

## Decision table

| variant | recall@10 | p95 (ms) | faithfulness | cost/query |
| --- | --- | --- | --- | --- |
| base | 0.920 | 42.7 | — | — |

> Note: queries were derived from the corpus...
```

**Comparison mode** — re-run the same query set against alternative plan variants (swap embedding model, index params, hybrid on/off, reranker on/off) and get a decision table. Capped at 6 variants by default:

```bash
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py evaluate \
    --qrels qrels.jsonl --compare variants.yaml
```

```yaml
# variants.yaml
variants:
  - name: small-m
    overrides: { index: { params: { M: 8 } } }
  - name: voyage
    overrides: { embedding: { model: voyage-3 } }
  - name: no-hybrid
    overrides: { hybrid: false }
```

See [`skills/zilliz-launchpad/references/knowledge/evaluation_guide.md`](skills/zilliz-launchpad/references/knowledge/evaluation_guide.md) for the full metric contract.

---

### Phase 6 — Deploy: promote to Zilliz Cloud

Recreates the plan's collection + index on a Zilliz Cloud cluster, ingests data (bulk import above the plan's threshold, client-side below), and writes `deploy.json` with observability pointers and a resumable state machine. Snapshots after every transition, so a rerun picks up where a failure left off.

```bash
# Target an existing cluster
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py deploy --cluster-id <id>

# Or provision a new one — --confirm is required because this bills real money
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py deploy --create --confirm
```

What it does:

1. **Preflight** — checks the `zilliz` CLI is installed + authenticated (`zilliz auth whoami`), and when targeting an existing cluster, verifies it's `RUNNING`
2. **Provision** (with `--create`) — calls `zilliz cluster create` with plan/region from `configure.json`, polls `zilliz cluster describe` with exponential backoff until `RUNNING`
3. **Collection + index** — reuses Phase 4's idempotent `create_collection` + `create_index`; surfaces `schema_conflict` if an incompatible collection already exists on the cluster
4. **Ingest** — routes through `zilliz import create` for corpora above the plan's `bulk_import_threshold` (default 100k), falls back to client-side upsert if bulk fails
5. **Observability** — records the Grafana dashboard URL (from `cluster describe`) in `deploy.json.observability.grafana_dashboard` and appends a post-ingest snapshot to `observability.json`

`deploy.json` carries the fixed schema the spec guarantees: `cluster_id`, `cluster_uri`, `token_source`, `collection_name`, `ingest_mode`, `ingest_row_count`, `ingest_status`, `observability`, `timestamps`. If Phase 6 stops mid-run (cluster provisioned, ingest failing), a rerun with `--cluster-id <id>` skips the already-ready steps and retries only what's left.

## Example prompts

The CLI is the seam, but day-to-day you'll talk to the skill in natural language inside your agent. Here are concrete prompts that map cleanly to the six phases — each one shows what the skill runs under the hood.

### 1. First time — just run the bundled sample

> *"Use zilliz-launchpad with the bundled `movies` sample so I can see the whole flow end-to-end."*

Skill runs Phases 1–4 against `sample_data/movies.jsonl` (`--use-case rag --dataset-size 20 --deployment local-standalone`) and starts the demo UI. **Best first invocation** — proves your environment is wired before you point it at real data. Once you've clicked around the UI, ask *"now run Phase 5 in derived mode"* for a quick latency+recall smoke.

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

Skill runs `execute --append --run-dir <previous-run-dir> --input docs_v2.jsonl`. The append path reuses `plan.json` (no re-plan), confirms the live collection schema matches, and upserts only the new rows. Results land in a fresh `execute_append.json` artifact so the original `execute.json` stays untouched (and a second append produces `execute_append.2.json`, etc.). If the new file's fields don't match the planned schema, it stops with `schema_conflict` and tells you how to resolve it.

### 7. Recover from `schema_conflict`

> *"Phase 4 just failed with `schema_conflict`. What do I do?"*

Skill parses the JSON error envelope on stderr and offers two paths: drop the existing collection (destructive — confirms first), or change `collection_name` in `plan.json` and rerun. **The general pattern**: every CLI error code in [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) maps to a remediation the skill knows how to drive — so when something goes red, just paste the error back into the conversation.

### 8. Compare embedding models before committing

> *"I have `qrels.jsonl` with 50 labelled queries. Before I commit to an embedding model, run Phase 5 with `voyage-3` and `text-embedding-3-small` side-by-side."*

Skill writes a `variants.yaml` with the two overrides, runs `evaluate --qrels qrels.jsonl --compare variants.yaml`, and shows the resulting decision table (recall@10, p95, faithfulness per variant). **Use this whenever you're about to change a plan axis and want signal, not vibes.**

### 9. Promote the local prototype to Cloud

> *"The local eval looks good. Promote this run to a new Zilliz Cloud serverless cluster."*

Skill confirms the projected cost, runs `deploy --create --confirm`, streams `zilliz cluster describe` progress to stderr while the cluster comes up, and tails `deploy.json` as the state machine advances. If anything fails mid-deploy, a rerun with `deploy --cluster-id <id>` resumes from the last checkpoint — the skill reads `deploy.json` to know which steps are already done.

### 10. Index my photos for text-to-image search

> *"I have ~5,000 photos in `~/Pictures/library`. Make them searchable by typing — I want to type 'sunset' and see sunsets."*

Skill installs the `[multimodal]` extra if it's missing, then runs:

- `collect --input ~/Pictures/library` — walks the directory, reads dimensions / EXIF / capped 256 px thumbnails per image
- `configure --use-case image-search --dataset-size 5000 --deployment local-standalone` — forces hybrid off, picks `clip-local` (ViT-B/32, 512 dim) by default
- `plan` — schema is dense-only with `image_path` as the primary key
- `execute` — downloads CLIP weights once (~150 MB), batches images through the encoder, ingests; the demo UI auto-detects the modality and renders a thumbnail grid instead of a text list

For Phase 5 add `--judge-llm openai:gpt-4o-mini` to derive captioned eval queries; cached to `derived_image_queries.jsonl` so reruns don't re-spend tokens.

## Going to Zilliz Cloud

There are two patterns — pick based on whether you want to iterate locally first.

### Pattern A — prototype locally, promote via Phase 6 (recommended)

Run Phases 1–5 against local Standalone, then `deploy` to Cloud once the eval looks good:

```bash
# Phases 1–4 against local Milvus (--deployment local-standalone)
# Phase 5 on local to sanity-check recall + latency
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py evaluate --qrels qrels.jsonl

# Phase 6 promotes to Cloud. --create provisions a new cluster; --cluster-id targets existing.
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py deploy --create --confirm
```

Phase 6 requires the [`zilliz` CLI](https://github.com/zilliztech/zilliz-cli) (≥ 0.3.0) on PATH with `zilliz auth login` done. Cluster plan (Serverless / Standard / Enterprise) and region are taken from `configure.json`.

### Pattern B — target Cloud from the start

Skip local Standalone entirely by setting the Cloud target in Phase 2:

```bash
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py configure \
    --use-case rag --dataset-size 500000 --deployment zilliz-serverless
```

With the `zilliz` CLI on PATH, the launchpad will:

- Discover your clusters (`zilliz cluster list`) and write `cluster_id` into `configure.json`
- Pre-flight the cluster state (`zilliz cluster describe`) before Phase 4 ingests anything
- Route ingestion through `zilliz import create` for corpora above the plan's `bulk_import_threshold` (default 100k rows)

Without the CLI, export `ZILLIZ_TOKEN` directly and Phase 4 falls back to client-side upsert. Phase 6's `--create` path requires the CLI — there's no paste-the-token fallback for cluster provisioning.

## Troubleshooting

See [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) for common errors (missing credentials, schema conflicts, Cloud cluster states) and their remediations. Every CLI error is a single-line JSON envelope on stderr — the `code` field maps to a row in that doc.

## License

Apache-2.0. Copyright 2026 Zilliz.
