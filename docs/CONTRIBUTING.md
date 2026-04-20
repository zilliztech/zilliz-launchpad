# Contributing

Thanks for considering a contribution. This document covers the two most common extension paths.

## Project layout

```
zilliz-launchpad/
├── skills/zilliz-launchpad/        # the agent skill
│   ├── SKILL.md                    # phase contract + tone for the agent (≤500 lines)
│   ├── skill.json                  # skill manifest
│   ├── scripts/
│   │   ├── zilliz_ops.py           # the only entry point — Typer CLI
│   │   ├── start_milvus.sh         # docker-compose helper for Standalone
│   │   ├── docker-compose.yml      # Milvus + etcd + minio
│   │   ├── lib/                    # implementation
│   │   │   ├── phases/             # collect / configure / plan / execute
│   │   │   ├── client.py           # pymilvus wrapper
│   │   │   ├── embeddings.py       # OpenAI / Voyage / Cohere / BYOM
│   │   │   ├── ingest.py           # batched upsert + bulk import
│   │   │   ├── search.py           # dense + sparse + hybrid query helpers
│   │   │   ├── zilliz_cli.py       # subprocess wrapper for the optional CLI
│   │   │   └── errors.py           # JSON error envelopes (see SKILL.md)
│   │   ├── sample_data/            # bundled JSONL samples (movies, scifact)
│   │   ├── ui/                     # Next.js demo UI (App Router, pnpm)
│   │   └── runs/                   # phase artifacts, one dir per run
│   └── references/                 # lazy-loaded knowledge for the agent
│       ├── knowledge/              # embedding / index / schema / hybrid guides
│       ├── deploy-*.md             # per-deployment notes
│       └── cli-reference.md        # CLI subcommand reference
├── docs/
│   ├── CONTRIBUTING.md
│   └── TROUBLESHOOTING.md
├── tests/                          # pytest suite for lib/
├── mcp/                            # placeholder for a future MCP server
└── pyproject.toml                  # uv-managed Python project
```

Two things worth knowing:

- **Every action goes through `scripts/zilliz_ops.py`.** It's the seam between the agent and Milvus — easy to script, easy to wrap in an MCP later.
- **`scripts/runs/<utc-iso>/` is the source of truth for a single launchpad invocation.** Delete a run directory to start over; nothing else holds state.

## Adding an embedding provider

1. Add a dataclass in `lib/embeddings.py` implementing the `EmbeddingProvider` protocol.
2. Register it in `make_embedder()` and `_PROVIDER_ENV`.
3. Add a note to `references/knowledge/dense_embedding_models.md`.
4. Update `references/requirement-profile.schema.json` to accept the new provider name.
5. Add a test in `tests/test_ingest.py` using `mocker.patch` on the HTTP client.

Run `uv run pytest -m "not cloud and not e2e"` — should pass.

## Adding a sample dataset

1. Drop a JSONL file in `scripts/sample_data/` and a `<name>.PROVENANCE.md` next to it.
2. Register it in `scripts/lib/samples.py:_DATASETS`.
3. If the dataset has qrels / queries, place them alongside and surface via `load_qrels` / `load_queries`.
4. License MUST be Apache-2.0-compatible; make this explicit in PROVENANCE.

## Making changes to the plan decision tree

`lib/phases/plan.py` is governed by `tests/test_plan.py` golden tests. Update both together — any behavioral change should come with a matching test.

## Running the full E2E

```bash
./skills/zilliz-launchpad/scripts/start_milvus.sh up
export OPENAI_API_KEY=...
uv run pytest -m e2e
./skills/zilliz-launchpad/scripts/start_milvus.sh down
```
