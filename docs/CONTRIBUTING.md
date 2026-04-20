# Contributing

Thanks for considering a contribution. This document covers the two most common extension paths.

## Project layout (quick)

```
skills/zilliz-launchpad/
├── SKILL.md                       # Claude Code skill entry (≤500 lines)
├── references/                    # Per-phase docs loaded lazily
└── scripts/
    ├── lib/                       # Capability primitives
    ├── sample_data/               # Bundled demo datasets
    ├── ui/                        # Next.js demo UI
    ├── zilliz_ops.py              # CLI entry
    └── start_milvus.sh            # docker-compose wrapper
```

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

## OpenSpec workflow

Changes to behavior should come with an OpenSpec change under `openspec/changes/`. For anything beyond a typo fix:

```bash
openspec new change <kebab-name>
# then: fill in proposal.md → design.md (if needed) → specs/**/*.md → tasks.md
```

The initial MVP lives at `openspec/changes/mvp-phases-1-to-4/`. Archive it upon first release.
