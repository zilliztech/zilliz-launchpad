# zilliz-launchpad

> **From a sample document to a running search app on Milvus / Zilliz Cloud — in minutes.**
> An opinionated, AI-guided scaffold delivered as a Claude Code skill.

`zilliz-launchpad` walks you through four guided phases and hands you a working Next.js search UI at the end:

1. **Collect** — analyze your data, detect schema
2. **Configure** — capture intent (use case, size, deployment target)
3. **Plan** — deterministically decide schema + embedding + index + pipeline
4. **Execute** — apply the plan, ingest, start a local demo UI

Phases 5 (Evaluate) and 6 (Deploy) are on the roadmap.

## Requirements

- Python ≥ 3.11 and [`uv`](https://github.com/astral-sh/uv)
- Node.js ≥ 18 (for the demo UI)
- Docker (for local Milvus Standalone) **or** a Zilliz Cloud account
- An API key from one of: OpenAI · Voyage · Cohere · Zilliz BYOM

## Quick start — local Milvus

```bash
# 1. Install the skill into Claude Code
npx skills add zilliztech/zilliz-launchpad -a claude-code

# 2. Start local Milvus Standalone
./skills/zilliz-launchpad/scripts/start_milvus.sh up

# 3. Set at least one embedding provider key
export OPENAI_API_KEY=<your-key>

# 4. Run the four phases against the bundled sample
uv sync
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py collect   --sample movies
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py configure --use-case rag --dataset-size 20 --deployment local-standalone
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py plan
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py execute   --sample movies
# → ✓ Top-1: m007 score=0.87...

# 5. Open the demo UI
cd skills/zilliz-launchpad/scripts/ui && npm install && npm run dev
# → http://localhost:3000
```

The simplest path is to invoke the skill from Claude Code and let it drive each phase for you — the commands above are what it runs under the hood.

## Using Zilliz Cloud

Local Milvus Standalone needs no extra tooling. For Zilliz Cloud, install the optional [zilliz CLI](https://github.com/zilliztech/zilliz-cli) (≥ 0.3.0) — the launchpad uses it for cluster auto-discovery, pre-flight checks, and bulk import. Without the CLI the Cloud path still works; you just paste the URI + token manually.

```bash
# 1. Install + log in (one-time)
zilliz auth login
zilliz cluster list   # sanity check

# 2. Configure for Cloud — the launchpad discovers your clusters automatically
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py configure \
    --use-case rag --dataset-size 500000 --deployment zilliz-serverless

# 3. Plan + Execute as usual — Phase 4 pre-flights the cluster and, for
#    corpora above 100k rows, routes ingestion through `zilliz import create`.
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py plan
uv run python skills/zilliz-launchpad/scripts/zilliz_ops.py execute --input big.jsonl
```

Without the CLI, export `ZILLIZ_TOKEN` directly and Phase 4 falls back to client-side upsert.

## Supported targets

- **Local** — Milvus Standalone via docker-compose
- **Cloud** — Zilliz Serverless / Dedicated / BYOC (connection only; automated Deploy is future work)

## Supported embedding providers

API-only, no on-device models: **OpenAI** · **Voyage** · **Cohere** · **Zilliz BYOM**.

## Troubleshooting

See [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) for common errors (missing credentials, schema conflicts, Cloud cluster states) and their remediations.

## Status

**Early** — MVP with Phases 1–4. No evaluation, no automatic deploy, no Milvus Lite, no on-device embeddings, text only.

## Contributing

We welcome patches, new embedding providers, and additional sample datasets. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

## License

Apache-2.0. Copyright 2026 Zilliz.
