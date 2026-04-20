# zilliz-launchpad

> **AI-powered scaffold for Milvus / Zilliz Cloud search apps.** From a sample document to a running local search UI in minutes — Skill-first, opinionated, Zilliz-Cloud compatible.

Four guided phases:

1. **Collect** — analyze your data, detect schema
2. **Configure** — capture intent (use case, size, deployment target)
3. **Plan** — deterministically decide schema + embedding + index + pipeline
4. **Execute** — apply the plan, ingest, start a Next.js demo UI

Phases 5 (Evaluate) and 6 (Deploy) are on the roadmap.

## Quick start

```bash
# 1. Install the skill into Claude Code
npx skills add zilliztech/zilliz-launchpad -a claude-code

# 2. Start local Milvus Standalone
cd skills/zilliz-launchpad/scripts
./start_milvus.sh up

# 3. Set at least one embedding provider key
export OPENAI_API_KEY=<your-key>

# 4. Run the four phases against the bundled sample
uv sync
uv run python zilliz_ops.py collect   --sample movies
uv run python zilliz_ops.py configure --use-case rag --dataset-size 20 --deployment local-standalone
uv run python zilliz_ops.py plan
uv run python zilliz_ops.py execute   --sample movies
# → ✓ Top-1: m007 score=0.87...

# 5. Open the demo UI
cd ui && npm install && npm run dev
# → http://localhost:3000
```

## What you get

- `skills/zilliz-launchpad/` — the Claude Code skill (SKILL.md + references + scripts)
- `scripts/lib/` — Python capability primitives (client, ingest, search)
- `scripts/zilliz_ops.py` — CLI entry, 1:1 with phases
- `scripts/ui/` — Next.js TypeScript demo UI
- `scripts/sample_data/` — bundled fictional movies + scifact samples
- `tests/` — unit + E2E tests (Milvus Standalone)

## Supported targets

- **Local**: Milvus Standalone via docker-compose
- **Cloud**: Zilliz Serverless / Dedicated / BYOC (connection only — Deploy phase is future work)

## Supported embedding providers (API-only)

- OpenAI · Voyage · Cohere · Zilliz BYOM

## Status

**Early** — MVP Phases 1–4 implemented. No evaluation, no automatic deploy, no Milvus Lite, no local embedding. See `openspec/changes/mvp-phases-1-to-4/` for the full spec.

## License

Apache-2.0. Copyright 2026 Zilliz.
