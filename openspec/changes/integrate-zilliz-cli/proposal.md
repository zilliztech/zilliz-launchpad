## Why

`zilliz-launchpad` ships with an MVP that operates Zilliz Cloud entirely through `pymilvus` data-plane calls. That covers create-collection / index / insert / search, but leaves three user-facing gaps:

1. **Control plane is manual.** Creating, listing, describing, or deleting a Zilliz Cloud cluster is not automated — `deploy-serverless.md` tells users to click around at <cloud.zilliz.com>. This blocks the future Phase 6 Deploy and also hurts Phase 2 Configure, which cannot auto-discover the user's existing clusters.
2. **Bulk ingestion is slow.** Phase 4 Execute upserts client-side in batches. For corpora above ~100k rows this is 10-30× slower than Zilliz Cloud's native `import` job API.
3. **No pre-flight check.** When a user targets Cloud, Phase 4 opens a pymilvus connection and only fails if the cluster is paused or unreachable. The error surface is cryptic; a `zilliz cluster describe` pre-check would let us fail fast with actionable guidance.

We already own a CLI that solves all three: <https://github.com/zilliztech/zilliz-cli>. This change wires `zilliz-cli` into the launchpad as an **optional capability** — present, it enhances Cloud flows; absent, the existing pymilvus paths keep working unchanged.

## What Changes

- Introduce a thin `lib/zilliz_cli.py` subprocess wrapper that detects whether `zilliz` is on `PATH`, whether the user is logged in, and which project/region is active.
- Phase 2 Configure: when `deployment_target` is a Zilliz Cloud target, attempt `zilliz cluster list` and offer the user a menu of existing clusters. Fallback to the current "type your URI + token" flow when the CLI is missing or unauthed.
- Phase 4 Execute: when the target URI is a Zilliz Cloud host, run `zilliz cluster describe` as a pre-flight. Fail fast with a structured `{"code": "cluster_not_ready", ...}` envelope if the cluster is paused / provisioning / deleting.
- Phase 4 Execute: route ingestion above a size threshold (default 100,000 rows, configurable in the plan) through `zilliz import create` instead of client-side upsert. Smaller datasets still use the current path.
- Phase 2 / Phase 4: prefer `ZILLIZ_TOKEN` from env, but when unset, read it from the active `zilliz` CLI profile (`zilliz auth whoami --json`) before prompting the user.
- Add a Phase 6 placeholder module (`lib/phases/deploy.py`) that uses `zilliz cluster create` end-to-end. This change **does not implement Phase 6**; it scaffolds the call sites so a later change can light up Deploy cheaply.
- Document the CLI dependency clearly: required for `deploy` / bulk-import / auto-discovery; **not** required for MVP Phases 1–4 against local Milvus Standalone.

## Capabilities

### New Capabilities

- `zilliz-cli-integration`: CLI detection, subprocess wrapper with JSON-first output parsing, credential fallback to the CLI session, cluster auto-discovery, pre-flight state check, bulk-import routing, and Phase 6 Deploy scaffold. Call-site changes to the existing `requirements-gathering`, `plan-and-execute`, and `skill-orchestration` capabilities are tracked in `tasks.md` and exercised through the new capability's scenarios — they are **behavioral additions**, not requirement changes to the existing specs.

### Modified Capabilities

_None._ The existing capabilities (`requirements-gathering`, `plan-and-execute`, `skill-orchestration`) keep their current requirements. This change adds new capability `zilliz-cli-integration` whose scenarios describe what happens *in addition* at the existing call sites. This change depends on `mvp-phases-1-to-4` being archived first.

## Impact

- **New Python module**: `scripts/lib/zilliz_cli.py`, no new Python dependency (uses stdlib `subprocess`).
- **External dependency** (optional): `zilliz` CLI on user's PATH. Minimum version pinned in `zilliz_cli.py`.
- **Affected code**:
  - `scripts/lib/phases/configure.py` — cluster auto-discovery
  - `scripts/lib/phases/execute.py` — pre-flight + bulk import routing
  - `scripts/lib/credentials.py` — fallback source for `ZILLIZ_TOKEN`
  - `scripts/lib/phases/deploy.py` — new, placeholder only
  - `scripts/lib/errors.py` — new `ClusterNotReadyError`, `ZillizCliMissingError`, `ZillizCliAuthError`
- **Affected docs**: `SKILL.md`, `references/deploy-serverless.md`, `references/deploy-dedicated.md`, `references/deploy-byoc.md`, `references/cli-reference.md`, `README.md` Quick Start.
- **Tests**: new `tests/test_zilliz_cli.py` with mocked subprocess calls; `tests/test_plan.py` gains a threshold-behavior case.
- **Out of scope**: implementing Phase 6 Deploy end-to-end; replacing pymilvus for data-plane ops; supporting `zilliz-cli` versions pre-dating the JSON output flags.
