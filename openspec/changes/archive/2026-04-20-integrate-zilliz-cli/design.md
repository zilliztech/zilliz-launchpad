## Context

The MVP (`mvp-phases-1-to-4`) deliberately took `pymilvus` as the single abstraction for both local Milvus Standalone and Zilliz Cloud. That was the right call for MVP — one code path, one test surface. It leaves control-plane operations manual and bulk ingestion slow.

`zilliz-cli` already exists and is the canonical tool for control-plane work. This change is about **additive integration** — using the CLI where it's strictly better, without forking the data-plane code paths or making the MVP harder to run.

## Goals / Non-Goals

**Goals:**

- Auto-discover existing Zilliz Cloud clusters during Phase 2 Configure when the CLI is available.
- Give Phase 4 a cheap pre-flight against Cloud targets so users get actionable errors instead of pymilvus connection timeouts.
- Route large ingestions (>100k rows) through `zilliz import create` for 10-30× throughput improvement.
- Land a Phase 6 scaffold that a follow-up change can flesh out to a working Deploy without new architectural decisions.
- Keep the local-Milvus path completely unchanged. The CLI is never required for local use.

**Non-Goals:**

- Replacing `pymilvus` for collection / index / search operations. These stay on `pymilvus` because (a) they work identically against local Milvus and Zilliz Cloud, and (b) `zilliz-cli` and `pymilvus` have different schema notations — translating between them is pure overhead for zero user-visible gain.
- Implementing Phase 6 Deploy. That is a separate, larger change; we only lay rails.
- Shipping a Python-side `zilliz-cli` equivalent. We always shell out to the real binary.
- Supporting multiple `zilliz-cli` versions at runtime. We pin a minimum version and require it.

## Decisions

### D1. The CLI is an **optional capability**, not a dependency

The launchpad must continue to run against local Milvus Standalone with zero external tools beyond Docker + `uv`. `zilliz_cli.py` exposes an `is_available()` probe that every call site checks first. If the CLI is missing:

- Phase 2: silently fall back to the current URI+token prompt path.
- Phase 4 pre-flight: skip (don't block Execute).
- Phase 4 bulk import: fall back to client-side upsert regardless of size, with a one-line notice.
- Phase 6 (future): refuse with a clear install hint — Deploy requires the CLI.

_Alternative considered:_ mandate the CLI. Rejected — increases first-run friction for the 60% of users who will only ever run local.

### D2. Subprocess wrapper with JSON-first parsing

`lib/zilliz_cli.py` exposes typed Python functions that invoke `zilliz <subcommand> --output json`, parse the response with `json.loads`, and surface errors through the existing `LaunchpadError` hierarchy. No Python bindings, no stdout scraping.

Rationale: simplest possible surface, CLI owns its own logic, we don't reimplement argument parsing or auth flows.

_Alternative considered:_ writing a Python SDK that talks to the Zilliz Cloud REST API directly. Rejected — duplicates work and drifts from CLI behavior.

### D3. Detection semantics

`is_available()` returns true only if **all** of the following hold:

1. `zilliz` is on `PATH` (`shutil.which`)
2. `zilliz version --output json` returns a version ≥ the pinned minimum
3. `zilliz auth whoami --output json` returns a logged-in principal

These are cached per-process (first call probes; subsequent calls reuse the result). An `invalidate()` helper exists so tests can reset the cache.

Detection failures at tiers 2 and 3 produce distinct errors (`ZillizCliStaleError`, `ZillizCliAuthError`) so the skill can give precise guidance ("run `zilliz upgrade`" vs. "run `zilliz auth login`").

### D4. Phase 2 — cluster auto-discovery UX

When `deployment_target ∈ {zilliz-serverless, zilliz-dedicated, zilliz-byoc}` and the CLI is available:

1. Call `zilliz cluster list --output json`.
2. If ≥ 1 cluster matches the chosen target tier, present the list to the agent; the agent asks the user to pick one, or to create a new one (future Phase 6).
3. Store `cluster_id`, the resolved URI, and the resolved token inside `configure.json`.

If the CLI is missing or the list is empty, fall back to the current prompt ("paste your URI and token").

The `configure.json` schema gains optional `cluster_id` and `resolved_from_cli: boolean` fields so `plan` and `execute` know whether to re-verify via CLI or trust env.

### D5. Phase 4 — pre-flight check

For Cloud targets, before opening any pymilvus connection:

```
zilliz cluster describe --cluster-id <id> --output json
```

Expected states:

- `RUNNING` → continue
- `PROVISIONING`, `MODIFYING` → wait with exponential backoff, max 60s, then continue or fail
- `PAUSED` → offer the user to resume via `zilliz cluster resume` (we only prompt; we do not auto-resume without consent)
- `DELETING`, `FAILED` → hard-fail with `ClusterNotReadyError`

If `cluster_id` isn't in `configure.json` (CLI wasn't used for discovery), skip the pre-flight — the user likely has a URI for a cluster we can't inspect programmatically.

### D6. Phase 4 — bulk import threshold

Plan gets a new field `bulk_import_threshold` (default `100_000`).

Execute chooses the ingestion path **after** counting the candidate rows from the input source:

- `rows <= threshold`: current `ingest_documents()` client-upsert path (known-good, idempotent via deterministic PKs).
- `rows > threshold` **and** `deployment_target` is a Cloud target **and** CLI available: upload JSONL + vectors to the cluster's object store (provisioned by the CLI) and submit `zilliz import create --files …`. Poll `zilliz import describe` until `DONE`, `FAILED`, or a 30-minute wall-clock cap.

If any precondition fails, we fall back to the client path and emit a stderr notice — the user gets a working collection, just slower.

### D7. Pre-computing embeddings for bulk import

Bulk import accepts vectors in the uploaded files. Before import, we run the existing embedder (`lib/embeddings.py`) over all chunks, write a JSONL with the `embedding` field populated, upload, then kick off the import job. This keeps embedding cost visible to the user and reuses the existing provider strategies.

_Alternative considered:_ server-side embedding via Zilliz BYOM-in-cluster. Rejected for this change — not universally available and requires extra provisioning.

### D8. Credential resolution fallback chain

`lib/credentials.py::resolve("ZILLIZ_TOKEN")` gains a new step in its resolution order:

1. Environment variable (unchanged)
2. **New**: `zilliz auth whoami --output json` (if the CLI is available and logged in, it returns an API token scoped to the current context)
3. `MissingCredentialError` (unchanged)

The CLI step is gated behind an opt-in: `resolve(key, allow_cli=True)`. Default stays `False` to avoid surprise subprocess calls during unit tests.

### D9. Phase 6 scaffold

`lib/phases/deploy.py` is created but intentionally minimal:

```python
def run_deploy(*, out_dir: Path, plan: dict) -> dict:
    raise NotImplementedError(
        "Phase 6 Deploy lands in a later change. "
        "Scaffold present so call sites compile."
    )
```

A `deploy` subcommand is **not** added to the CLI yet — avoids exposing a broken flow to users.

### D10. Errors & observability

Three new error types:

- `ZillizCliMissingError` — CLI not on PATH; emitted only when a call site explicitly requires it (Phase 6). Includes the install URL.
- `ZillizCliAuthError` — CLI present but not logged in. Includes the exact `zilliz auth login` command.
- `ClusterNotReadyError` — pre-flight found a non-RUNNING state; includes the observed state and the remediation command.

All three implement the existing `LaunchpadError.to_json()` envelope — the skill surface layer doesn't need new handling.

### D11. Testing strategy

- `tests/test_zilliz_cli.py` mocks `subprocess.run` and asserts command construction, JSON parsing, version gating, and error mapping.
- `tests/test_plan.py` gains a `bulk_import_threshold` default-value test.
- `tests/test_execute.py` (marked `e2e`) grows a branch for the CLI-present path, skipped by default.
- No changes to the local-Milvus E2E test.

## Risks / Trade-offs

| Risk | Mitigation |
| --- | --- |
| `zilliz-cli` version drift breaks our JSON parsing | Pin a minimum version; version check in `is_available()`; integration test matrix locked to the pinned version. |
| Users mistakenly think CLI is required | README Quick Start keeps local Milvus flow first and never mentions the CLI until the "Going to Cloud" section. |
| Subprocess latency inflates Phase 2 runtime | Cache detection; only call `cluster list` once per configure run; avoid repeated `whoami` probes. |
| Bulk import failure modes are new | Fall back to client-side upsert with a clear notice; don't leave the user in an "import pending forever" state — enforce the 30-minute cap. |
| Embedding cost during bulk import is hidden | Log embedding count + estimated provider-cost range before starting the import job; require explicit confirmation only when a plan-declared budget would be exceeded. |
| CLI auth token leaks into logs | Never log the output of `zilliz auth whoami`; mask tokens matching the `za_*` pattern in the CLI log prefix. |
| Phase 6 scaffold rots | Delete `lib/phases/deploy.py` entirely if the follow-up change doesn't land within 2 quarters — don't keep a `NotImplementedError` around forever. |

## Migration Plan

Not applicable — additive. Users on `mvp-phases-1-to-4` who don't install `zilliz-cli` see no behavior change. Users who do install it get auto-discovery, pre-flight, and bulk import for free.

Rollback is `git revert`; nothing persists on disk outside of `runs/<timestamp>/`.

## Open Questions

1. **Minimum `zilliz-cli` version to pin** — needs a check against the latest tagged release that ships `--output json` uniformly across subcommands we touch (`cluster list`, `cluster describe`, `import create`, `import describe`, `auth whoami`, `version`). Current placeholder: `>= 0.3.0`.
2. **Object-store upload surface** — `zilliz import create` accepts data from the cluster's own bucket. Does the CLI provide a direct upload helper, or do we still need `aws s3 cp`-style tooling? Affects whether `lib/zilliz_cli.py` grows an `upload()` function or just orchestrates.
3. **Bulk-import partitioning** — do we submit one job per 10M rows or one job total? Scales and failure-isolation argue for sharding; defer until we see real corpora.
4. **Token scoping** — `zilliz auth whoami` tokens may be org-scoped or project-scoped. Phase 4 needs project-scoped access for the target cluster; behavior when the active CLI context mismatches the chosen cluster's project is TBD.
5. **Windows / WSL path handling** for the file-upload step — assumed POSIX for MVP; explicit WSL testing deferred.
