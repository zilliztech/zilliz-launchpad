> **Prerequisite**: this change depends on `mvp-phases-1-to-4` being archived first. The integration touches specs that must already live in `openspec/specs/` before this change can be cleanly archived.

## 1. CLI wrapper foundation

- [x] 1.1 Create `scripts/lib/zilliz_cli.py` with `is_available()`, `invalidate()`, and per-process caching; version probe via `zilliz version --output json`; auth probe via `zilliz auth whoami --output json`
- [x] 1.2 Add typed wrapper functions: `cluster_list`, `cluster_describe`, `cluster_create_stub` (Phase 6 placeholder), `import_create`, `import_describe`, `auth_whoami`
- [x] 1.3 Pin the minimum `zilliz-cli` version constant (default `>=0.3.0` pending verification)
- [x] 1.4 Ensure no subprocess output containing tokens is logged (mask `za_*` patterns in any log prefixes)

## 2. Error taxonomy

- [x] 2.1 Add `ZillizCliMissingError` in `lib/errors.py` (with `install_url` payload)
- [x] 2.2 Add `ZillizCliAuthError` (with `remediation` payload naming `zilliz auth login`)
- [x] 2.3 Add `ClusterNotReadyError` (with `state`, `cluster_id`, `remediation` payload)

## 3. Credential fallback

- [x] 3.1 Extend `lib/credentials.py::resolve` with an `allow_cli: bool = False` parameter
- [x] 3.2 When `allow_cli=True` and env is unset, call `zilliz_cli.auth_whoami()` as a fallback source
- [x] 3.3 Update `lib/client.py`'s token resolution to pass `allow_cli=True` for Cloud URIs

## 4. Phase 2 — cluster auto-discovery

- [x] 4.1 In `lib/phases/configure.py`, detect Cloud `deployment_target` and gated on `zilliz_cli.is_available()`, call `cluster_list` and surface results
- [x] 4.2 Persist `cluster_id` and `resolved_from_cli: true` into `configure.json`
- [x] 4.3 Update `requirement-profile.schema.json` to accept the new optional fields
- [x] 4.4 Fall back silently to the current prompt when CLI absent, returns empty, or errors

## 5. Phase 4 — pre-flight

- [x] 5.1 In `lib/phases/execute.py`, before opening pymilvus, run `cluster_describe` when `cluster_id` is present
- [x] 5.2 Implement exponential-backoff wait (max 60s) for `PROVISIONING` / `MODIFYING` states
- [x] 5.3 Raise `ClusterNotReadyError` for `PAUSED`, `DELETING`, `FAILED`; include remediation command

## 6. Phase 4 — bulk-import routing

- [x] 6.1 Add `bulk_import_threshold` (default `100_000`) to `lib/phases/plan.py` output and to `plan.json` shape
- [x] 6.2 In `lib/phases/execute.py`, count documents before ingestion and branch on threshold × target × CLI availability
- [x] 6.3 Implement the bulk path: pre-compute embeddings (reuse `lib/embeddings.py`), write JSONL with embeddings, upload via the CLI's supported mechanism, call `import_create`
- [x] 6.4 Poll `import_describe` with backoff until `DONE`, `FAILED`, or 30-minute cap; map failures to `LaunchpadError`
- [x] 6.5 Fall back to the client-side path with a notice when any precondition fails

## 7. Phase 6 scaffold

- [x] 7.1 Create `lib/phases/deploy.py` with a `run_deploy(*, out_dir, plan) -> dict` that raises `NotImplementedError` pointing at a future change
- [x] 7.2 Do **not** add a `deploy` subcommand to `zilliz_ops.py` yet
- [x] 7.3 Import the module from a call site that ensures `python -c "from lib.phases import deploy"` works (smoke test)

## 8. Skill surface updates

- [x] 8.1 Update `SKILL.md` with: optional CLI dependency block; phase → CLI-command table; new error codes (`cluster_not_ready`, `zilliz_cli_missing`, `zilliz_cli_auth`)
- [x] 8.2 Keep `SKILL.md` under 500 lines (enforce via the existing CI check once added)

## 9. Reference & docs

- [x] 9.1 Rewrite `references/deploy-serverless.md` around "use `zilliz cluster list` to pick" and "Deploy phase will automate creation in a later change"
- [x] 9.2 Add CLI-present and CLI-absent example blocks to `references/cli-reference.md`
- [x] 9.3 Update `README.md` Quick Start — add a "Going to Cloud" subsection that introduces the CLI after the local flow
- [x] 9.4 Update `docs/TROUBLESHOOTING.md` with the three new error codes and their fixes

## 10. Tests

- [x] 10.1 `tests/test_zilliz_cli.py`: mock `subprocess.run`, assert command construction, JSON parsing, version gating, and error mapping for every wrapper function
- [x] 10.2 `tests/test_configure.py` (new): discovery-present and discovery-absent paths produce the expected `configure.json` shape
- [x] 10.3 `tests/test_plan.py`: add a case asserting `bulk_import_threshold` default and that the planner emits it
- [x] 10.4 `tests/test_execute.py`: add an `e2e`-marked branch exercising the CLI-present bulk-import path; skipped by default
- [x] 10.5 Confirm `pytest -m "not cloud and not e2e"` stays green

## 11. CI

- [x] 11.1 Install the pinned `zilliz-cli` version in the CI lane (cache binary)
- [x] 11.2 Add a CI step that asserts `pytest -m "not cloud and not e2e"` passes with and without the CLI on `PATH` (two lanes)
- [x] 11.3 Add a lint check that `SKILL.md` stays ≤ 500 lines

## 12. Close-out

- [x] 12.1 Note the `mvp-phases-1-to-4` archive prerequisite at the top of `tasks.md` — this change cannot be applied cleanly until the MVP specs live in `openspec/specs/`
- [ ] 12.2 Archive this change after merge (`openspec archive integrate-zilliz-cli`)
