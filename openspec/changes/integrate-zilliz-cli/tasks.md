## 1. CLI wrapper foundation

- [ ] 1.1 Create `scripts/lib/zilliz_cli.py` with `is_available()`, `invalidate()`, and per-process caching; version probe via `zilliz version --output json`; auth probe via `zilliz auth whoami --output json`
- [ ] 1.2 Add typed wrapper functions: `cluster_list`, `cluster_describe`, `cluster_create_stub` (Phase 6 placeholder), `import_create`, `import_describe`, `auth_whoami`
- [ ] 1.3 Pin the minimum `zilliz-cli` version constant (default `>=0.3.0` pending verification)
- [ ] 1.4 Ensure no subprocess output containing tokens is logged (mask `za_*` patterns in any log prefixes)

## 2. Error taxonomy

- [ ] 2.1 Add `ZillizCliMissingError` in `lib/errors.py` (with `install_url` payload)
- [ ] 2.2 Add `ZillizCliAuthError` (with `remediation` payload naming `zilliz auth login`)
- [ ] 2.3 Add `ClusterNotReadyError` (with `state`, `cluster_id`, `remediation` payload)

## 3. Credential fallback

- [ ] 3.1 Extend `lib/credentials.py::resolve` with an `allow_cli: bool = False` parameter
- [ ] 3.2 When `allow_cli=True` and env is unset, call `zilliz_cli.auth_whoami()` as a fallback source
- [ ] 3.3 Update `lib/client.py`'s token resolution to pass `allow_cli=True` for Cloud URIs

## 4. Phase 2 — cluster auto-discovery

- [ ] 4.1 In `lib/phases/configure.py`, detect Cloud `deployment_target` and gated on `zilliz_cli.is_available()`, call `cluster_list` and surface results
- [ ] 4.2 Persist `cluster_id` and `resolved_from_cli: true` into `configure.json`
- [ ] 4.3 Update `requirement-profile.schema.json` to accept the new optional fields
- [ ] 4.4 Fall back silently to the current prompt when CLI absent, returns empty, or errors

## 5. Phase 4 — pre-flight

- [ ] 5.1 In `lib/phases/execute.py`, before opening pymilvus, run `cluster_describe` when `cluster_id` is present
- [ ] 5.2 Implement exponential-backoff wait (max 60s) for `PROVISIONING` / `MODIFYING` states
- [ ] 5.3 Raise `ClusterNotReadyError` for `PAUSED`, `DELETING`, `FAILED`; include remediation command

## 6. Phase 4 — bulk-import routing

- [ ] 6.1 Add `bulk_import_threshold` (default `100_000`) to `lib/phases/plan.py` output and to `plan.json` shape
- [ ] 6.2 In `lib/phases/execute.py`, count documents before ingestion and branch on threshold × target × CLI availability
- [ ] 6.3 Implement the bulk path: pre-compute embeddings (reuse `lib/embeddings.py`), write JSONL with embeddings, upload via the CLI's supported mechanism, call `import_create`
- [ ] 6.4 Poll `import_describe` with backoff until `DONE`, `FAILED`, or 30-minute cap; map failures to `LaunchpadError`
- [ ] 6.5 Fall back to the client-side path with a notice when any precondition fails

## 7. Phase 6 scaffold

- [ ] 7.1 Create `lib/phases/deploy.py` with a `run_deploy(*, out_dir, plan) -> dict` that raises `NotImplementedError` pointing at a future change
- [ ] 7.2 Do **not** add a `deploy` subcommand to `zilliz_ops.py` yet
- [ ] 7.3 Import the module from a call site that ensures `python -c "from lib.phases import deploy"` works (smoke test)

## 8. Skill surface updates

- [ ] 8.1 Update `SKILL.md` with: optional CLI dependency block; phase → CLI-command table; new error codes (`cluster_not_ready`, `zilliz_cli_missing`, `zilliz_cli_auth`)
- [ ] 8.2 Keep `SKILL.md` under 500 lines (enforce via the existing CI check once added)

## 9. Reference & docs

- [ ] 9.1 Rewrite `references/deploy-serverless.md` around "use `zilliz cluster list` to pick" and "Deploy phase will automate creation in a later change"
- [ ] 9.2 Add CLI-present and CLI-absent example blocks to `references/cli-reference.md`
- [ ] 9.3 Update `README.md` Quick Start — add a "Going to Cloud" subsection that introduces the CLI after the local flow
- [ ] 9.4 Update `docs/TROUBLESHOOTING.md` with the three new error codes and their fixes

## 10. Tests

- [ ] 10.1 `tests/test_zilliz_cli.py`: mock `subprocess.run`, assert command construction, JSON parsing, version gating, and error mapping for every wrapper function
- [ ] 10.2 `tests/test_configure.py` (new): discovery-present and discovery-absent paths produce the expected `configure.json` shape
- [ ] 10.3 `tests/test_plan.py`: add a case asserting `bulk_import_threshold` default and that the planner emits it
- [ ] 10.4 `tests/test_execute.py`: add an `e2e`-marked branch exercising the CLI-present bulk-import path; skipped by default
- [ ] 10.5 Confirm `pytest -m "not cloud and not e2e"` stays green

## 11. CI

- [ ] 11.1 Install the pinned `zilliz-cli` version in the CI lane (cache binary)
- [ ] 11.2 Add a CI step that asserts `pytest -m "not cloud and not e2e"` passes with and without the CLI on `PATH` (two lanes)
- [ ] 11.3 Add a lint check that `SKILL.md` stays ≤ 500 lines

## 12. Close-out

- [ ] 12.1 Note the `mvp-phases-1-to-4` archive prerequisite at the top of `tasks.md` — this change cannot be applied cleanly until the MVP specs live in `openspec/specs/`
- [ ] 12.2 Archive this change after merge (`openspec archive integrate-zilliz-cli`)
