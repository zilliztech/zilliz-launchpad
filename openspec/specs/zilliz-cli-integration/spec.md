# zilliz-cli-integration Specification

## Purpose
TBD - created by archiving change integrate-zilliz-cli. Update Purpose after archive.
## Requirements

### Requirement: CLI detection

The launchpad SHALL provide `lib.zilliz_cli.is_available()` that returns `True` only when all of the following hold: the `zilliz` binary is on `PATH`, its reported version is at or above the pinned minimum, and the active CLI session is authenticated. The result MUST be cached per-process and MUST be resettable via `lib.zilliz_cli.invalidate()` for tests.

#### Scenario: CLI absent

- **WHEN** `zilliz` is not on `PATH`
- **THEN** `is_available()` returns `False`
- **AND** no subprocess is launched after the `shutil.which` probe

#### Scenario: CLI present but unauthed

- **WHEN** `zilliz auth whoami --output json` exits non-zero
- **THEN** `is_available()` returns `False`
- **AND** a distinct `ZillizCliAuthError` is raised only when a call site explicitly requires auth

#### Scenario: CLI present, authed, version too old

- **WHEN** `zilliz version --output json` reports a version below the pinned minimum
- **THEN** `is_available()` returns `False`
- **AND** raising the error surface identifies the installed version and the required version

### Requirement: Subprocess wrapper

The wrapper SHALL invoke `zilliz` subcommands with `--output json`, parse stdout as JSON, and map non-zero exit codes to `LaunchpadError` subclasses. It MUST NOT log the raw output of `zilliz auth whoami`.

#### Scenario: Successful JSON roundtrip

- **WHEN** `cluster_list()` is called and the CLI returns a JSON array
- **THEN** the wrapper returns a Python list of cluster dicts without string manipulation

#### Scenario: Non-zero exit is surfaced

- **WHEN** the CLI exits with an error code
- **THEN** the wrapper raises a `LaunchpadError` subclass carrying the CLI's stderr message

#### Scenario: Token is not logged

- **WHEN** `auth_whoami()` succeeds
- **THEN** no log entry (stdout, stderr, or structured log) contains the returned token value

### Requirement: Credential fallback to CLI session

`lib.credentials.resolve(key, allow_cli=True)` SHALL, for `key == "ZILLIZ_TOKEN"` and when `is_available()` returns `True`, use the CLI's authenticated session token as a fallback after the environment variable check. The fallback MUST be off by default (`allow_cli=False`) so unit tests do not spawn subprocesses unintentionally.

#### Scenario: Env takes precedence

- **WHEN** `ZILLIZ_TOKEN` is set in the environment and `allow_cli=True`
- **THEN** the env value is returned without invoking the CLI

#### Scenario: CLI fills the gap

- **WHEN** `ZILLIZ_TOKEN` is unset, `allow_cli=True`, and `is_available()` is `True`
- **THEN** the CLI-resolved token is returned

#### Scenario: Default opt-out

- **WHEN** `allow_cli` is not specified
- **THEN** the CLI is not invoked and missing env raises `MissingCredentialError`

### Requirement: Phase 2 cluster auto-discovery

When the user's `deployment_target` is a Zilliz Cloud target and `is_available()` is `True`, the Configure phase SHALL call `zilliz cluster list` and surface the matching clusters to the agent so the user can pick one. The choice MUST be persisted in `configure.json` as `cluster_id` and `resolved_from_cli: true`. When the CLI is absent or returns no matching clusters, Configure MUST fall back to the existing prompt flow without error.

#### Scenario: Discovery succeeds

- **WHEN** CLI is available and returns ≥ 1 matching cluster
- **THEN** `configure.json` contains `cluster_id` and `resolved_from_cli: true`

#### Scenario: CLI absent

- **WHEN** `zilliz` is not on `PATH`
- **THEN** the existing prompt flow runs
- **AND** `configure.json` does not contain `cluster_id`

#### Scenario: No matching clusters

- **WHEN** CLI returns an empty list for the selected target tier
- **THEN** Configure falls back to the prompt flow with a log notice

### Requirement: Phase 4 pre-flight

When `configure.json` contains a `cluster_id`, Phase 4 Execute SHALL call `zilliz cluster describe --cluster-id <id>` before opening any pymilvus connection. If the cluster state is `PROVISIONING` or `MODIFYING`, Execute MUST wait with exponential backoff up to 60 seconds. If the state is `PAUSED`, `DELETING`, or `FAILED`, Execute MUST raise `ClusterNotReadyError` with the observed state and a remediation hint.

#### Scenario: Cluster running

- **WHEN** describe returns `state=RUNNING`
- **THEN** Execute proceeds to create-collection

#### Scenario: Cluster paused

- **WHEN** describe returns `state=PAUSED`
- **THEN** `ClusterNotReadyError` is raised
- **AND** the error payload names `zilliz cluster resume --cluster-id <id>` as the remediation

#### Scenario: No cluster_id present

- **WHEN** `configure.json` lacks `cluster_id` (user pasted URI manually)
- **THEN** the pre-flight is skipped

### Requirement: Bulk-import routing

Phase 3 Plan SHALL emit a `bulk_import_threshold` integer (default `100000`). Phase 4 Execute SHALL, when the row count exceeds the threshold **and** the target is a Zilliz Cloud cluster **and** `is_available()` is `True`, perform ingestion via `zilliz import create` instead of client-side upsert. All other combinations MUST use the existing client-side path.

#### Scenario: Small dataset uses client path

- **WHEN** `rows <= bulk_import_threshold`
- **THEN** `ingest_documents` (client-side) is called

#### Scenario: Large cloud dataset uses import job

- **WHEN** `rows > bulk_import_threshold`, target is a Cloud cluster, and CLI is available
- **THEN** a `zilliz import create` job is submitted
- **AND** Execute polls `zilliz import describe` until `DONE`, `FAILED`, or a 30-minute wall-clock cap elapses

#### Scenario: Threshold met but CLI absent

- **WHEN** the threshold is exceeded but `is_available()` is `False`
- **THEN** Execute falls back to the client-side path
- **AND** a notice is printed explaining the fallback and how to enable the faster path

#### Scenario: Import job failure

- **WHEN** `zilliz import describe` reports `state=FAILED`
- **THEN** Execute raises a `LaunchpadError` subclass carrying the CLI's failure reason

### Requirement: Phase 6 scaffold

The launchpad SHALL ship a `lib/phases/deploy.py` module that raises `NotImplementedError` when invoked and a `ZillizCliMissingError` class reusable by a future Deploy implementation. No `deploy` CLI subcommand MAY be exposed until the Deploy phase is fully implemented in a later change.

#### Scenario: Deploy module present

- **WHEN** `scripts/lib/phases/deploy.py` is imported
- **THEN** the import succeeds
- **AND** `run_deploy` exists and raises `NotImplementedError` with a pointer to the future change

#### Scenario: No deploy subcommand

- **WHEN** `python zilliz_ops.py --help` is run
- **THEN** `deploy` is not listed in the subcommand list

### Requirement: Error taxonomy

The integration SHALL introduce three error classes derived from `LaunchpadError`: `ZillizCliMissingError` (binary not on `PATH`), `ZillizCliAuthError` (present but unauthed or expired), and `ClusterNotReadyError` (pre-flight found a non-`RUNNING` state). Each MUST serialize through `LaunchpadError.to_json()` without additional code paths.

#### Scenario: Missing-binary payload

- **WHEN** `ZillizCliMissingError` is serialized
- **THEN** the payload contains an `install_url` field pointing at the public install instructions

#### Scenario: Auth-error payload

- **WHEN** `ZillizCliAuthError` is serialized
- **THEN** the payload contains `remediation: "zilliz auth login"`

#### Scenario: Cluster-not-ready payload

- **WHEN** `ClusterNotReadyError` is serialized
- **THEN** the payload contains `state`, `cluster_id`, and a `remediation` command string
