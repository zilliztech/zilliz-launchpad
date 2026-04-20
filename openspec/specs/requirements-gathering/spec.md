# requirements-gathering Specification

## Purpose
TBD - created by archiving change mvp-phases-1-to-4. Update Purpose after archive.
## Requirements
### Requirement: Sample document analysis (Phase 1 Collect)

Phase 1 SHALL accept a user-provided sample file (JSONL, CSV, or plain text) and produce a structured data-shape report: detected fields, their types, approximate text lengths, and a suggested primary-key field. The report MUST be written to the current run directory as `collect.json`.

#### Scenario: JSONL sample with mixed fields

- **WHEN** the user supplies `sample.jsonl` with fields `{id, title, body, year}`
- **AND** Phase 1 runs
- **THEN** `collect.json` lists each field with inferred type
- **AND** `id` is suggested as the primary key
- **AND** `body` is flagged as the primary text field (longest average length)

#### Scenario: Plain-text file

- **WHEN** the user supplies `docs.txt`
- **THEN** `collect.json` reports a single synthetic field `text` with type `string` and suggests generating surrogate ids at ingest time

### Requirement: Requirement capture (Phase 2 Configure)

Phase 2 SHALL hold a structured dialogue with the user to capture: primary use case (RAG / semantic search / recommendations), expected query patterns, hard constraints (latency target, dataset size, deployment target), and optional preferences (preferred embedding provider, budget sensitivity). Output MUST be written as `configure.json`.

#### Scenario: Required fields collected before exit

- **WHEN** Phase 2 dialogue completes
- **THEN** `configure.json` contains `use_case`, `query_patterns`, `dataset_size`, and `deployment_target` (all non-null)

#### Scenario: Deployment target constrained to supported values

- **WHEN** the user indicates deployment target
- **THEN** `deployment_target` is one of: `local-standalone`, `zilliz-serverless`, `zilliz-dedicated`, `zilliz-byoc`

### Requirement: Requirement profile contract

The combined output of Phases 1 and 2 SHALL conform to a documented JSON schema (`skills/zilliz-launchpad/references/requirement-profile.schema.json`). Phase 3 (Plan) MUST treat this profile as its sole input for deterministic decisioning.

#### Scenario: Profile validates against schema

- **WHEN** `collect.json` and `configure.json` exist after Phases 1–2
- **THEN** their merged content validates against `requirement-profile.schema.json`

#### Scenario: Invalid profile blocks Phase 3

- **WHEN** the merged profile fails schema validation
- **THEN** `zilliz_ops.py plan` exits non-zero
- **AND** the error names the offending JSON Pointer path

