## ADDED Requirements

### Requirement: Plan generation (Phase 3)

Phase 3 SHALL read the merged requirement profile and produce a concrete plan covering: collection schema (fields, types, primary key, vector dim), embedding provider + model, sparse-vector enablement flag, index type + parameters (including quantization if applicable), optional reranker, chunking config, and target backend URI. The plan SHALL be deterministic given the same profile.

#### Scenario: Small dataset picks HNSW without quantization

- **WHEN** the profile reports `dataset_size <= 100000` and `deployment_target=local-standalone`
- **THEN** `plan.json` specifies `index.type="HNSW"` and `index.quantization=null`

#### Scenario: Large dataset on Zilliz Cloud picks DiskANN

- **WHEN** the profile reports `dataset_size >= 10000000` and `deployment_target` is a Zilliz Cloud target
- **THEN** `plan.json` specifies `index.type="DISKANN"`

#### Scenario: Determinism

- **WHEN** Phase 3 runs twice against the same profile
- **THEN** both `plan.json` outputs are byte-identical (ignoring timestamp fields)

### Requirement: Plan artifacts

Each plan invocation SHALL write `plan.json` (machine-readable) and `plan.md` (human-readable) into `scripts/runs/<timestamp>/`. `plan.md` MUST present the chosen decisions as a checklist with a short rationale per choice.

#### Scenario: Both formats present

- **WHEN** Phase 3 completes
- **THEN** the run directory contains both `plan.json` and `plan.md`

#### Scenario: Rationale accompanies each choice

- **WHEN** the plan selects `HNSW` over `IVF_FLAT`
- **THEN** `plan.md` contains a one-line rationale referencing the relevant requirement (e.g., dataset size, latency target)

### Requirement: Plan execution (Phase 4)

Phase 4 SHALL apply the plan to the target backend: create collection, build indices, ingest sample data (if provided), and start the local UI sidecar. Phase 4 MUST be idempotent — rerunning it on the same plan MUST converge without duplicating data or rebuilding indices that already match.

#### Scenario: Fresh execute

- **WHEN** Phase 4 runs against an empty Milvus Standalone instance with a valid plan
- **THEN** the collection exists with the planned schema
- **AND** the planned index exists on the embedding field
- **AND** the local UI sidecar listens on the configured port

#### Scenario: Re-run is a no-op

- **WHEN** Phase 4 runs twice in a row with the same plan
- **THEN** the second run completes without recreating the collection or rebuilding the index
- **AND** the row count is unchanged

### Requirement: Post-execute smoke check

After execution, Phase 4 SHALL run a canned query against the collection and report the top-1 hit. Failure of this smoke check MUST cause the phase to exit non-zero.

#### Scenario: Smoke check succeeds

- **WHEN** Phase 4 finishes ingest and indexing
- **THEN** it runs a query derived from the sample's first document
- **AND** emits a console line like `✓ Top-1: <id> score=<score>`

#### Scenario: Smoke check fails (empty result)

- **WHEN** the smoke query returns zero hits
- **THEN** Phase 4 exits non-zero with a diagnostic hint (schema / index / data missing)
