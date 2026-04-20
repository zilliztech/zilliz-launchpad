# ingestion-pipeline Specification

## Purpose
TBD - created by archiving change mvp-phases-1-to-4. Update Purpose after archive.
## Requirements
### Requirement: Document chunking

The pipeline SHALL split raw text into chunks before embedding. Default strategy is recursive character splitting targeting ~512 tokens with 64-token overlap. The chunking strategy MUST be configurable per run via the plan artifact.

#### Scenario: Default chunking on plain text

- **WHEN** a 4000-token document is ingested with defaults
- **THEN** roughly 8 chunks are produced
- **AND** each non-final chunk ends with 64 tokens of the next chunk (overlap)

#### Scenario: Chunking config from plan

- **WHEN** `plan.json` specifies `chunking.size=256, overlap=0`
- **THEN** chunks of that size are produced with no overlap

### Requirement: API-based embedding invocation

The pipeline SHALL embed chunks by calling an HTTP-API provider. Supported providers for MVP: OpenAI, Voyage, Cohere, and a generic `zilliz-byom` endpoint. Provider selection MUST be driven by the plan. The pipeline MUST NOT load local model weights and MUST NOT depend on `torch` or `sentence-transformers`.

#### Scenario: Provider dispatch

- **WHEN** the plan specifies `embedding.provider=openai` and `model=text-embedding-3-small`
- **AND** Phase 4 runs ingestion
- **THEN** HTTPS requests are sent to the OpenAI embeddings endpoint using that model
- **AND** no other provider SDK is imported

#### Scenario: Missing API key surfaces clearly

- **WHEN** the selected provider's API key env var is unset
- **THEN** the pipeline exits with a `MissingCredentialError` naming the expected variable

### Requirement: Batched insert with retry

The pipeline SHALL batch inserts (default 64 per batch) and SHALL retry transient network errors with exponential backoff (max 5 attempts). Non-retryable errors MUST fail the run.

#### Scenario: Transient error is retried

- **WHEN** an insert batch receives a 503 response
- **THEN** the client retries with backoff up to 5 attempts
- **AND** if any attempt succeeds, ingestion continues

#### Scenario: Schema-validation error aborts

- **WHEN** the Milvus server rejects a batch with an invalid-schema error
- **THEN** the pipeline stops immediately without retry
- **AND** the error surfaces to the CLI with the offending field

### Requirement: Idempotent re-ingest

Primary keys SHALL be deterministic (SHA-256 hash of the source document id + chunk index). Re-running ingest on the same source MUST upsert rather than duplicate rows.

#### Scenario: Re-running ingest keeps row count stable

- **WHEN** `ingest movies.jsonl` runs twice on the same file
- **THEN** the collection row count after the second run equals the row count after the first run

