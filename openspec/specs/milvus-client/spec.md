# milvus-client Specification

## Purpose
TBD - created by archiving change mvp-phases-1-to-4. Update Purpose after archive.
## Requirements
### Requirement: URI-based connection

The library SHALL expose a single `MilvusClient` factory that accepts a `uri` and optional `token`, and SHALL detect the target backend (local Milvus Standalone, Zilliz Cloud, or another OSS Milvus deployment) from the URI. Business code MUST NOT branch on backend type.

#### Scenario: Local Standalone URI connects without token

- **WHEN** `MilvusClient(uri="http://localhost:19530")` is constructed
- **THEN** a live connection is opened to the local Standalone server
- **AND** no token is required

#### Scenario: Zilliz Cloud URI requires token

- **WHEN** `MilvusClient(uri="https://in03-xxx.api.gcp-us-west1.zillizcloud.com")` is constructed without a token
- **THEN** the client raises `MissingCredentialError` naming `ZILLIZ_TOKEN`

#### Scenario: Token resolved from environment

- **WHEN** `ZILLIZ_TOKEN` is set
- **AND** a Zilliz Cloud URI is passed with no explicit token argument
- **THEN** the client picks up the env value and connects

### Requirement: Collection lifecycle

The library SHALL provide `create_collection`, `load_collection`, `drop_collection`, and `collection_exists` operations. `create_collection` MUST be idempotent when called with an identical schema and MUST fail loudly when an existing collection has a different schema.

#### Scenario: Re-create with same schema is a no-op

- **WHEN** `create_collection("docs", schema=S)` is called twice with the same `S`
- **THEN** the second call returns successfully without recreating
- **AND** the collection count is 1

#### Scenario: Schema mismatch raises an error

- **WHEN** a collection `docs` exists with schema `S1`
- **AND** `create_collection("docs", schema=S2)` is called with `S2 != S1`
- **THEN** the operation raises `SchemaConflictError` identifying the mismatched field(s)

### Requirement: Index management

The library SHALL provide `create_index`, `describe_index`, and `drop_index`. `create_index` MUST check current index parameters; if they match the request, it SHALL be a no-op; if they differ, it SHALL drop and rebuild. Supported index types for MVP: `FLAT`, `IVF_FLAT`, `HNSW`, `DISKANN` (Zilliz Cloud / Standalone only).

#### Scenario: Matching index params skip rebuild

- **WHEN** an HNSW index with `M=16, efConstruction=200` exists on field `embedding`
- **AND** `create_index("docs", "embedding", type="HNSW", params={"M":16,"efConstruction":200})` is called
- **THEN** no rebuild happens and the call returns quickly

#### Scenario: Different params trigger rebuild

- **WHEN** the existing index has `M=16` and the request has `M=32`
- **THEN** the old index is dropped and a new one is built

