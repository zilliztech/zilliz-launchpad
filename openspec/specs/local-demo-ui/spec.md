# local-demo-ui Specification

## Purpose
TBD - created by archiving change mvp-phases-1-to-4. Update Purpose after archive.
## Requirements
### Requirement: Next.js app structure

The UI SHALL be a standard Next.js 14+ TypeScript application under `skills/zilliz-launchpad/scripts/ui/`, scaffolded from `create-next-app --ts`. It MUST run via `npm run dev` and expose a search page at the root route `/`.

#### Scenario: Dev server starts cleanly

- **WHEN** `cd scripts/ui && npm install && npm run dev` is run
- **THEN** a Next.js dev server listens on the configured port (default 3000)
- **AND** visiting `http://localhost:3000/` renders the search page without console errors

### Requirement: Search sidecar API

A FastAPI sidecar SHALL be started by `zilliz_ops.py execute` on a configurable port (default 8000). The sidecar MUST expose `POST /search` accepting `{query: string, top_k?: number, mode?: "dense"|"sparse"|"hybrid"}` and returning `{hits: [{id, score, fields}]}`.

#### Scenario: Sidecar serves a hybrid query

- **WHEN** the UI POSTs `{"query":"action movies","mode":"hybrid","top_k":5}` to `/search`
- **THEN** the response status is 200
- **AND** the body contains an array of 5 hits with `id`, `score`, and selected fields

#### Scenario: Sidecar handles invalid mode

- **WHEN** the request specifies an unsupported `mode`
- **THEN** the response status is 400
- **AND** the body names the allowed values

### Requirement: Results presentation

The UI SHALL render each hit as a card showing the primary text field, score, and any other scalar fields returned. It SHALL offer a mode selector (Dense / Sparse / Hybrid) and a top-k input. It SHALL NOT require user-supplied API keys in the browser — all credentials live in the sidecar process.

#### Scenario: Mode selector switches queries

- **WHEN** the user changes the mode from Dense to Hybrid
- **AND** submits the same query
- **THEN** the sidecar is called with `mode: "hybrid"`
- **AND** the rendered results reflect the new response

#### Scenario: No credentials in browser bundle

- **WHEN** the built JS bundle is grepped for env var names like `OPENAI_API_KEY`, `ZILLIZ_TOKEN`, `COHERE_API_KEY`
- **THEN** no match is found

