## ADDED Requirements

### Requirement: Bundled datasets

The skill SHALL bundle at least two sample datasets under `skills/zilliz-launchpad/scripts/sample_data/`: a conversational one (`movies.jsonl`, ≤5k rows) and a retrieval-benchmark subset (`beir_scifact_mini.jsonl`, ≤1k rows with its qrels file `beir_scifact_qrels.tsv`). Each dataset MUST be loadable by `lib/samples.py` via a single function call.

#### Scenario: List available samples

- **WHEN** `lib.samples.list_datasets()` is called
- **THEN** the returned list includes `"movies"` and `"beir-scifact-mini"`

#### Scenario: Load movies dataset

- **WHEN** `lib.samples.load("movies")` is called
- **THEN** an iterable of dict records is returned
- **AND** each record has at least `id` and a text field

### Requirement: License compliance

Every bundled dataset SHALL have a `LICENSE.txt` or `PROVENANCE.md` adjacent to it in the same directory, naming the source and the license. Licenses MUST permit redistribution alongside an Apache-2.0 project.

#### Scenario: Provenance present for every dataset

- **WHEN** the contents of `scripts/sample_data/` are listed
- **THEN** each `.jsonl` file has a corresponding `*.PROVENANCE.md` or the directory has a single covering `PROVENANCE.md`

### Requirement: Zero-setup demo path

`zilliz_ops.py` SHALL accept `--sample <name>` as an alternative to a user-supplied input file for Phases 1 and 4. This MUST allow a first-time user to run the entire MVP flow without providing any data of their own.

#### Scenario: End-to-end with sample data

- **WHEN** a user runs `zilliz_ops.py collect --sample movies` followed by the later phases
- **THEN** all four phases complete against the bundled movies dataset
- **AND** the local UI returns non-empty results for a query drawn from the dataset
