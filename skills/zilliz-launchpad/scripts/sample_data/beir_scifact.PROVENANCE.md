# beir_scifact_mini — Provenance

**Source**: synthetic, inspired by the shape of the BEIR SciFact benchmark
(https://github.com/beir-cellar/beir) but with originally-composed entries
to avoid license mixing. No BEIR data is redistributed here.

**License**: Apache-2.0 (same as the repository).

**Files**:
- `beir_scifact_mini.jsonl` — corpus (15 docs, scientific-claim style)
- `beir_scifact_queries.jsonl` — queries (8 test queries)
- `beir_scifact_qrels.tsv` — query→doc relevance judgments (TREC qrels style)

**Schema**:
- corpus: `{id, title, body}`
- queries: `{id, text}`
- qrels: `query-id \t corpus-id \t score` (score=1 = relevant)

**Intent**: demonstrate the evaluate-ready sample-data pattern. For real
benchmarking, replace with the actual BEIR SciFact split and mind its
CC-BY-SA-4.0 license.
