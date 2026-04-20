# Dense embedding models

API-only providers supported in the MVP (no local / torch models).

| Provider | Recommended model          | Dim  | Notes |
| ---      | ---                        | ---  | --- |
| openai   | `text-embedding-3-small`   | 1536 | Default. Balanced cost/quality. |
| openai   | `text-embedding-3-large`   | 3072 | Higher quality, 5× cost. |
| voyage   | `voyage-3`                 | 1024 | Strong on retrieval; reasonably priced. |
| voyage   | `voyage-3-large`           | 2048 | Top-of-line quality. |
| cohere   | `embed-english-v3.0`       | 1024 | Solid English; `embed-multilingual-v3.0` for non-English. |
| zilliz-byom | (user-chosen)           | (user-chosen) | For self-hosted / private endpoints. |

## How to pick

1. **Default**: `openai/text-embedding-3-small` — cheap enough to prototype, good enough for most.
2. **English-heavy retrieval**: `voyage-3` usually edges OpenAI on BEIR-style benchmarks.
3. **Multilingual**: `cohere embed-multilingual-v3.0` or `voyage-3` (multilingual-capable).
4. **Cost-sensitive large corpora**: OpenAI small.
5. **Data cannot leave VPC**: BYOM with a self-hosted model behind a Zilliz BYOM endpoint.

## Dimensions matter downstream

The `dim` field is part of the collection schema. Changing embedding model → requires a new collection (or Matryoshka-style truncation, which the MVP does not support).
