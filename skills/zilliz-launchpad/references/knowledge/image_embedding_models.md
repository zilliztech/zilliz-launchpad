# Image embedding models

Used by the `image-search` use case (text-to-image MVP). Selected in Phase 3 (Plan) and consumed by Phase 4 (Execute) and Phase 5 (Evaluate).

| Provider | Model | Dim | API key | Notes |
| --- | --- | --- | --- | --- |
| `clip-local` | `ViT-B-32` | 512 | none | Default. `open-clip-torch` weights (~150 MB), runs on CPU / MPS / CUDA. Permissive license (MIT). One-time download cached under `~/.cache/clip` (open-clip default). |
| `voyage` | `voyage-multimodal-3` | 1024 | `VOYAGE_API_KEY` | Higher quality. API-only. Honors `Retry-After` on 429. |

## How to pick

1. **Default**: `clip-local` — no key, no network at query time, good zero-shot quality on natural photos.
2. **Quality matters more than cost / latency**: `voyage-multimodal-3`.
3. **No-internet / air-gapped**: `clip-local` with the model already prefetched (`execute --prefetch-models`).

## Trade-offs to know

- **First-run delay**: `clip-local` downloads ~150 MB of weights on first invocation. The CLI prints a one-line notice before the download starts.
- **Device hint**: Phase 3 detects MPS / CUDA / CPU once and writes `embedding.device_hint` into `plan.json`. Phase 4 honors it and falls back to CPU with a warning if the hint device errors at runtime.
- **Dim affects schema**: switching from `clip-local` (512) to `voyage-multimodal-3` (1024) requires a new collection — same constraint as text embedding swaps.
- **Sparse field disabled**: image collections have no useful sparse signal; Phase 3 forces `hybrid: false` and skips the sparse field.

## Install

`clip-local` lives in the `[multimodal]` extra:

```bash
uv pip install -e '.[multimodal]'
```

Without the extra, image code paths exit with a `missing_dependency` error envelope and an `install_hint` pointing at the same command.

`voyage-multimodal-3` only needs `VOYAGE_API_KEY` — no extra install (the base `voyageai` client is already in core deps).

## Pricing snapshot (April 2026)

- `clip-local`: free (compute only). Roughly 100 images / sec on M1 CPU, 1000+ / sec on MPS.
- `voyage-multimodal-3`: see [Voyage pricing](https://docs.voyageai.com/docs/pricing). Update this row when populating the `cost/query` column in `eval_report.md` (issue #10).

## Future providers (not in MVP)

- Cohere `embed-v4` (multimodal) — pending separate evaluation
- BYOM image endpoints — pending Zilliz BYOM publishing one
- Larger CLIP variants (`ViT-L/14`, `ViT-H/14`) — exposed once Phase 3 grows a `--clip-model` knob
