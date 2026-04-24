# Manual smoke: video-search end-to-end

Exercises the full video slice against the synthetic fixtures. Run from the repo root.

## Setup

```bash
uv sync --extra multimodal
python tests/fixtures/videos/generate.py  # idempotent; only needed first time

# Bring up Milvus Standalone (skip if you already have one)
bash skills/zilliz-launchpad/scripts/start_milvus.sh
```

## Ingest the fixtures

```bash
python skills/zilliz-launchpad/scripts/zilliz_ops.py collect --input tests/fixtures/videos
python skills/zilliz-launchpad/scripts/zilliz_ops.py configure --use-case video-search
python skills/zilliz-launchpad/scripts/zilliz_ops.py plan
python skills/zilliz-launchpad/scripts/zilliz_ops.py execute --no-ui
```

Expect: `execute.json` lists all 5 videos under `processed_videos`, ~25 frame rows total, `ingest_path == video-batch`.

## Start the sidecar and UI

```bash
python skills/zilliz-launchpad/scripts/zilliz_ops.py execute --ui-port 8000
# in another shell
cd skills/zilliz-launchpad/scripts/ui && pnpm install && pnpm dev
```

Open http://localhost:3000.

## Smoke checklist

- [ ] `/info` shows `modality: "video"`, `data_shape: "video_dir"`, `video_static_prefix: "/videos"`.
- [ ] Type `red bar sweeping` — the primary card shows `clip_red_sweep.mp4` at `00:00` or similar with the thumbnail.
- [ ] Click the card — inline `<video>` mounts and seeks to the matched timestamp.
- [ ] Click one of the clustered frame thumbnails under the card — the same `<video>` element seeks to that frame's timestamp without reloading.
- [ ] Type `blue pulsing circle` — `clip_blue_pulse.mp4` is the top card.
- [ ] Click the upload button, pick `tests/fixtures/photos/sunset_01.jpg` — image→video query routes through `/search_image`, re-renders the grid with video cards.
- [ ] Verify fallback: set `LAUNCHPAD_VIDEO_STATIC_ROOT=/tmp/nonexistent`, rerun sidecar, observe each card shows the `video_url_warning` and an `Open file` link in place of the inline player.

## Quick CLI smoke (no UI)

```bash
python skills/zilliz-launchpad/scripts/zilliz_ops.py evaluate \
  --run-dir $(ls -t skills/zilliz-launchpad/scripts/runs | head -1 | xargs -I{} echo skills/zilliz-launchpad/scripts/runs/{}) \
  --query-image tests/fixtures/photos/sunset_01.jpg
```

Expect: top-k frame PKs with scores, exit 0.
