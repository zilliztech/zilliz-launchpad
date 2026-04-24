# Video fixture provenance

5 synthetic 160×120 MPEG-4 clips, 10 fps, 10 seconds each (~100 KB total), used by the video-search test path. Each clip has a deterministic, visually distinct motif so CLIP frame embeddings cluster per-video (the core property the recall tests exercise).

| File | Motif |
| --- | --- |
| `clip_red_sweep.mp4` | Vertical red bar sweeping left-to-right on black |
| `clip_blue_pulse.mp4` | Pulsing blue disc, radius modulated by a sine |
| `clip_green_zigzag.mp4` | Green square bouncing on a sinusoid path, moving left-to-right |
| `clip_yellow_checkerboard.mp4` | Yellow / black checkerboard with phase drift |
| `clip_white_dots.mp4` | Pseudo-random white dot field, seeded for slow variation |

Regenerate with `python tests/fixtures/videos/generate.py` (needs the `[multimodal]` extra for `av`). Output is deterministic per `numpy.random.default_rng(seed=42)` for the dots clip; all others are pure arithmetic. License is CC0 — these are our own synthetic patterns.
