"""Generate synthetic video fixtures for the video-search test path.

Each clip is ~10 s at 10 fps with a deterministic, visually distinct motif so
frame embeddings cluster per-video. Running this script is idempotent — if the
.mp4 files already exist it overwrites them.

Usage (from the repo root):

    python tests/fixtures/videos/generate.py
"""

from __future__ import annotations

import math
from pathlib import Path

import av
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
FPS = 10
DURATION_S = 10
FRAME_COUNT = FPS * DURATION_S
WIDTH = 160
HEIGHT = 120


def _red_sweep(i: int) -> np.ndarray:
    """Vertical red bar sweeping left-to-right on a black background."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    x = int((i / FRAME_COUNT) * WIDTH)
    bar_w = 10
    frame[:, max(0, x - bar_w // 2) : min(WIDTH, x + bar_w // 2), 0] = 255
    return frame


def _blue_pulse(i: int) -> np.ndarray:
    """Pulsing blue disc in the middle of the frame."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    t = i / FPS
    radius = int(10 + 20 * (0.5 + 0.5 * math.sin(t * 2)))
    y, x = np.ogrid[:HEIGHT, :WIDTH]
    mask = (x - WIDTH // 2) ** 2 + (y - HEIGHT // 2) ** 2 <= radius**2
    frame[mask, 2] = 255
    return frame


def _green_zigzag(i: int) -> np.ndarray:
    """Green square bouncing vertically along a sinusoid path."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    t = i / FPS
    cx = int((i / FRAME_COUNT) * WIDTH)
    cy = int(HEIGHT // 2 + (HEIGHT // 3) * math.sin(t * 4))
    size = 12
    x0, y0 = max(0, cx - size), max(0, cy - size)
    x1, y1 = min(WIDTH, cx + size), min(HEIGHT, cy + size)
    frame[y0:y1, x0:x1, 1] = 255
    return frame


def _yellow_checkerboard(i: int) -> np.ndarray:
    """Yellow-and-black checkerboard with a phase offset per frame."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cell = 15
    phase = i % (cell * 2)
    y, x = np.indices((HEIGHT, WIDTH))
    mask = ((x + phase) // cell + y // cell) % 2 == 0
    frame[mask, 0] = 255
    frame[mask, 1] = 255
    return frame


def _white_dots(i: int) -> np.ndarray:
    """White pseudo-random dot pattern seeded per-frame for slow variation."""
    rng = np.random.default_rng(seed=42 + i // 5)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    ys = rng.integers(0, HEIGHT, size=80)
    xs = rng.integers(0, WIDTH, size=80)
    frame[ys, xs, :] = 255
    return frame


CLIPS: list[tuple[str, object]] = [
    ("clip_red_sweep.mp4", _red_sweep),
    ("clip_blue_pulse.mp4", _blue_pulse),
    ("clip_green_zigzag.mp4", _green_zigzag),
    ("clip_yellow_checkerboard.mp4", _yellow_checkerboard),
    ("clip_white_dots.mp4", _white_dots),
]


def render(path: Path, generator) -> None:  # type: ignore[no-untyped-def]
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("mpeg4", rate=FPS)
        stream.width = WIDTH
        stream.height = HEIGHT
        stream.pix_fmt = "yuv420p"
        for i in range(FRAME_COUNT):
            array = generator(i)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, gen in CLIPS:
        out_path = OUT_DIR / name
        render(out_path, gen)
        print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
