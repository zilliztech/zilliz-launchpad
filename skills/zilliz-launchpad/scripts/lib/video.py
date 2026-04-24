"""Video frame sampling used by Phase 1 Collect and Phase 4 Execute.

Keeps two sampling strategies:

- ``every_n_seconds`` — primary path, uses PyAV to decode frames at target
  timestamps. If the naive frame count would exceed ``max_frames`` the
  effective interval is adjusted upward so we land exactly on ``max_frames``.
- ``scene_change`` — shells out to the ``ffmpeg`` binary with
  ``select='gt(scene,<threshold>)'`` and parses the ``showinfo`` output.
  Raises ``FfmpegMissingError`` if the binary is not on ``PATH``.

PyAV lives behind the ``[multimodal]`` extra, so importing this module is
gated to callers that already depend on the extra (collect + execute both do
when they detect a video directory).
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import LaunchpadError
from .image_io import make_thumbnail_b64

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES: frozenset[str] = frozenset({".mp4", ".mov", ".mkv", ".webm"})


class FfmpegMissingError(LaunchpadError):
    code = "ffmpeg_missing"

    def __init__(self) -> None:
        super().__init__(
            "ffmpeg binary not found on PATH; required for scene-change sampling",
            remediation=(
                "Install ffmpeg (e.g., `brew install ffmpeg`) or use "
                "sampling_strategy=every_n_seconds"
            ),
        )


class VideoProbeError(LaunchpadError):
    code = "video_probe_failed"

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            f"Could not probe video '{path}': {reason}",
            path=path,
            reason=reason,
        )


@dataclass
class VideoMetadata:
    duration_s: float
    fps: float
    codec: str
    width: int
    height: int


@dataclass
class FrameRow:
    video_path: str
    t_seconds: float
    frame_path: str
    thumbnail_b64: str
    source_index: int

    def to_dict(self) -> dict[str, object]:
        return {
            "video_path": self.video_path,
            "t_seconds": self.t_seconds,
            "frame_path": self.frame_path,
            "thumbnail_b64": self.thumbnail_b64,
            "source_index": self.source_index,
        }


def list_videos(directory: Path) -> list[Path]:
    """Return supported video files under ``directory``, sorted for determinism."""
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )


def probe_video(path: Path) -> VideoMetadata:
    """Open the container with PyAV and read its primary video stream metadata.

    Falls back to ``ffprobe`` if PyAV fails to open the file (corrupt container,
    unsupported codec). Raises ``VideoProbeError`` if both fail.
    """
    try:
        import av  # noqa: PLC0415

        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                raise VideoProbeError(str(path), "no video stream in container")
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            duration_s = (
                float(stream.duration * stream.time_base)
                if stream.duration and stream.time_base
                else 0.0
            )
            if duration_s <= 0 and container.duration:
                duration_s = container.duration / 1_000_000
            codec_ctx = stream.codec_context
            return VideoMetadata(
                duration_s=duration_s,
                fps=fps,
                codec=str(codec_ctx.name or ""),
                width=int(getattr(codec_ctx, "width", 0) or 0),
                height=int(getattr(codec_ctx, "height", 0) or 0),
            )
    except VideoProbeError:
        raise
    except Exception as av_exc:  # noqa: BLE001
        probe = shutil.which("ffprobe")
        if probe is None:
            raise VideoProbeError(str(path), f"PyAV open failed: {av_exc}") from av_exc
        try:
            out = subprocess.check_output(
                [
                    probe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,width,height,avg_frame_rate,duration",
                    "-of",
                    "default=noprint_wrappers=1",
                    str(path),
                ],
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise VideoProbeError(
                str(path), f"ffprobe failed: {exc.stderr or exc.stdout or exc}"
            ) from exc
        fields: dict[str, str] = {}
        for line in out.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                fields[k.strip()] = v.strip()
        rate = fields.get("avg_frame_rate", "0/1")
        try:
            num, den = rate.split("/") if "/" in rate else (rate, "1")
            fps = float(num) / float(den) if float(den) else 0.0
        except ValueError:
            fps = 0.0
        return VideoMetadata(
            duration_s=float(fields.get("duration", "0") or 0),
            fps=fps,
            codec=fields.get("codec_name", ""),
            width=int(fields.get("width", "0") or 0),
            height=int(fields.get("height", "0") or 0),
        )


def _effective_interval(duration_s: float, interval_s: float, max_frames: int) -> float:
    """Widen the interval so we never emit more than ``max_frames`` frames."""
    if duration_s <= 0 or interval_s <= 0:
        return max(interval_s, 1.0)
    naive_count = int(duration_s / interval_s) + 1
    if naive_count <= max_frames:
        return interval_s
    return duration_s / max(1, max_frames - 1)


def _save_frame(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = img.convert("RGB") if img.mode != "RGB" else img
    rgb.save(path, format="JPEG", quality=85)


def sample_every_n_seconds(
    video_path: Path,
    *,
    interval_s: float,
    max_frames: int,
    out_dir: Path,
) -> list[FrameRow]:
    """Decode ``video_path`` with PyAV, emit a frame every ``interval_s``.

    Writes each frame as a JPEG under ``out_dir/<video_stem>/<ordinal>.jpg``
    and returns the row list. Skips silently if the stream has zero duration.
    """
    import av  # noqa: PLC0415

    meta = probe_video(video_path)
    eff_interval = _effective_interval(meta.duration_s, interval_s, max_frames)
    target_times = [i * eff_interval for i in range(max_frames)]
    target_times = [t for t in target_times if t < meta.duration_s or meta.duration_s == 0]
    if not target_times:
        target_times = [0.0]

    rows: list[FrameRow] = []
    stem = video_path.stem
    frame_dir = out_dir / stem
    frame_dir.mkdir(parents=True, exist_ok=True)

    with av.open(str(video_path)) as container:
        stream = next(s for s in container.streams if s.type == "video")
        time_base = float(stream.time_base or 0) or 1 / (meta.fps or 30)
        next_target_idx = 0
        for packet in container.demux(stream):
            if next_target_idx >= len(target_times):
                break
            for frame in packet.decode():
                if next_target_idx >= len(target_times):
                    break
                # PyAV's demux() may yield non-video frame types when the
                # container has side streams; the `if stream.type == "video"`
                # filter above ensures we only get VideoFrames in practice.
                pts = getattr(frame, "pts", None)
                pts_s = float(pts * time_base) if pts is not None else 0.0
                target = target_times[next_target_idx]
                if pts_s >= target:
                    img = frame.to_image()  # type: ignore[union-attr, unused-ignore]
                    frame_path = frame_dir / f"{next_target_idx}.jpg"
                    _save_frame(img, frame_path)
                    rows.append(
                        FrameRow(
                            video_path=str(video_path),
                            t_seconds=round(pts_s, 3),
                            frame_path=str(frame_path),
                            thumbnail_b64=make_thumbnail_b64(img),
                            source_index=next_target_idx,
                        )
                    )
                    next_target_idx += 1

    return rows


_SHOWINFO_PTS_RE = re.compile(r"pts_time:([0-9.]+)")


def sample_scene_change(
    video_path: Path,
    *,
    threshold: float,
    max_frames: int,
    out_dir: Path,
) -> list[FrameRow]:
    """Shell out to ``ffmpeg`` with a scene-change select filter.

    Parses the ``showinfo`` output to recover timestamps and then uses PyAV
    to pull matching frames back out. Raises ``FfmpegMissingError`` when the
    binary is missing — callers can catch this to fall back to interval mode.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FfmpegMissingError()

    proc = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(video_path),
            "-vf",
            f"select='gt(scene,{threshold})',showinfo",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    timestamps: list[float] = []
    for match in _SHOWINFO_PTS_RE.finditer(proc.stderr):
        try:
            timestamps.append(float(match.group(1)))
        except ValueError:
            continue
        if len(timestamps) >= max_frames:
            break
    if not timestamps:
        # No scene change detected (or filter rejected everything). Fall back
        # to a single frame at t=0 so callers never get a row-less row list.
        timestamps = [0.0]

    import av  # noqa: PLC0415

    rows: list[FrameRow] = []
    stem = video_path.stem
    frame_dir = out_dir / stem
    frame_dir.mkdir(parents=True, exist_ok=True)

    with av.open(str(video_path)) as container:
        stream = next(s for s in container.streams if s.type == "video")
        time_base = float(stream.time_base or 0) or 1 / 30
        next_idx = 0
        for packet in container.demux(stream):
            if next_idx >= len(timestamps):
                break
            for frame in packet.decode():
                if next_idx >= len(timestamps):
                    break
                pts = getattr(frame, "pts", None)
                pts_s = float(pts * time_base) if pts is not None else 0.0
                target = timestamps[next_idx]
                if pts_s >= target:
                    img = frame.to_image()  # type: ignore[union-attr, unused-ignore]
                    frame_path = frame_dir / f"{next_idx}.jpg"
                    _save_frame(img, frame_path)
                    rows.append(
                        FrameRow(
                            video_path=str(video_path),
                            t_seconds=round(pts_s, 3),
                            frame_path=str(frame_path),
                            thumbnail_b64=make_thumbnail_b64(img),
                            source_index=next_idx,
                        )
                    )
                    next_idx += 1
    return rows


def sample_frames(
    video_path: Path,
    *,
    strategy: str,
    interval_s: float,
    max_frames: int,
    scene_threshold: float,
    out_dir: Path,
) -> list[FrameRow]:
    """Dispatch to the configured strategy with a graceful ffmpeg fallback."""
    if strategy == "scene_change":
        try:
            return sample_scene_change(
                video_path,
                threshold=scene_threshold,
                max_frames=max_frames,
                out_dir=out_dir,
            )
        except FfmpegMissingError:
            logger.warning(
                "scene-change sampling requested but ffmpeg not on PATH — falling back "
                "to every_n_seconds for %s",
                video_path,
            )
    return sample_every_n_seconds(
        video_path,
        interval_s=interval_s,
        max_frames=max_frames,
        out_dir=out_dir,
    )
