"""Video frame sampler unit tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("av")

from lib.video import (  # noqa: E402
    FfmpegMissingError,
    VideoProbeError,
    _effective_interval,
    probe_video,
    sample_every_n_seconds,
    sample_frames,
    sample_scene_change,
)

FIXTURE_VIDEOS = Path(__file__).parent / "fixtures" / "videos"


@pytest.fixture
def red_sweep() -> Path:
    path = FIXTURE_VIDEOS / "clip_red_sweep.mp4"
    if not path.exists():
        pytest.skip("video fixtures not generated — run tests/fixtures/videos/generate.py")
    return path


def test_probe_video_reads_duration_fps_codec(red_sweep: Path):
    meta = probe_video(red_sweep)
    assert meta.duration_s > 9.0  # clips are 10s
    assert meta.fps == pytest.approx(10.0, rel=0.01)
    assert meta.codec  # e.g., "mpeg4"
    assert meta.width == 160
    assert meta.height == 120


def test_probe_failure_raises_videoprobeerror(tmp_path: Path):
    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"definitely not a valid container")
    # When ffprobe is absent, PyAV's error propagates as VideoProbeError.
    with (
        patch("lib.video.shutil.which", return_value=None),
        pytest.raises(VideoProbeError) as exc,
    ):
        probe_video(bad)
    assert str(bad) in str(exc.value.payload.get("path", ""))


def test_probe_falls_back_to_ffprobe_when_pyav_open_fails(tmp_path: Path, red_sweep: Path):
    """ffprobe path is exercised indirectly — simulate PyAV failure."""
    import lib.video as mod

    calls = {"av_tries": 0}

    class _Boom(Exception):
        pass

    def boom(*_a, **_kw):
        calls["av_tries"] += 1
        raise _Boom("simulated pyav failure")

    # If ffprobe isn't on PATH, the fallback raises VideoProbeError — skip.
    import shutil

    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe not installed")

    with patch.object(mod.av, "open", side_effect=boom):  # type: ignore[attr-defined]
        meta = probe_video(red_sweep)
    assert calls["av_tries"] == 1
    assert meta.duration_s > 0
    assert meta.width == 160


def test_effective_interval_respects_cap():
    # 100s video, 1s interval, cap=10 → widen to ~11s interval
    assert _effective_interval(100.0, 1.0, 10) == pytest.approx(100 / 9, rel=0.01)
    # Naive count under cap → no adjustment
    assert _effective_interval(20.0, 2.0, 600) == 2.0
    # Zero duration / interval degenerate → returns max(interval, 1.0)
    assert _effective_interval(0.0, 0.5, 10) == 1.0
    assert _effective_interval(10.0, 0.0, 10) == 1.0


def test_sample_every_n_seconds_row_count(red_sweep: Path, tmp_path: Path):
    rows = sample_every_n_seconds(red_sweep, interval_s=2.0, max_frames=600, out_dir=tmp_path)
    # 10s video at 2s interval → roughly 5 frames
    assert 4 <= len(rows) <= 6
    assert all(r.video_path == str(red_sweep) for r in rows)
    assert all(r.source_index == i for i, r in enumerate(rows))
    assert all(r.t_seconds >= 0 for r in rows)
    # Timestamps increase monotonically
    ts = [r.t_seconds for r in rows]
    assert ts == sorted(ts)
    # Frame JPEGs actually written
    for r in rows:
        assert Path(r.frame_path).exists()
        assert Path(r.frame_path).stat().st_size > 0


def test_sample_every_n_seconds_caps_at_max_frames(red_sweep: Path, tmp_path: Path):
    """10s video at 0.1s interval would be ~100 frames; cap to 8."""
    rows = sample_every_n_seconds(red_sweep, interval_s=0.1, max_frames=8, out_dir=tmp_path)
    assert len(rows) <= 8
    # Spread spans the duration — last timestamp is well past the first
    assert rows[-1].t_seconds > rows[0].t_seconds


def test_sample_scene_change_missing_ffmpeg_raises(red_sweep: Path, tmp_path: Path):
    with patch("lib.video.shutil.which", return_value=None), pytest.raises(FfmpegMissingError):
        sample_scene_change(red_sweep, threshold=0.3, max_frames=10, out_dir=tmp_path)


def test_sample_frames_falls_back_when_ffmpeg_missing(red_sweep: Path, tmp_path: Path, caplog):
    """scene_change strategy + no ffmpeg should fall back to interval with a warning."""
    caplog.set_level("WARNING")
    with patch("lib.video.shutil.which", return_value=None):
        rows = sample_frames(
            red_sweep,
            strategy="scene_change",
            interval_s=2.0,
            max_frames=600,
            scene_threshold=0.3,
            out_dir=tmp_path,
        )
    assert rows  # got rows via fallback
    assert "falling back" in caplog.text


def test_sample_frames_every_n_path(red_sweep: Path, tmp_path: Path):
    rows = sample_frames(
        red_sweep,
        strategy="every_n_seconds",
        interval_s=2.0,
        max_frames=600,
        scene_threshold=0.3,
        out_dir=tmp_path,
    )
    assert 4 <= len(rows) <= 6
