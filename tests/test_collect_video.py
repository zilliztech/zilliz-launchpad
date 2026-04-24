"""Phase 1 Collect — video directory branch."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("av")

from lib.errors import InvalidProfileError  # noqa: E402
from lib.phases.collect import run_collect  # noqa: E402

FIXTURE_VIDEOS = Path(__file__).parent / "fixtures" / "videos"


def _have_video_fixtures() -> bool:
    return any(FIXTURE_VIDEOS.glob("clip_*.mp4"))


pytestmark = pytest.mark.skipif(
    not _have_video_fixtures(),
    reason="video fixtures missing — run tests/fixtures/videos/generate.py",
)


def test_video_dir_produces_correct_collect_json(tmp_path: Path):
    result = run_collect(input_path=str(FIXTURE_VIDEOS), sample=None, out_dir=tmp_path)
    assert result["data_shape"] == "video_dir"
    assert result["suggested_primary_key"] == "frame_path"
    assert result["suggested_text_field"] is None
    assert result["video_count"] == 5
    # ~5 frames per 10s clip at 2s interval × 5 clips ≈ 25
    assert 20 <= result["record_count_estimate"] <= 30

    field_names = {f["name"] for f in result["fields"]}
    assert {"video_path", "t_seconds", "frame_path", "source_index"} <= field_names

    # Rows point at extracted JPEGs under out_dir/frames/...
    first = result["rows"][0]
    assert first["frame_path"].startswith(str(tmp_path))
    assert Path(first["frame_path"]).exists()
    assert first["video_path"].endswith(".mp4")
    assert "thumbnail_b64" in first
    assert first["t_seconds"] >= 0

    # configure.json didn't exist; defaults should have been used
    sampling = result["video_sampling"]
    assert sampling["strategy"] == "every_n_seconds"
    assert sampling["frame_interval_seconds"] == 2.0
    assert sampling["max_frames_per_video"] == 600


def test_no_supported_videos_errors_with_invalid_profile(tmp_path: Path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    (empty_dir / "notes.txt").write_text("not a video")

    # A .txt-only dir is treated as "not a video collection" → falls through
    # to the image branch, which also rejects with invalid_profile.
    with pytest.raises(InvalidProfileError) as exc:
        run_collect(input_path=str(empty_dir), sample=None, out_dir=tmp_path)
    assert "no supported" in exc.value.message


def test_corrupt_video_skipped_with_warning(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    video_dir = tmp_path / "clips"
    video_dir.mkdir()
    real = sorted(FIXTURE_VIDEOS.glob("*.mp4"))[:2]
    for p in real:
        shutil.copy(p, video_dir / p.name)
    (video_dir / "broken.mp4").write_bytes(b"definitely not an mpeg container")

    result = run_collect(input_path=str(video_dir), sample=None, out_dir=tmp_path)
    err = capsys.readouterr().err
    assert "broken.mp4" in err
    videos_in_rows = {r["video_path"] for r in result["rows"]}
    assert all("broken.mp4" not in v for v in videos_in_rows)
    # Two good clips, ~5 frames each
    assert result["video_count"] == 3  # listing still counts the corrupt one
    assert len(videos_in_rows) == 2


def test_max_frames_cap_triggers_interval_adjustment(tmp_path: Path):
    """With max_frames=3, a 10s clip at 2s interval must not exceed 3 rows per video."""
    # Simulate configure.json with a low cap
    (tmp_path / "configure.json").write_text(
        json.dumps(
            {
                "frame_interval_seconds": 2.0,
                "max_frames_per_video": 3,
                "sampling_strategy": "every_n_seconds",
                "scene_threshold": 0.3,
            }
        )
    )
    result = run_collect(input_path=str(FIXTURE_VIDEOS), sample=None, out_dir=tmp_path)
    per_video: dict[str, int] = {}
    for r in result["rows"]:
        per_video[r["video_path"]] = per_video.get(r["video_path"], 0) + 1
    assert all(n <= 3 for n in per_video.values()), per_video


def test_scene_change_strategy_routed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When configure says scene_change, sample_frames should route there."""
    import lib.video as video_mod

    called = {"scene": 0, "every": 0}

    real_scene = video_mod.sample_scene_change
    real_every = video_mod.sample_every_n_seconds

    def spy_scene(*args, **kw):
        called["scene"] += 1
        return real_scene(*args, **kw)

    def spy_every(*args, **kw):
        called["every"] += 1
        return real_every(*args, **kw)

    monkeypatch.setattr(video_mod, "sample_scene_change", spy_scene)
    monkeypatch.setattr(video_mod, "sample_every_n_seconds", spy_every)

    (tmp_path / "configure.json").write_text(
        json.dumps(
            {
                "sampling_strategy": "scene_change",
                "scene_threshold": 0.3,
                "frame_interval_seconds": 2.0,
                "max_frames_per_video": 600,
            }
        )
    )
    result = run_collect(input_path=str(FIXTURE_VIDEOS), sample=None, out_dir=tmp_path)

    # scene_change tried first (per video); if ffmpeg is absent it raises
    # FfmpegMissingError and the dispatcher falls back to every_n_seconds for
    # the same video. Either way, scene was attempted once per video.
    assert called["scene"] == 5
    assert result["video_sampling"]["strategy"] == "scene_change"
