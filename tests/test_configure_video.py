"""Phase 2 Configure — video-search use case."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from lib.errors import InvalidProfileError
from lib.phases.configure import run_configure


def _write_collect(out_dir: Path, data_shape: str) -> None:
    (out_dir / "collect.json").write_text(json.dumps({"data_shape": data_shape}))


def test_video_search_defaults_populated(tmp_path: Path):
    _write_collect(tmp_path, "video_dir")
    data = run_configure(
        from_json=None,
        out_dir=tmp_path,
        overrides={"use_case": "video-search", "deployment_target": "local-standalone"},
    )
    assert data["use_case"] == "video-search"
    assert data["hybrid_preference"] is False
    assert data["reranker_preference"] == "none"
    assert data["frame_interval_seconds"] == 2.0
    assert data["max_frames_per_video"] == 600
    assert data["sampling_strategy"] == "every_n_seconds"


def test_video_search_respects_user_overrides(tmp_path: Path):
    _write_collect(tmp_path, "video_dir")
    data = run_configure(
        from_json=None,
        out_dir=tmp_path,
        overrides={
            "use_case": "video-search",
            "deployment_target": "local-standalone",
            "frame_interval_seconds": 5.0,
            "max_frames_per_video": 120,
            "sampling_strategy": "scene_change",
            "scene_threshold": 0.5,
        },
    )
    assert data["frame_interval_seconds"] == 5.0
    assert data["max_frames_per_video"] == 120
    assert data["sampling_strategy"] == "scene_change"
    assert data["scene_threshold"] == 0.5


def test_video_search_warns_on_conflicting_hybrid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    _write_collect(tmp_path, "video_dir")
    profile = {
        "use_case": "video-search",
        "deployment_target": "local-standalone",
        "hybrid_preference": "on",
        "reranker_preference": "cohere-rerank-3",
    }
    (tmp_path / "profile.json").write_text(json.dumps(profile))
    run_configure(from_json=str(tmp_path / "profile.json"), out_dir=tmp_path)
    err = capsys.readouterr().err
    assert "hybrid_preference" in err
    assert "reranker_preference" in err


def test_video_dir_with_text_use_case_raises(tmp_path: Path):
    _write_collect(tmp_path, "video_dir")
    with pytest.raises(InvalidProfileError) as exc:
        run_configure(
            from_json=None,
            out_dir=tmp_path,
            overrides={"use_case": "rag", "deployment_target": "local-standalone"},
        )
    assert "video_dir" in exc.value.message
    assert "video-search" in exc.value.message


def test_video_search_against_text_collect_raises(tmp_path: Path):
    _write_collect(tmp_path, "jsonl")
    with pytest.raises(InvalidProfileError) as exc:
        run_configure(
            from_json=None,
            out_dir=tmp_path,
            overrides={"use_case": "video-search", "deployment_target": "local-standalone"},
        )
    assert "video-search" in exc.value.message


def test_scene_change_without_ffmpeg_warns(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _write_collect(tmp_path, "video_dir")
    with patch("shutil.which", return_value=None):
        run_configure(
            from_json=None,
            out_dir=tmp_path,
            overrides={
                "use_case": "video-search",
                "deployment_target": "local-standalone",
                "sampling_strategy": "scene_change",
            },
        )
    err = capsys.readouterr().err
    assert "ffmpeg" in err
    assert "fall back" in err
