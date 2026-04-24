"""Sidecar /info, /search, /video_frames for video collections."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def video_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Static root defaults to tmp_path.parent; lay out video files under tmp_path
    # so the relative-path resolution lands correctly.
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    video_a = clips_dir / "a.mp4"
    video_b = clips_dir / "b.mp4"
    video_a.write_bytes(b"fake a")
    video_b.write_bytes(b"fake b")

    plan = {
        "collection_name": "video_demo",
        "target_uri": "http://localhost:19530",
        "schema": {
            "primary_key": "frame_path",
            "vector_field": "embedding",
            "text_field": None,
            "sparse_field": None,
            "extra_fields": [
                {"name": "video_path", "type": "string", "max_length": 1024},
                {"name": "t_seconds", "type": "float"},
            ],
            "dim": 512,
            "is_video": True,
        },
        "embedding": {
            "provider": "clip-local",
            "model": "ViT-B-32",
            "dim": 512,
            "modality": "image",
            "device_hint": "cpu",
        },
        "sparse_enabled": False,
    }
    rows = [
        {
            "frame_path": str(tmp_path / "frames" / "a" / "0.jpg"),
            "video_path": str(video_a),
            "t_seconds": 0.0,
            "thumbnail_b64": "AAA",
            "source_index": 0,
        },
        {
            "frame_path": str(tmp_path / "frames" / "a" / "1.jpg"),
            "video_path": str(video_a),
            "t_seconds": 2.0,
            "thumbnail_b64": "BBB",
            "source_index": 1,
        },
    ]
    collect = {"data_shape": "video_dir", "rows": rows}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "collect.json").write_text(json.dumps(collect))
    monkeypatch.setenv("LAUNCHPAD_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("LAUNCHPAD_VIDEO_STATIC_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def sidecar_app(video_run_dir: Path):
    import lib.ui as ui_mod

    importlib.reload(ui_mod)
    yield ui_mod.app
    importlib.reload(ui_mod)


def _fake_hit(frame_path: str, video_path: str, t: float, score: float):
    hit = MagicMock()
    hit.score = score
    hit.entity = {
        "frame_path": frame_path,
        "video_path": video_path,
        "t_seconds": t,
    }
    return hit


def test_info_reports_video_modality(sidecar_app):
    client = TestClient(sidecar_app)
    resp = client.get("/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["modality"] == "video"
    assert data["primary_key"] == "frame_path"
    assert data["data_shape"] == "video_dir"
    assert data["video_static_prefix"] == "/videos"


def test_search_enriches_hits_with_video_fields(sidecar_app, video_run_dir: Path):
    clips_dir = video_run_dir / "clips"
    fake_client = MagicMock()
    fake_client.search.return_value = [
        [
            _fake_hit(
                str(video_run_dir / "frames" / "a" / "0.jpg"),
                str(clips_dir / "a.mp4"),
                0.0,
                0.9,
            )
        ]
    ]
    with (
        patch("lib.ui.MilvusClient", return_value=fake_client),
        patch("lib.ui.embed_text_with_clip", return_value=[[0.1] * 512]),
    ):
        client = TestClient(sidecar_app)
        resp = client.post("/search", json={"query": "red bar", "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["modality"] == "video"
    assert len(body["hits"]) == 1
    fields = body["hits"][0]["fields"]
    assert fields["video_path"] == str(clips_dir / "a.mp4")
    assert fields["t_seconds"] == 0.0
    assert fields["video_url"] == "/videos/clips/a.mp4"
    assert fields.get("video_url_warning") is None


def test_search_hit_outside_static_root_gets_null_url(
    sidecar_app, video_run_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A video path outside the static mount → video_url=None plus warning."""
    monkeypatch.setenv("LAUNCHPAD_VIDEO_STATIC_ROOT", str(video_run_dir / "clips"))
    # Reload so the new env var is read
    import lib.ui as ui_mod

    importlib.reload(ui_mod)

    outside = video_run_dir.parent / "elsewhere" / "c.mp4"
    fake_client = MagicMock()
    fake_client.search.return_value = [[_fake_hit("/frames/c/0.jpg", str(outside), 1.5, 0.8)]]
    with (
        patch("lib.ui.MilvusClient", return_value=fake_client),
        patch("lib.ui.embed_text_with_clip", return_value=[[0.1] * 512]),
    ):
        client = TestClient(ui_mod.app)
        resp = client.post("/search", json={"query": "whatever", "top_k": 1})
    fields = resp.json()["hits"][0]["fields"]
    assert fields["video_url"] is None
    assert "outside static mount" in fields["video_url_warning"]
    importlib.reload(ui_mod)


def test_video_frames_requires_prior_query(sidecar_app):
    client = TestClient(sidecar_app)
    resp = client.post("/video_frames", json={"video_path": "/a.mp4", "top_k": 3})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "no_prior_query"


def test_video_frames_reuses_last_query_vector(sidecar_app, video_run_dir: Path):
    clips_dir = video_run_dir / "clips"
    fake_client = MagicMock()
    fake_client.search.return_value = [
        [
            _fake_hit(
                str(video_run_dir / "frames" / "a" / "0.jpg"),
                str(clips_dir / "a.mp4"),
                0.0,
                0.9,
            )
        ]
    ]
    with (
        patch("lib.ui.MilvusClient", return_value=fake_client),
        patch("lib.ui.embed_text_with_clip", return_value=[[0.5] * 512]),
    ):
        client = TestClient(sidecar_app)
        # Seed a query first
        client.post("/search", json={"query": "seed", "top_k": 1})
        # Now ask for additional frames of that video
        resp = client.post(
            "/video_frames",
            json={"video_path": str(clips_dir / "a.mp4"), "top_k": 3},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["modality"] == "video"
    # The second search call used the cached query vector
    assert fake_client.search.call_count == 2
    filter_expr = fake_client.search.call_args.kwargs["filter"]
    assert filter_expr == f'video_path == "{clips_dir / "a.mp4"}"'


def test_video_frames_rejects_non_video_collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    plan = {
        "collection_name": "txt",
        "target_uri": "http://localhost:19530",
        "schema": {
            "primary_key": "id",
            "vector_field": "embedding",
            "text_field": "text",
            "sparse_field": None,
            "extra_fields": [],
            "dim": 512,
        },
        "embedding": {"provider": "openai", "model": "text-embedding-3-small", "dim": 512},
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setenv("LAUNCHPAD_RUN_DIR", str(tmp_path))
    monkeypatch.delenv("LAUNCHPAD_VIDEO_STATIC_ROOT", raising=False)
    import lib.ui as ui_mod

    importlib.reload(ui_mod)
    client = TestClient(ui_mod.app)
    resp = client.post("/video_frames", json={"video_path": "/a.mp4", "top_k": 3})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "not_video_collection"
    importlib.reload(ui_mod)
