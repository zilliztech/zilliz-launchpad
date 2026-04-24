"""Sidecar /info and /search behavior for image collections.

Mocks the Milvus client and the CLIP text encoder so the test stays fast.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def image_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    plan = {
        "collection_name": "img_demo",
        "target_uri": "http://localhost:19530",
        "schema": {
            "primary_key": "image_path",
            "vector_field": "embedding",
            "text_field": None,
            "sparse_field": None,
            "extra_fields": [],
            "dim": 512,
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
            "image_path": "/photos/a.jpg",
            "thumbnail_b64": "AAA",
            "width": 256,
            "height": 256,
            "bytes": 1234,
        },
        {
            "image_path": "/photos/b.jpg",
            "thumbnail_b64": "BBB",
            "width": 320,
            "height": 240,
            "bytes": 4567,
        },
    ]
    collect = {"data_shape": "image_dir", "rows": rows}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "collect.json").write_text(json.dumps(collect))
    monkeypatch.setenv("LAUNCHPAD_RUN_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def sidecar_app(image_run_dir: Path):
    """Reload lib.ui so module-level caches pick up the new run dir."""
    import lib.ui as ui_mod

    importlib.reload(ui_mod)
    yield ui_mod.app
    importlib.reload(ui_mod)  # leave module clean for other tests


def test_info_reports_image_modality_and_thumbnails(sidecar_app):
    client = TestClient(sidecar_app)
    resp = client.get("/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["modality"] == "image"
    assert data["primary_key"] == "image_path"
    assert data["sparse_enabled"] is False
    assert data["has_thumbnails"] is True
    assert data["embedding"]["provider"] == "clip-local"


def test_image_search_routes_through_clip_text_encoder(sidecar_app):
    """Mock the Milvus client + CLIP text encoder; verify response shape and
    that thumbnails are joined from collect.json."""
    fake_client = MagicMock()
    fake_hit_a = MagicMock()
    fake_hit_a.entity = None
    fake_hit_a.score = 0.91
    fake_hit_a.get = lambda key: "/photos/a.jpg" if key == "image_path" else None
    fake_hit_b = MagicMock()
    fake_hit_b.entity = None
    fake_hit_b.score = 0.78
    fake_hit_b.get = lambda key: "/photos/b.jpg" if key == "image_path" else None
    fake_client.search.return_value = [[fake_hit_a, fake_hit_b]]

    with (
        patch("lib.ui._client", return_value=fake_client),
        patch("lib.ui.embed_text_with_clip", return_value=[[0.1] * 512]),
    ):
        client = TestClient(sidecar_app)
        resp = client.post("/search", json={"query": "sunset", "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["modality"] == "image"
    assert len(data["hits"]) == 2
    assert data["hits"][0]["id"] == "/photos/a.jpg"
    assert data["hits"][0]["fields"]["thumbnail_b64"] == "AAA"
    assert data["hits"][0]["fields"]["width"] == 256
    assert data["hits"][1]["fields"]["thumbnail_b64"] == "BBB"
    fake_client.search.assert_called_once()


def test_embed_image_returns_vector_and_dim(sidecar_app):
    with patch("lib.ui.embed_image_batch", return_value=[[0.25] * 512]):
        client = TestClient(sidecar_app)
        resp = client.post(
            "/embed_image",
            files={"file": ("q.jpg", b"\xff\xd8\xff\xe0fake", "image/jpeg")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["dim"] == 512
    assert len(data["embedding"]) == 512


def test_embed_image_rejects_large_upload(sidecar_app):
    # 11 MB payload — > 10 MB cap
    payload = b"x" * (11 * 1024 * 1024)
    with patch("lib.ui.embed_image_batch", return_value=[[0.0] * 512]):
        client = TestClient(sidecar_app)
        resp = client.post("/embed_image", files={"file": ("big.jpg", payload, "image/jpeg")})
    assert resp.status_code == 413
    assert resp.json()["detail"]["code"] == "upload_too_large"


def test_embed_image_rejects_undecodable_upload(sidecar_app):
    with patch(
        "lib.ui.embed_image_batch",
        side_effect=OSError("cannot identify image file"),
    ):
        client = TestClient(sidecar_app)
        resp = client.post(
            "/embed_image",
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "image_decode_failed"


def test_search_image_returns_search_response_shape(sidecar_app):
    fake_client = MagicMock()
    hit_a = MagicMock()
    hit_a.id = "/photos/a.jpg"
    hit_a.score = 0.91
    hit_a.entity = {"image_path": "/photos/a.jpg"}
    fake_client.search.return_value = [[hit_a]]
    with (
        patch("lib.ui._client", return_value=fake_client),
        patch("lib.search.embed_image_batch", return_value=[[0.1] * 512]),
    ):
        client = TestClient(sidecar_app)
        resp = client.post(
            "/search_image",
            files={"file": ("q.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
            data={"top_k": "5"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["modality"] == "image"
    assert data["mode"] == "dense"
    assert len(data["hits"]) == 1
    assert data["hits"][0]["id"] == "/photos/a.jpg"
    # Thumbnail metadata joined from collect.json fixture
    assert data["hits"][0]["fields"]["thumbnail_b64"] == "AAA"


def test_search_image_clamps_top_k(sidecar_app):
    fake_client = MagicMock()
    hit = MagicMock()
    hit.id = "/photos/a.jpg"
    hit.score = 0.5
    hit.entity = {"image_path": "/photos/a.jpg"}
    fake_client.search.return_value = [[hit]]
    with (
        patch("lib.ui._client", return_value=fake_client),
        patch("lib.search.embed_image_batch", return_value=[[0.1] * 512]),
    ):
        client = TestClient(sidecar_app)
        resp = client.post(
            "/search_image",
            files={"file": ("q.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
            data={"top_k": "500"},
        )
    assert resp.status_code == 200
    # Milvus was called with limit clamped to 100, not 500
    _, kwargs = fake_client.search.call_args
    assert kwargs["limit"] == 100


def test_search_image_on_text_collection_returns_400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    plan = {
        "collection_name": "movies",
        "target_uri": "http://localhost:19530",
        "schema": {
            "primary_key": "id",
            "vector_field": "embedding",
            "text_field": "body",
            "sparse_field": "sparse",
            "extra_fields": [],
            "dim": 1536,
        },
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dim": 1536,
            "modality": "text",
        },
        "sparse_enabled": True,
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setenv("LAUNCHPAD_RUN_DIR", str(tmp_path))
    import importlib as _importlib

    import lib.ui as ui_mod

    _importlib.reload(ui_mod)
    try:
        client = TestClient(ui_mod.app)
        resp = client.post(
            "/search_image",
            files={"file": ("q.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["code"] == "not_image_collection"
    finally:
        _importlib.reload(ui_mod)


def test_text_run_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The existing text path keeps modality=text and no thumbnail join."""
    plan = {
        "collection_name": "movies",
        "target_uri": "http://localhost:19530",
        "schema": {
            "primary_key": "id",
            "vector_field": "embedding",
            "text_field": "body",
            "sparse_field": "sparse",
            "extra_fields": [],
            "dim": 1536,
        },
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dim": 1536,
            "modality": "text",
        },
        "sparse_enabled": True,
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setenv("LAUNCHPAD_RUN_DIR", str(tmp_path))

    import lib.ui as ui_mod

    importlib.reload(ui_mod)
    try:
        client = TestClient(ui_mod.app)
        info = client.get("/info").json()
        assert info["modality"] == "text"
        assert info["has_thumbnails"] is False
    finally:
        importlib.reload(ui_mod)
