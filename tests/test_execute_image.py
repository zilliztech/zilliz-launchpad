"""Phase 4 Execute — image branch.

Heavy real-CLIP tests are marked `multimodal` so they only run in the
`[multimodal]` CI matrix slot. Plain unit tests use a fake embedder.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lib.errors import MissingCredentialError
from lib.phases.execute import _ingest_image_rows, run_execute

FIXTURE_PHOTOS = Path(__file__).parent / "fixtures" / "photos"


# ---------------------------------------------------------------------------
# Fake embedder + Milvus client helpers
# ---------------------------------------------------------------------------


def _fake_embedder(dim: int = 512):
    def _run(paths: Iterable[Any]) -> list[list[float]]:
        return [[0.1 * (i + 1)] * dim for i, _ in enumerate(list(paths))]

    return _run


class _FakeMilvusClient:
    def __init__(self) -> None:
        self.upserts: list[list[dict[str, Any]]] = []
        self.fail_on_paths: set[str] = set()

    def upsert(self, *, collection_name: str, data: list[dict[str, Any]]):
        for row in data:
            if row.get("image_path") in self.fail_on_paths:
                raise RuntimeError(f"forced upsert failure: {row['image_path']}")
        self.upserts.append(list(data))


# ---------------------------------------------------------------------------
# Resumability + per-file failure (no torch needed)
# ---------------------------------------------------------------------------


def _rows_for(paths: list[Path]) -> list[dict[str, Any]]:
    return [{"image_path": str(p), "width": 100, "height": 100, "bytes": 1234} for p in paths]


def test_ingest_image_rows_happy_path(tmp_path: Path):
    paths = sorted(FIXTURE_PHOTOS.glob("*.jpg"))[:5]
    rows = _rows_for(paths)
    client = _FakeMilvusClient()
    new, skipped, batches = _ingest_image_rows(
        client=client,
        collection="x",
        rows=rows,
        pk_field="image_path",
        vector_field="embedding",
        extra_keys=("width", "height", "bytes"),
        embedder_fn=_fake_embedder(),
        out_dir=tmp_path,
        processed=set(),
    )
    assert len(new) == 5
    assert skipped == []
    assert batches == 1
    snap = json.loads((tmp_path / "execute.json").read_text())
    assert sorted(snap["processed_files"]) == sorted(str(p) for p in paths)


def test_ingest_image_rows_skips_already_processed(tmp_path: Path):
    paths = sorted(FIXTURE_PHOTOS.glob("*.jpg"))[:5]
    rows = _rows_for(paths)
    already = {str(paths[0]), str(paths[2])}
    client = _FakeMilvusClient()
    new, _, _ = _ingest_image_rows(
        client=client,
        collection="x",
        rows=rows,
        pk_field="image_path",
        vector_field="embedding",
        extra_keys=(),
        embedder_fn=_fake_embedder(),
        out_dir=tmp_path,
        processed=set(already),
    )
    assert len(new) == 3
    assert all(p not in already for p in new)


def test_ingest_image_rows_records_skipped_on_upsert_failure(tmp_path: Path):
    paths = sorted(FIXTURE_PHOTOS.glob("*.jpg"))[:3]
    rows = _rows_for(paths)
    client = _FakeMilvusClient()
    client.fail_on_paths = {str(paths[1])}
    new, skipped, batches = _ingest_image_rows(
        client=client,
        collection="x",
        rows=rows,
        pk_field="image_path",
        vector_field="embedding",
        extra_keys=(),
        embedder_fn=_fake_embedder(),
        out_dir=tmp_path,
        processed=set(),
    )
    # Upsert fails for the whole batch (all 3 paths)
    assert new == []
    assert len(skipped) == 3
    assert batches == 0


def test_ingest_image_rows_records_skipped_on_embed_failure(tmp_path: Path):
    paths = sorted(FIXTURE_PHOTOS.glob("*.jpg"))[:2]
    rows = _rows_for(paths)
    client = _FakeMilvusClient()

    def _bad(_paths):
        raise RuntimeError("embed boom")

    new, skipped, _ = _ingest_image_rows(
        client=client,
        collection="x",
        rows=rows,
        pk_field="image_path",
        vector_field="embedding",
        extra_keys=(),
        embedder_fn=_bad,
        out_dir=tmp_path,
        processed=set(),
    )
    assert new == []
    assert len(skipped) == 2
    assert all("embed boom" in s["reason"] for s in skipped)


# ---------------------------------------------------------------------------
# Pre-flight: missing Voyage key fails before Milvus
# ---------------------------------------------------------------------------


def test_voyage_missing_key_errors_before_milvus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    plan = {
        "collection_name": "x",
        "target_uri": "http://localhost:19530",
        "schema": {
            "primary_key": "image_path",
            "text_field": None,
            "vector_field": "embedding",
            "dim": 1024,
            "sparse_field": None,
            "extra_fields": [],
        },
        "embedding": {
            "provider": "voyage",
            "model": "voyage-multimodal-3",
            "dim": 1024,
            "modality": "image",
            "device_hint": "cpu",
        },
        "sparse_enabled": False,
        "index": {"type": "HNSW", "metric": "COSINE", "params": {"M": 16, "efConstruction": 200}},
        "reranker": None,
        "chunking": {"size": 512, "overlap": 64},
        "deployment_target": "local-standalone",
    }
    paths = sorted(FIXTURE_PHOTOS.glob("*.jpg"))[:2]
    collect = {
        "data_shape": "image_dir",
        "rows": _rows_for(paths),
        "fields": [{"name": "image_path", "type": "string"}],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "collect.json").write_text(json.dumps(collect))

    # If MilvusClient were constructed we'd hit an unrelated connection error;
    # the test asserts MissingCredentialError fires first.
    with patch("lib.phases.execute.MilvusClient", side_effect=AssertionError("touched milvus")):
        with pytest.raises(MissingCredentialError) as exc:
            run_execute(
                out_dir=tmp_path,
                sample=None,
                input_path=None,
                ui_port=8000,
                start_ui=False,
            )
    assert exc.value.payload["env_var"] == "VOYAGE_API_KEY"


# ---------------------------------------------------------------------------
# CLIP smoke test — opt-in via [multimodal] extra
# ---------------------------------------------------------------------------


@pytest.mark.multimodal
def test_clip_local_embed_image_and_text_match_dim():
    """Sanity: ViT-B-32 returns 512-dim normalized vectors for both modalities."""
    from lib.embeddings import embed_image_batch, embed_text_with_clip

    img_path = sorted(FIXTURE_PHOTOS.glob("*.jpg"))[0]
    img_vec = embed_image_batch([img_path])
    txt_vec = embed_text_with_clip(["a photograph"])
    assert len(img_vec) == 1
    assert len(img_vec[0]) == 512
    assert len(txt_vec) == 1
    assert len(txt_vec[0]) == 512
    # Normalized → norm ~ 1.0
    norm = sum(x * x for x in img_vec[0]) ** 0.5
    assert 0.99 < norm < 1.01


@pytest.mark.multimodal
def test_clip_local_full_batch_dimensions():
    """Embed all 20 fixture images in one call; check shape and determinism."""
    from lib.embeddings import embed_image_batch

    paths = sorted(FIXTURE_PHOTOS.glob("*.jpg"))
    assert len(paths) == 20
    vecs = embed_image_batch(paths)
    assert len(vecs) == 20
    assert all(len(v) == 512 for v in vecs)
    # Re-run for the cache hit path → identical vectors.
    vecs2 = embed_image_batch(paths)
    assert vecs == vecs2


# ---------------------------------------------------------------------------
# Smoke: prefetch_clip is importable and idempotent
# ---------------------------------------------------------------------------


@pytest.mark.multimodal
def test_prefetch_clip_is_idempotent():
    from lib.embeddings import prefetch_clip

    prefetch_clip()
    prefetch_clip()
