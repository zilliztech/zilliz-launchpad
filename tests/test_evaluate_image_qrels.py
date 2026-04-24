"""Image-to-image qrels loader and evaluation — Phase 5.

Covers the extended qrels shape introduced by issue #15: mixed text→image
and image→image rows in one file, missing-file skip behaviour, typed
error for malformed rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from lib.errors import InvalidProfileError
from lib.evaluator import QueryWithExpectedIds
from lib.phases.evaluate import (
    _evaluate_single_image,
    _load_image_qrels,
)

IMAGE_PLAN = {
    "collection_name": "img_demo",
    "target_uri": "http://localhost:19530",
    "schema": {
        "primary_key": "image_path",
        "vector_field": "embedding",
        "text_field": None,
        "sparse_field": None,
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


def _fake_hit(pk: str, score: float):
    hit = MagicMock()
    hit.id = pk
    hit.score = score
    hit.entity = {"image_path": pk}
    return hit


# --- Loader ----------------------------------------------------------------


def test_loader_parses_pure_image_to_image_qrels(tmp_path: Path):
    query_img = tmp_path / "query.jpg"
    query_img.write_bytes(b"\xff\xd8")
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(
        json.dumps(
            {
                "query_image_path": str(query_img),
                "expected_image_ids": ["/photos/a.jpg", "/photos/b.jpg"],
                "grade": 2,
            }
        )
        + "\n"
    )
    rows = _load_image_qrels(qrels)
    assert len(rows) == 1
    assert rows[0].query_image_path == str(query_img)
    assert rows[0].relevant_ids == ("/photos/a.jpg", "/photos/b.jpg")
    assert rows[0].grade == 2


def test_loader_parses_mixed_text_and_image_rows(tmp_path: Path):
    query_img = tmp_path / "query.jpg"
    query_img.write_bytes(b"\xff\xd8")
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(
        "\n".join(
            [
                json.dumps({"query_text": "sunset", "image_paths": ["a.jpg"]}),
                json.dumps(
                    {
                        "query_image_path": str(query_img),
                        "expected_image_ids": ["b.jpg"],
                    }
                ),
            ]
        )
    )
    rows = _load_image_qrels(qrels)
    assert len(rows) == 2
    assert rows[0].query_image_path is None
    assert rows[0].relevant_ids == ("a.jpg",)
    assert rows[1].query_image_path == str(query_img)
    assert rows[1].relevant_ids == ("b.jpg",)


def test_loader_rejects_row_with_neither_key(tmp_path: Path):
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(json.dumps({"expected_image_ids": ["a.jpg"]}))
    with pytest.raises(InvalidProfileError) as exc:
        _load_image_qrels(qrels)
    assert "query_text" in exc.value.payload["reason"]
    assert "query_image_path" in exc.value.payload["reason"]


def test_loader_rejects_image_row_without_expected_ids(tmp_path: Path):
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(json.dumps({"query_image_path": "/tmp/q.jpg"}) + "\n")
    with pytest.raises(InvalidProfileError):
        _load_image_qrels(qrels)


# --- Evaluation ------------------------------------------------------------


def test_evaluate_single_image_routes_image_rows_through_image_encoder(
    tmp_path: Path,
):
    query_img = tmp_path / "query.jpg"
    query_img.write_bytes(b"\xff\xd8\xff\xe0fake")
    queries = [
        QueryWithExpectedIds(
            query=str(query_img),
            relevant_ids=("/photos/a.jpg",),
            query_image_path=str(query_img),
        ),
    ]
    fake_client = MagicMock()
    fake_client.search.return_value = [[_fake_hit("/photos/a.jpg", 0.9)]]

    with (
        patch("lib.phases.evaluate.MilvusClient", return_value=fake_client),
        patch("lib.search.embed_image_batch", return_value=[[0.1] * 512]) as mock_encode,
        patch("lib.phases.evaluate.embed_text_with_clip") as mock_text,
    ):
        out = _evaluate_single_image(label="base", plan=IMAGE_PLAN, queries=queries, concurrency=1)
    # Image row routed through image encoder, not text
    mock_encode.assert_called()
    mock_text.assert_not_called()
    assert out["retrieval"]["recall@10"] == pytest.approx(1.0)


def test_evaluate_single_image_caches_image_vectors(tmp_path: Path):
    query_img = tmp_path / "query.jpg"
    query_img.write_bytes(b"\xff\xd8")
    queries = [
        QueryWithExpectedIds(
            query=str(query_img),
            relevant_ids=("/photos/a.jpg",),
            query_image_path=str(query_img),
        ),
        QueryWithExpectedIds(
            query=str(query_img),
            relevant_ids=("/photos/a.jpg",),
            query_image_path=str(query_img),
        ),
    ]
    fake_client = MagicMock()
    fake_client.search.return_value = [[_fake_hit("/photos/a.jpg", 0.9)]]

    with (
        patch("lib.phases.evaluate.MilvusClient", return_value=fake_client),
        patch("lib.search.embed_image_batch", return_value=[[0.1] * 512]) as mock_encode,
    ):
        _evaluate_single_image(label="base", plan=IMAGE_PLAN, queries=queries, concurrency=1)
    # Two identical rows → encoder called once because of the in-run cache
    assert mock_encode.call_count == 1


def test_evaluate_single_image_skips_missing_file_and_continues(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    present = tmp_path / "present.jpg"
    present.write_bytes(b"\xff\xd8")
    queries = [
        QueryWithExpectedIds(
            query=str(present),
            relevant_ids=("/photos/a.jpg",),
            query_image_path=str(present),
        ),
        QueryWithExpectedIds(
            query=str(tmp_path / "missing.jpg"),
            relevant_ids=("/photos/b.jpg",),
            query_image_path=str(tmp_path / "missing.jpg"),
        ),
    ]
    fake_client = MagicMock()
    fake_client.search.return_value = [[_fake_hit("/photos/a.jpg", 0.9)]]
    with (
        patch("lib.phases.evaluate.MilvusClient", return_value=fake_client),
        patch("lib.search.embed_image_batch", return_value=[[0.1] * 512]),
    ):
        out = _evaluate_single_image(label="base", plan=IMAGE_PLAN, queries=queries, concurrency=1)
    # The missing file does not abort the run
    assert "skipped_queries" in out
    assert out["skipped_queries"] == [str(tmp_path / "missing.jpg")]
    # The surviving query still scored
    assert out["retrieval"]["recall@10"] == pytest.approx(1.0)


def test_evaluate_single_image_mixed_rows_route_correctly(tmp_path: Path):
    query_img = tmp_path / "query.jpg"
    query_img.write_bytes(b"\xff\xd8")
    queries = [
        QueryWithExpectedIds(query="sunset", relevant_ids=("/photos/sun.jpg",)),
        QueryWithExpectedIds(
            query=str(query_img),
            relevant_ids=("/photos/a.jpg",),
            query_image_path=str(query_img),
        ),
    ]
    fake_client = MagicMock()
    fake_client.search.side_effect = [
        [[_fake_hit("/photos/sun.jpg", 0.8)]],
        [[_fake_hit("/photos/a.jpg", 0.9)]],
        # Latency pass replays both queries, so the mock needs more returns:
        [[_fake_hit("/photos/sun.jpg", 0.8)]],
        [[_fake_hit("/photos/a.jpg", 0.9)]],
    ]
    with (
        patch("lib.phases.evaluate.MilvusClient", return_value=fake_client),
        patch("lib.search.embed_image_batch", return_value=[[0.1] * 512]) as mock_img,
        patch(
            "lib.phases.evaluate.embed_text_with_clip",
            return_value=[[0.2] * 512],
        ) as mock_text,
    ):
        out = _evaluate_single_image(label="base", plan=IMAGE_PLAN, queries=queries, concurrency=1)
    # Each branch was exercised at least once
    assert mock_img.called
    assert mock_text.called
    assert out["retrieval"]["recall@10"] == pytest.approx(1.0)
