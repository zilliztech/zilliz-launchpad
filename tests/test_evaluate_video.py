"""Phase 5 Evaluate — video-level metrics and qrels."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from lib.errors import InvalidProfileError, JudgeUnavailableError
from lib.evaluator import QueryWithExpectedIds
from lib.phases.evaluate import (
    _derive_video_queries,
    _evaluate_single_video,
    _load_video_qrels,
)

VIDEO_PLAN = {
    "collection_name": "video_demo",
    "target_uri": "http://localhost:19530",
    "schema": {
        "primary_key": "frame_path",
        "vector_field": "embedding",
        "text_field": None,
        "sparse_field": None,
        "dim": 512,
        "is_video": True,
        "extra_fields": [
            {"name": "video_path", "type": "string", "max_length": 1024},
            {"name": "t_seconds", "type": "float"},
        ],
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


def _fake_hit(frame_path: str, video_path: str, score: float):
    hit = MagicMock()
    hit.id = frame_path
    hit.score = score
    hit.entity = {"frame_path": frame_path, "video_path": video_path}
    return hit


# --- Loader ---------------------------------------------------------------


def test_loader_parses_video_level_rows(tmp_path: Path):
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(
        json.dumps({"query_text": "sunset", "expected_video_ids": ["a.mp4", "b.mp4"]}) + "\n"
    )
    rows = _load_video_qrels(qrels)
    assert len(rows) == 1
    assert rows[0].expected_video_ids == ("a.mp4", "b.mp4")
    assert rows[0].granularity == "video"


def test_loader_parses_image_to_video_row(tmp_path: Path):
    query_img = tmp_path / "query.jpg"
    query_img.write_bytes(b"\xff\xd8")
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(
        json.dumps(
            {
                "query_image_path": str(query_img),
                "expected_video_ids": ["a.mp4"],
            }
        )
        + "\n"
    )
    rows = _load_video_qrels(qrels)
    assert len(rows) == 1
    assert rows[0].query_image_path == str(query_img)
    assert rows[0].expected_video_ids == ("a.mp4",)


def test_loader_parses_mixed_frame_and_video_rows(tmp_path: Path):
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(
        "\n".join(
            [
                json.dumps({"query_text": "frame-level", "expected_image_ids": ["f1.jpg"]}),
                json.dumps({"query_text": "video-level", "expected_video_ids": ["a.mp4"]}),
            ]
        )
    )
    rows = _load_video_qrels(qrels)
    assert len(rows) == 2
    assert rows[0].granularity == "frame"
    assert rows[0].relevant_ids == ("f1.jpg",)
    assert rows[1].granularity == "video"
    assert rows[1].expected_video_ids == ("a.mp4",)


def test_loader_rejects_row_with_neither_expectation(tmp_path: Path):
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(json.dumps({"query_text": "no expectations"}) + "\n")
    with pytest.raises(InvalidProfileError):
        _load_video_qrels(qrels)


def test_loader_rejects_row_with_neither_query(tmp_path: Path):
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(json.dumps({"expected_video_ids": ["a.mp4"]}) + "\n")
    with pytest.raises(InvalidProfileError):
        _load_video_qrels(qrels)


# --- Evaluation ----------------------------------------------------------


def test_video_level_recall_counts_any_frame_from_expected_video():
    queries = [
        QueryWithExpectedIds(
            query="scene-a",
            expected_video_ids=("a.mp4",),
            granularity="video",
        ),
    ]
    fake_client = MagicMock()
    # Top hits are all frames from the target video but different frame PKs
    fake_client.search.return_value = [
        [
            _fake_hit("a.mp4#frame-3", "a.mp4", 0.9),
            _fake_hit("b.mp4#frame-0", "b.mp4", 0.7),
        ]
    ]
    with (
        patch("lib.phases.evaluate.MilvusClient", return_value=fake_client),
        patch("lib.phases.evaluate.embed_text_with_clip", return_value=[[0.1] * 512]),
    ):
        out = _evaluate_single_video(label="base", plan=VIDEO_PLAN, queries=queries, concurrency=1)
    # Exactly one expected video (a.mp4), one hit from it → recall@k = 1.0
    recall_video = {k: v for k, v in out["retrieval"].items() if "(video)" in k}
    assert recall_video["recall@1 (video)"] == pytest.approx(1.0)
    assert recall_video["recall@5 (video)"] == pytest.approx(1.0)
    assert recall_video["recall@10 (video)"] == pytest.approx(1.0)


def test_frame_level_recall_uses_pk_not_video_path():
    queries = [
        QueryWithExpectedIds(
            query="scene-a",
            relevant_ids=("a.mp4#frame-2",),
            granularity="frame",
        ),
    ]
    fake_client = MagicMock()
    fake_client.search.return_value = [
        [
            _fake_hit("a.mp4#frame-2", "a.mp4", 0.9),  # exact frame
            _fake_hit("a.mp4#frame-0", "a.mp4", 0.7),
        ]
    ]
    with (
        patch("lib.phases.evaluate.MilvusClient", return_value=fake_client),
        patch("lib.phases.evaluate.embed_text_with_clip", return_value=[[0.1] * 512]),
    ):
        out = _evaluate_single_video(label="base", plan=VIDEO_PLAN, queries=queries, concurrency=1)
    frame_keys = {k for k in out["retrieval"] if "(frame)" in k}
    assert "recall@10 (frame)" in frame_keys
    assert out["retrieval"]["recall@10 (frame)"] == pytest.approx(1.0)


def test_mixed_granularity_query_set():
    queries = [
        QueryWithExpectedIds(query="q1", relevant_ids=("f1.jpg",), granularity="frame"),
        QueryWithExpectedIds(query="q2", expected_video_ids=("b.mp4",), granularity="video"),
    ]
    fake_client = MagicMock()

    def _respond(*, data, limit, anns_field, collection_name, output_fields):
        # Return a response whose dominant hit matches the current query by
        # a simple heuristic on the query text — not the most rigorous mock,
        # but keeps this test focused on metric aggregation, not routing.
        return [[_fake_hit("f1.jpg", "a.mp4", 0.9)]]

    fake_client.search.side_effect = _respond
    with (
        patch("lib.phases.evaluate.MilvusClient", return_value=fake_client),
        patch("lib.phases.evaluate.embed_text_with_clip", return_value=[[0.1] * 512]),
    ):
        out = _evaluate_single_video(label="base", plan=VIDEO_PLAN, queries=queries, concurrency=1)
    keys = set(out["retrieval"])
    assert any("(frame)" in k for k in keys)
    assert any("(video)" in k for k in keys)


def test_derive_video_queries_requires_vision_judge(tmp_path: Path):
    (tmp_path / "collect.json").write_text(
        json.dumps(
            {
                "data_shape": "video_dir",
                "rows": [
                    {
                        "video_path": "a.mp4",
                        "frame_path": "a.mp4#frame-0",
                        "t_seconds": 0.0,
                    }
                ],
            }
        )
    )
    with pytest.raises(JudgeUnavailableError):
        _derive_video_queries(out_dir=tmp_path, plan=VIDEO_PLAN, judge=None)
