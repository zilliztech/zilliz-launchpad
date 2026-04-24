"""Phase 4 Execute — video branch (per-video batched ingest)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from lib.errors import MissingCredentialError
from lib.phases.execute import _group_rows_by_video, _ingest_video_rows, run_execute


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
            if row.get("frame_path") in self.fail_on_paths:
                raise RuntimeError(f"forced upsert failure: {row['frame_path']}")
        self.upserts.append(list(data))


def _make_video_rows(video: str, n: int) -> list[dict[str, Any]]:
    return [
        {
            "video_path": video,
            "t_seconds": i * 2.0,
            "frame_path": f"{video}#frame-{i}",
            "source_index": i,
        }
        for i in range(n)
    ]


def test_group_rows_by_video():
    rows = _make_video_rows("a.mp4", 3) + _make_video_rows("b.mp4", 2)
    groups = _group_rows_by_video(rows)
    assert set(groups) == {"a.mp4", "b.mp4"}
    assert len(groups["a.mp4"]) == 3
    assert len(groups["b.mp4"]) == 2


def test_ingest_video_rows_happy_path(tmp_path: Path):
    rows = _make_video_rows("a.mp4", 3) + _make_video_rows("b.mp4", 2)
    groups = _group_rows_by_video(rows)
    client = _FakeMilvusClient()
    processed: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []
    new, batches = _ingest_video_rows(
        client=client,
        collection="x",
        groups=groups,
        pk_field="frame_path",
        vector_field="embedding",
        embedder_fn=_fake_embedder(),
        out_dir=tmp_path,
        processed_videos=processed,
        skipped_frames=skipped,
        frame_progress=False,
    )
    assert sorted(new) == ["a.mp4", "b.mp4"]
    assert batches == 2
    assert skipped == []
    assert processed["a.mp4"]["frame_count"] == 3
    assert processed["b.mp4"]["frame_count"] == 2
    # Every upserted row carried the deep-link scalars
    for data in client.upserts:
        for row in data:
            assert "video_path" in row and "t_seconds" in row


def test_resume_skips_already_processed_videos(tmp_path: Path):
    rows = _make_video_rows("a.mp4", 2) + _make_video_rows("b.mp4", 2)
    groups = _group_rows_by_video(rows)
    client = _FakeMilvusClient()
    processed = {"a.mp4": {"frame_count": 2, "skipped_frame_count": 0}}
    skipped: list[dict[str, str]] = []
    new, _ = _ingest_video_rows(
        client=client,
        collection="x",
        groups=groups,
        pk_field="frame_path",
        vector_field="embedding",
        embedder_fn=_fake_embedder(),
        out_dir=tmp_path,
        processed_videos=processed,
        skipped_frames=skipped,
        frame_progress=False,
    )
    assert new == ["b.mp4"]
    assert len(client.upserts) == 1  # only b.mp4's single batch


def test_single_frame_failure_skipped(tmp_path: Path):
    rows = _make_video_rows("a.mp4", 3)
    groups = _group_rows_by_video(rows)
    client = _FakeMilvusClient()
    client.fail_on_paths = {"a.mp4#frame-1"}
    processed: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []
    # Put the failing frame in its own batch of 1 by monkeypatching batch size
    with patch("lib.phases.execute.VIDEO_BATCH_SIZE", 1):
        new, _ = _ingest_video_rows(
            client=client,
            collection="x",
            groups=groups,
            pk_field="frame_path",
            vector_field="embedding",
            embedder_fn=_fake_embedder(),
            out_dir=tmp_path,
            processed_videos=processed,
            skipped_frames=skipped,
            frame_progress=False,
        )
    assert new == ["a.mp4"]  # the video still counts as attempted
    skipped_paths = {s["frame_path"] for s in skipped}
    assert "a.mp4#frame-1" in skipped_paths
    assert processed["a.mp4"]["skipped_frame_count"] == 1


def test_per_video_progress_logging(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    caplog.set_level("INFO", logger="lib.phases.execute")
    rows = _make_video_rows("a.mp4", 2)
    groups = _group_rows_by_video(rows)
    client = _FakeMilvusClient()
    _ingest_video_rows(
        client=client,
        collection="x",
        groups=groups,
        pk_field="frame_path",
        vector_field="embedding",
        embedder_fn=_fake_embedder(),
        out_dir=tmp_path,
        processed_videos={},
        skipped_frames=[],
        frame_progress=False,
    )
    assert any("video 1/1" in rec.message for rec in caplog.records), [
        rec.message for rec in caplog.records
    ]


def test_voyage_missing_key_fails_fast(tmp_path: Path):
    """Video plan with Voyage provider but no API key must raise before Milvus."""
    plan = {
        "embedding": {
            "provider": "voyage",
            "model": "voyage-multimodal-3",
            "dim": 1024,
            "modality": "image",
        },
        "schema": {"primary_key": "frame_path", "vector_field": "embedding"},
        "target_uri": "http://localhost:19530",
        "collection_name": "x",
        "index": {"type": "HNSW", "metric": "COSINE", "params": {}},
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "collect.json").write_text(
        json.dumps({"data_shape": "video_dir", "rows": _make_video_rows("a.mp4", 1)})
    )
    with patch.dict("os.environ", {}, clear=False):
        import os

        os.environ.pop("VOYAGE_API_KEY", None)
        with pytest.raises(MissingCredentialError) as exc:
            run_execute(
                out_dir=tmp_path,
                sample=None,
                input_path=None,
                ui_port=8000,
                start_ui=False,
            )
    assert exc.value.payload.get("env_var") == "VOYAGE_API_KEY"
