"""CLI smoke tests for `evaluate --query-image`.

Mocks the Milvus client and CLIP encoder so the test is fast.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner
from zilliz_ops import app


def _write_plan(out_dir: Path, *, image: bool) -> None:
    if image:
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
    else:
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
    (out_dir / "plan.json").write_text(json.dumps(plan))
    # preflight_execute_artifact also checks for execute.json
    (out_dir / "execute.json").write_text(json.dumps({"status": "ok"}))


@pytest.fixture
def image_run(tmp_path: Path) -> Path:
    _write_plan(tmp_path, image=True)
    return tmp_path


@pytest.fixture
def text_run(tmp_path: Path) -> Path:
    _write_plan(tmp_path, image=False)
    return tmp_path


def _fake_hit(pk: str, score: float):
    hit = MagicMock()
    hit.id = pk
    hit.score = score
    hit.entity = {"image_path": pk}
    return hit


def test_query_image_smoke_prints_top_k_and_exits_zero(image_run: Path, tmp_path_factory):
    query_img = tmp_path_factory.mktemp("query") / "q.jpg"
    query_img.write_bytes(b"\xff\xd8\xff\xe0fake")

    fake_client = MagicMock()
    fake_client.search.return_value = [
        [
            _fake_hit("/photos/a.jpg", 0.91),
            _fake_hit("/photos/b.jpg", 0.85),
        ]
    ]
    runner = CliRunner()
    with (
        patch("lib.phases.evaluate.MilvusClient", return_value=fake_client),
        patch("lib.search.embed_image_batch", return_value=[[0.1] * 512]),
    ):
        result = runner.invoke(
            app,
            ["evaluate", "--run-dir", str(image_run), "--query-image", str(query_img)],
        )
    assert result.exit_code == 0, result.stdout + result.stderr
    # Top-2 ranks rendered with both id and score
    assert "/photos/a.jpg" in result.stdout
    assert "/photos/b.jpg" in result.stdout
    assert "score=0.9100" in result.stdout


def test_query_image_and_qrels_are_mutually_exclusive(image_run: Path, tmp_path_factory):
    query_img = tmp_path_factory.mktemp("query") / "q.jpg"
    query_img.write_bytes(b"\xff\xd8\xff\xe0")
    qrels = tmp_path_factory.mktemp("q") / "qrels.jsonl"
    qrels.write_text('{"query": "x", "relevant_ids": ["1"]}\n')

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--run-dir",
            str(image_run),
            "--query-image",
            str(query_img),
            "--qrels",
            str(qrels),
        ],
    )
    assert result.exit_code != 0
    assert "invalid_profile" in (result.stdout + result.stderr)


def test_query_image_on_text_collection_errors(text_run: Path, tmp_path_factory):
    query_img = tmp_path_factory.mktemp("query") / "q.jpg"
    query_img.write_bytes(b"\xff\xd8\xff\xe0")

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["evaluate", "--run-dir", str(text_run), "--query-image", str(query_img)],
    )
    assert result.exit_code != 0
    combined = result.stdout + result.stderr
    assert "invalid_profile" in combined
    assert "image or video collection" in combined


def test_query_image_missing_file_errors(image_run: Path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--run-dir",
            str(image_run),
            "--query-image",
            str(image_run / "no-such-file.jpg"),
        ],
    )
    assert result.exit_code != 0
    assert "not found" in (result.stdout + result.stderr)
