"""Unit and integration coverage for the `execute --append` flow."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _write_plan(run_dir: Path, dim: int = 4) -> None:
    plan = {
        "target_uri": "http://localhost:19530",
        "collection_name": "movies_append_test",
        "schema": {
            "primary_key": "id",
            "text_field": "text",
            "vector_field": "embedding",
            "dim": dim,
            "extra_fields": [],
            "sparse_field": None,
        },
        "embedding": {"provider": "openai", "model": "text-embedding-3-small", "dim": dim},
        "index": {"type": "HNSW", "metric": "COSINE", "params": {"M": 16, "efConstruction": 128}},
        "chunking": {"size": 512, "overlap": 64},
        "sparse_enabled": False,
    }
    (run_dir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")


def test_invalid_profile_when_plan_missing(tmp_path: Path):
    from lib.errors import InvalidProfileError
    from lib.phases.execute import run_execute_append

    (tmp_path / "new.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(InvalidProfileError) as exc:
        run_execute_append(out_dir=tmp_path, input_path=str(tmp_path / "new.jsonl"))
    assert "plan.json" in exc.value.message


def test_next_append_artifact_path_numbering(tmp_path: Path):
    from lib.phases.execute import _next_append_artifact_path

    assert _next_append_artifact_path(tmp_path) == tmp_path / "execute_append.json"
    (tmp_path / "execute_append.json").write_text("{}", encoding="utf-8")
    assert _next_append_artifact_path(tmp_path) == tmp_path / "execute_append.2.json"
    (tmp_path / "execute_append.2.json").write_text("{}", encoding="utf-8")
    assert _next_append_artifact_path(tmp_path) == tmp_path / "execute_append.3.json"


def test_schema_conflict_on_dim_mismatch(tmp_path: Path, monkeypatch):
    """Live collection schema differs from plan → SchemaConflictError with field detail."""
    from lib import phases
    from lib.errors import SchemaConflictError
    from lib.phases.execute import run_execute_append

    _write_plan(tmp_path, dim=8)  # plan says dim=8
    (tmp_path / "new.jsonl").write_text("", encoding="utf-8")

    class FakeClient:
        def list_collections(self):
            return ["movies_append_test"]

        def describe_collection(self, name):
            # Live schema has dim=4 — mismatch.
            return {
                "fields": [
                    {"name": "id", "type": "DataType.VARCHAR", "params": {"max_length": 128}},
                    {"name": "text", "type": "DataType.VARCHAR", "params": {"max_length": 65535}},
                    {"name": "embedding", "type": "DataType.FLOAT_VECTOR", "params": {"dim": 4}},
                ]
            }

    monkeypatch.setattr(phases.execute, "MilvusClient", lambda uri: FakeClient())
    with pytest.raises(SchemaConflictError) as exc:
        run_execute_append(out_dir=tmp_path, input_path=str(tmp_path / "new.jsonl"))
    assert any("dim" in m for m in exc.value.payload["mismatches"])
    # original execute.json must remain absent (we never wrote it)
    assert not (tmp_path / "execute.json").exists()


def test_schema_conflict_when_collection_missing(tmp_path: Path, monkeypatch):
    from lib import phases
    from lib.errors import SchemaConflictError
    from lib.phases.execute import run_execute_append

    _write_plan(tmp_path)
    (tmp_path / "new.jsonl").write_text("", encoding="utf-8")

    class FakeClient:
        def list_collections(self):
            return []

    monkeypatch.setattr(phases.execute, "MilvusClient", lambda uri: FakeClient())
    with pytest.raises(SchemaConflictError) as exc:
        run_execute_append(out_dir=tmp_path, input_path=str(tmp_path / "new.jsonl"))
    assert "does not exist" in " ".join(exc.value.payload["mismatches"])


@pytest.mark.e2e
@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None,
    reason="OPENAI_API_KEY is required for real embedding calls",
)
def test_append_end_to_end(tmp_path: Path):
    """Base ingest 20 rows → append 5 more → row count is 25 and original execute.json untouched."""
    from lib.phases.collect import run_collect
    from lib.phases.configure import run_configure
    from lib.phases.execute import run_execute, run_execute_append
    from lib.phases.plan import run_plan

    run_collect(input_path=None, sample="movies", out_dir=tmp_path)
    run_configure(
        from_json=None,
        out_dir=tmp_path,
        overrides={"dataset_size": 20, "deployment_target": "local-standalone"},
    )
    run_plan(out_dir=tmp_path)
    base = run_execute(
        out_dir=tmp_path,
        sample="movies",
        input_path=None,
        ui_port=8011,
        start_ui=False,
    )
    assert base["ingest"]["documents"] == 20

    # Snapshot bytes to assert immutability.
    execute_json = tmp_path / "execute.json"
    before = execute_json.read_bytes()

    # Build a 5-row append input from the original sample's last 5 entries.
    from lib.samples import load as load_sample

    rows = list(load_sample("movies"))[-5:]
    append_input = tmp_path / "docs_v2.jsonl"
    append_input.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    report = run_execute_append(out_dir=tmp_path, input_path=str(append_input))
    assert report["appended_rows"] == 5
    assert (tmp_path / "execute_append.json").exists()
    # Original execute.json byte-for-byte unchanged.
    assert execute_json.read_bytes() == before
