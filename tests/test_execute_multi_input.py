"""Phase 4 Execute — multi-file streaming + resume (unit-level, no Milvus)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from lib.errors import EmptyInputSetError


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@dataclass
class _FakeStats:
    documents: int
    chunks: int
    batches: int
    retries: int


def _build_plan() -> dict[str, Any]:
    return {
        "collection_name": "test_coll",
        "schema": {
            "primary_key": "id",
            "text_field": "text",
            "vector_field": "embedding",
            "extra_fields": [],
            "dim": 4,
        },
        "chunking": {"size": 256, "overlap": 0},
        "embedding": {"provider": "fake", "model": "fake", "dim": 4},
        "target_uri": "http://fake",
    }


def test_resolve_execute_inputs_filters_to_jsonl_only(tmp_path: Path):
    (tmp_path / "a.jsonl").write_text("{}\n")
    (tmp_path / "b.pdf").write_bytes(b"%PDF")
    from lib.phases.execute import _resolve_execute_inputs

    with pytest.raises(EmptyInputSetError) as exc:
        # The .pdf alone should resolve empty (filtered out).
        _resolve_execute_inputs(str(tmp_path / "*.pdf"))
    assert exc.value.code == "empty_input"

    # Mixed dir is OK — only JSONL members come through.
    result = _resolve_execute_inputs(str(tmp_path))
    assert [p.name for p in result] == ["a.jsonl"]


def test_empty_resolved_input_set(tmp_path: Path):
    from lib.phases.execute import _resolve_execute_inputs

    (tmp_path / "z.bin").write_bytes(b"\x00")
    with pytest.raises(EmptyInputSetError):
        _resolve_execute_inputs(str(tmp_path))


def test_execute_input_files_single_file_falls_to_legacy_path(tmp_path: Path):
    f = tmp_path / "a.jsonl"
    f.write_text("{}\n")
    from lib.phases.execute import _execute_input_files

    assert _execute_input_files(str(f), tmp_path) is None  # single file -> legacy
    assert _execute_input_files(None, tmp_path) is None


def test_execute_input_files_directory_returns_list(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    _write_jsonl(docs / "a.jsonl", [{"id": "1", "text": "x"}])
    _write_jsonl(docs / "b.jsonl", [{"id": "2", "text": "y"}])
    from lib.phases.execute import _execute_input_files

    result = _execute_input_files(str(docs), tmp_path)
    assert result is not None
    assert [p.name for p in result] == ["a.jsonl", "b.jsonl"]


def test_iter_collect_sources_prefers_source_files():
    from lib.phases.execute import iter_collect_sources

    assert iter_collect_sources({"source_files": [{"path": "/a"}, {"path": "/b"}]}) == [
        Path("/a"),
        Path("/b"),
    ]
    assert iter_collect_sources({"source_path": "/x"}) == [Path("/x")]
    assert iter_collect_sources({}) == []


def test_load_prior_execute_snapshot(tmp_path: Path):
    from lib.phases.execute import _load_prior_execute_snapshot

    # No prior snapshot
    processed, warnings = _load_prior_execute_snapshot(tmp_path)
    assert processed == set() and warnings == []

    (tmp_path / "execute.json").write_text(
        json.dumps(
            {
                "phase": "ingesting",
                "processed_files": ["/a", "/b"],
                "warnings": ["stale: /c"],
            }
        )
    )
    processed, warnings = _load_prior_execute_snapshot(tmp_path)
    assert processed == {"/a", "/b"}
    assert warnings == ["stale: /c"]


def test_multi_file_run_ingests_all_and_snapshots(tmp_path: Path, monkeypatch):
    """_run_text_execute_multi_file ingests each file in order and snapshots progress."""
    from lib.phases import execute as ex

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    _write_jsonl(docs_dir / "a.jsonl", [{"id": "1", "text": "alpha"}])
    _write_jsonl(docs_dir / "b.jsonl", [{"id": "2", "text": "beta"}])
    _write_jsonl(docs_dir / "c.jsonl", [{"id": "3", "text": "gamma"}])

    ingested: list[str] = []

    def fake_ingest(
        client,
        collection,
        docs,
        embedder,
        *,
        text_field,
        id_field,
        vector_field,
        chunk_config,
        extra_field_keys,
    ):
        ingested.extend(d["id"] for d in docs)
        return _FakeStats(documents=len(docs), chunks=len(docs), batches=1, retries=0)

    monkeypatch.setattr(ex, "ingest_documents", fake_ingest)
    monkeypatch.setattr(ex, "search_dense", lambda *a, **kw: [])

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    files = sorted((docs_dir).glob("*.jsonl"))
    report = ex._run_text_execute_multi_file(
        client=object(),
        plan=_build_plan(),
        files=files,
        out_dir=out_dir,
        embedder=object(),
        chunk_config=object(),
        extra_keys=(),
        coll_status={},
        idx_status={},
        ui_port=0,
        start_ui=False,
        preflight=None,
        cluster_id=None,
    )

    assert ingested == ["1", "2", "3"]
    assert report["phase"] == "complete"
    assert report["ingest"]["documents"] == 3
    assert sorted(report["processed_files"]) == sorted(str(f.resolve()) for f in files)


def test_resume_skips_already_processed_files(tmp_path: Path, monkeypatch):
    from lib.phases import execute as ex

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    _write_jsonl(docs_dir / "a.jsonl", [{"id": "1", "text": "alpha"}])
    _write_jsonl(docs_dir / "b.jsonl", [{"id": "2", "text": "beta"}])
    _write_jsonl(docs_dir / "c.jsonl", [{"id": "3", "text": "gamma"}])
    files = sorted((docs_dir).glob("*.jsonl"))

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    # Simulate a prior partial run that already ingested file a.
    (out_dir / "execute.json").write_text(
        json.dumps(
            {
                "phase": "ingesting",
                "processed_files": [str(files[0].resolve())],
            }
        )
    )

    ingested: list[str] = []

    def fake_ingest(client, collection, docs, embedder, **kw):
        ingested.extend(d["id"] for d in docs)
        return _FakeStats(documents=len(docs), chunks=len(docs), batches=1, retries=0)

    monkeypatch.setattr(ex, "ingest_documents", fake_ingest)
    monkeypatch.setattr(ex, "search_dense", lambda *a, **kw: [])

    report = ex._run_text_execute_multi_file(
        client=object(),
        plan=_build_plan(),
        files=files,
        out_dir=out_dir,
        embedder=object(),
        chunk_config=object(),
        extra_keys=(),
        coll_status={},
        idx_status={},
        ui_port=0,
        start_ui=False,
        preflight=None,
        cluster_id=None,
    )
    # a.jsonl was already processed; only b and c should be ingested this run.
    assert ingested == ["2", "3"]
    assert sorted(report["processed_files"]) == sorted(str(f.resolve()) for f in files)


def test_stale_processed_files_entry_emits_warning(tmp_path: Path, monkeypatch):
    from lib.phases import execute as ex

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    _write_jsonl(docs_dir / "b.jsonl", [{"id": "2", "text": "beta"}])
    files = sorted((docs_dir).glob("*.jsonl"))

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    missing_path = "/nonexistent/old/a.jsonl"
    (out_dir / "execute.json").write_text(
        json.dumps({"phase": "ingesting", "processed_files": [missing_path]})
    )

    monkeypatch.setattr(
        ex,
        "ingest_documents",
        lambda *a, **kw: _FakeStats(documents=1, chunks=1, batches=1, retries=0),
    )
    monkeypatch.setattr(ex, "search_dense", lambda *a, **kw: [])

    report = ex._run_text_execute_multi_file(
        client=object(),
        plan=_build_plan(),
        files=files,
        out_dir=out_dir,
        embedder=object(),
        chunk_config=object(),
        extra_keys=(),
        coll_status={},
        idx_status={},
        ui_port=0,
        start_ui=False,
        preflight=None,
        cluster_id=None,
    )
    assert any(missing_path in w for w in report.get("warnings", []))
    assert missing_path in report["processed_files"]  # preserved
