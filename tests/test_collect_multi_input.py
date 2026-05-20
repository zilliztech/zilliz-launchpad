"""Phase 1 Collect — multi-file (directory / glob) input."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from lib.errors import EmptyInputSetError, InputSchemaConflictError
from lib.phases.collect import run_collect

FIXTURES = Path(__file__).parent / "fixtures" / "documents"
SAMPLE_PDF = FIXTURES / "sample.pdf"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.mark.documents
def test_mixed_jsonl_and_pdf_directory(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    _write_jsonl(docs / "a.jsonl", [{"id": "1", "text": "hello world"}])
    (docs / "b.pdf").write_bytes(SAMPLE_PDF.read_bytes())

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = run_collect(input_path=str(docs), sample=None, out_dir=out_dir)

    assert "source_files" in result
    assert len(result["source_files"]) == 2
    paths = [sf["path"] for sf in result["source_files"]]
    assert paths == sorted(paths)
    assert result["data_shape"] == "mixed"
    assert "source_path" not in result
    field_names = {f["name"] for f in result["fields"]}
    assert {"id", "text", "page_number"} <= field_names


def test_union_schema_nullable_for_partial_fields(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    _write_jsonl(docs / "a.jsonl", [{"id": "1", "text": "x", "author": "alice"}])
    _write_jsonl(docs / "b.jsonl", [{"id": "2", "text": "y"}])

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = run_collect(input_path=str(docs), sample=None, out_dir=out_dir)

    author = next(f for f in result["fields"] if f["name"] == "author")
    assert author.get("nullable") is True
    text = next(f for f in result["fields"] if f["name"] == "text")
    assert text.get("nullable") is not True  # present in all files


def test_type_conflict_raises_input_schema_conflict(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    _write_jsonl(docs / "a.jsonl", [{"id": 1, "text": "x"}])  # id: int
    _write_jsonl(docs / "b.jsonl", [{"id": "2", "text": "y"}])  # id: string

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(InputSchemaConflictError) as exc:
        run_collect(input_path=str(docs), sample=None, out_dir=out_dir)
    assert exc.value.code == "input_schema_conflict"
    assert exc.value.payload["field"] == "id"
    files = exc.value.payload["files"]
    types = sorted(ft["type"] for ft in files)
    assert types == ["int", "string"]
    assert not (out_dir / "collect.json").exists()


def test_empty_directory_raises_empty_input(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "skip.bin").write_bytes(b"\x00")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(EmptyInputSetError) as exc:
        run_collect(input_path=str(docs), sample=None, out_dir=out_dir)
    assert exc.value.code == "empty_input"


@pytest.mark.documents
def test_glob_matches_multiple_pdfs_sorted(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    pdf_bytes = SAMPLE_PDF.read_bytes()
    for name in ("c.pdf", "a.pdf", "b.pdf"):
        (docs / name).write_bytes(pdf_bytes)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pattern = str(docs / "*.pdf")
    result = run_collect(input_path=pattern, sample=None, out_dir=out_dir)
    assert result["data_shape"] == "pdf"
    names = [Path(sf["path"]).name for sf in result["source_files"]]
    assert names == ["a.pdf", "b.pdf", "c.pdf"]


def test_uniform_jsonl_directory_keeps_jsonl_shape(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    _write_jsonl(docs / "a.jsonl", [{"id": "1", "text": "x"}])
    _write_jsonl(docs / "b.jsonl", [{"id": "2", "text": "y"}])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = run_collect(input_path=str(docs), sample=None, out_dir=out_dir)
    assert result["data_shape"] == "jsonl"
    assert len(result["source_files"]) == 2
