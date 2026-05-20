"""Unit tests for the shared --input resolver."""

from __future__ import annotations

from pathlib import Path

import pytest
from lib.errors import EmptyInputSetError
from lib.inputs import has_glob_chars, resolve_inputs

SUPPORTED = frozenset({".jsonl", ".csv", ".txt", ".md", ".pdf"})


def test_single_file_returns_one_absolute_path(tmp_path: Path):
    f = tmp_path / "a.jsonl"
    f.write_text('{"id":1}\n')
    result = resolve_inputs(str(f), supported_suffixes=SUPPORTED)
    assert result == [f.resolve()]
    assert result[0].is_absolute()


def test_single_file_with_unsupported_suffix_passes_through(tmp_path: Path):
    """Single-file dispatch defers suffix validation to the caller."""
    f = tmp_path / "x.xyz"
    f.write_text("x")
    result = resolve_inputs(str(f), supported_suffixes=SUPPORTED)
    assert result == [f.resolve()]


def test_directory_walks_recursively_and_filters_suffix(tmp_path: Path):
    (tmp_path / "a.jsonl").write_text("{}\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "skip.bin").write_bytes(b"\x00")
    result = resolve_inputs(str(tmp_path), supported_suffixes=SUPPORTED)
    names = [p.name for p in result]
    assert names == sorted(names)
    assert "a.jsonl" in names
    assert "b.pdf" in names
    assert "skip.bin" not in names


def test_empty_directory_raises_empty_input(tmp_path: Path):
    (tmp_path / "x.bin").write_bytes(b"\x00")
    with pytest.raises(EmptyInputSetError) as exc:
        resolve_inputs(str(tmp_path), supported_suffixes=SUPPORTED)
    assert exc.value.code == "empty_input"
    assert exc.value.payload["raw"] == str(tmp_path)


def test_glob_matches_and_sorts(tmp_path: Path):
    for name in ("c.pdf", "a.pdf", "b.pdf"):
        (tmp_path / name).write_bytes(b"%PDF")
    pattern = str(tmp_path / "*.pdf")
    result = resolve_inputs(pattern, supported_suffixes=SUPPORTED)
    assert [p.name for p in result] == ["a.pdf", "b.pdf", "c.pdf"]


def test_glob_with_no_matches_raises_empty_input(tmp_path: Path):
    with pytest.raises(EmptyInputSetError) as exc:
        resolve_inputs(str(tmp_path / "*.pdf"), supported_suffixes=SUPPORTED)
    assert exc.value.code == "empty_input"


def test_nonexistent_literal_path_hints_at_glob_quoting(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as exc:
        resolve_inputs(str(tmp_path / "nope.pdf"), supported_suffixes=SUPPORTED)
    assert "quote" in str(exc.value).lower()


def test_has_glob_chars():
    assert has_glob_chars("docs/*.pdf")
    assert has_glob_chars("docs/?.pdf")
    assert has_glob_chars("docs/[ab].pdf")
    assert not has_glob_chars("docs/a.pdf")
    assert not has_glob_chars("/absolute/path/file.pdf")
