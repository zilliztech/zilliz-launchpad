"""Phase 1 Collect — Markdown branch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from lib.phases.collect import (
    SUPPORTED_SUFFIXES,
    _strip_yaml_frontmatter,
    run_collect,
)

FIXTURE = Path(__file__).parent / "fixtures" / "documents" / "sample.md"


def test_markdown_default_emits_one_record(tmp_path: Path):
    result = run_collect(input_path=str(FIXTURE), sample=None, out_dir=tmp_path)

    assert result["data_shape"] == "markdown"
    assert result["record_count_estimate"] == 1
    assert result["suggested_primary_key"] == "id"
    assert result["suggested_text_field"] == "text"

    field_names = {f["name"] for f in result["fields"]}
    assert field_names == {"id", "text"}

    text_field = next(f for f in result["fields"] if f["name"] == "text")
    assert text_field["sample_value"].startswith("# Sample document")


def test_markdown_split_headings_emits_one_record_per_section(tmp_path: Path):
    result = run_collect(
        input_path=str(FIXTURE),
        sample=None,
        out_dir=tmp_path,
        split_markdown_headings=True,
    )

    assert result["data_shape"] == "markdown"
    assert result["record_count_estimate"] == 3

    field_names = {f["name"] for f in result["fields"]}
    assert {"text", "section_heading"} <= field_names

    written = json.loads((tmp_path / "collect.json").read_text())
    assert written["data_shape"] == "markdown"
    assert written["record_count_estimate"] == 3


def test_yaml_frontmatter_is_stripped(tmp_path: Path):
    result = run_collect(input_path=str(FIXTURE), sample=None, out_dir=tmp_path)
    sample = next(f for f in result["fields"] if f["name"] == "text")["sample_value"]
    assert "title: Sample document" not in sample
    assert "---" not in sample.split("\n")[0]


def test_strip_yaml_frontmatter_unit():
    raw = "---\nkey: value\nother: 1\n---\nbody line\n"
    assert _strip_yaml_frontmatter(raw) == "body line\n"
    # No front-matter → unchanged
    assert _strip_yaml_frontmatter("body only\n") == "body only\n"
    # Unterminated front-matter → unchanged
    assert _strip_yaml_frontmatter("---\nkey: value\n") == "---\nkey: value\n"


def test_md_in_supported_suffixes():
    assert ".md" in SUPPORTED_SUFFIXES


def test_unsupported_suffix_message_lists_md_and_pdf(tmp_path: Path):
    bogus = tmp_path / "data.xyz"
    bogus.write_text("nothing")
    with pytest.raises(ValueError) as exc:
        run_collect(input_path=str(bogus), sample=None, out_dir=tmp_path)
    assert ".md" in str(exc.value)
    assert ".pdf" in str(exc.value)
