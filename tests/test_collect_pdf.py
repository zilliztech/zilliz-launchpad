"""Phase 1 Collect — PDF branch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from lib.errors import MissingDependencyError
from lib.phases.collect import SUPPORTED_SUFFIXES, run_collect

FIXTURES = Path(__file__).parent / "fixtures" / "documents"
SAMPLE_PDF = FIXTURES / "sample.pdf"
SCANNED_PDF = FIXTURES / "scanned.pdf"


@pytest.mark.documents
def test_pdf_happy_path_emits_one_record_per_page(tmp_path: Path):
    result = run_collect(input_path=str(SAMPLE_PDF), sample=None, out_dir=tmp_path)

    assert result["data_shape"] == "pdf"
    assert result["record_count_estimate"] == 3
    assert result["suggested_primary_key"] == "id"
    assert result["suggested_text_field"] == "text"

    field_names = {f["name"] for f in result["fields"]}
    assert {"id", "text", "page_number", "source_path"} <= field_names

    page_number_field = next(f for f in result["fields"] if f["name"] == "page_number")
    assert page_number_field["type"] == "int"
    assert page_number_field["sample_value"] == 1

    assert "warnings" not in result


@pytest.mark.documents
def test_pdf_with_no_extractable_text_emits_warning(tmp_path: Path):
    result = run_collect(input_path=str(SCANNED_PDF), sample=None, out_dir=tmp_path)

    assert result["data_shape"] == "pdf"
    assert "warnings" in result
    assert any("scanned" in w.lower() for w in result["warnings"])


def test_pdf_in_supported_suffixes():
    assert ".pdf" in SUPPORTED_SUFFIXES


def test_missing_pypdf_raises_missing_dependency_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Force the lazy `from pypdf import PdfReader` inside _read_pdf to fail.
    monkeypatch.setitem(sys.modules, "pypdf", None)

    with pytest.raises(MissingDependencyError) as exc:
        run_collect(input_path=str(SAMPLE_PDF), sample=None, out_dir=tmp_path)

    assert exc.value.payload["feature"] == "pdf-collect"
    assert exc.value.payload["install_hint"] == "pip install zilliz-launchpad[documents]"
    envelope = exc.value.to_dict()
    assert envelope["code"] == "missing_dependency"
