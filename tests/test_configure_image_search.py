"""Phase 2 Configure — image-search use case."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from lib.errors import InvalidProfileError
from lib.phases.configure import run_configure


def _write_collect(out_dir: Path, data_shape: str) -> None:
    (out_dir / "collect.json").write_text(json.dumps({"data_shape": data_shape}))


def test_image_search_forces_hybrid_off_and_reranker_none(tmp_path: Path):
    _write_collect(tmp_path, "image_dir")
    data = run_configure(
        from_json=None,
        out_dir=tmp_path,
        overrides={"use_case": "image-search", "deployment_target": "local-standalone"},
    )
    assert data["use_case"] == "image-search"
    assert data["hybrid_preference"] is False
    assert data["reranker_preference"] == "none"


def test_image_search_warns_on_conflicting_hybrid_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    _write_collect(tmp_path, "image_dir")
    run_configure(
        from_json=None,
        out_dir=tmp_path,
        overrides={
            "use_case": "image-search",
            "deployment_target": "local-standalone",
        },
    )
    # The default "auto" is silently overridden — no warning.
    err1 = capsys.readouterr().err
    assert "warn:" not in err1

    # An explicit conflicting value should warn.
    profile = json.loads((tmp_path / "configure.json").read_text())
    profile["hybrid_preference"] = "on"
    profile["reranker_preference"] = "cohere-rerank-3"
    profile_path = tmp_path / "in.json"
    profile_path.write_text(json.dumps(profile))
    run_configure(from_json=str(profile_path), out_dir=tmp_path)
    err2 = capsys.readouterr().err
    assert "hybrid_preference" in err2
    assert "reranker_preference" in err2


def test_modality_mismatch_raises_invalid_profile(tmp_path: Path):
    _write_collect(tmp_path, "image_dir")
    with pytest.raises(InvalidProfileError) as exc:
        run_configure(
            from_json=None,
            out_dir=tmp_path,
            overrides={"use_case": "rag", "deployment_target": "local-standalone"},
        )
    assert "image_dir" in exc.value.message
    assert "image-search" in exc.value.message


def test_image_search_against_text_collect_raises(tmp_path: Path):
    _write_collect(tmp_path, "jsonl")
    with pytest.raises(InvalidProfileError) as exc:
        run_configure(
            from_json=None,
            out_dir=tmp_path,
            overrides={"use_case": "image-search", "deployment_target": "local-standalone"},
        )
    assert "image-search" in exc.value.message


def test_unknown_use_case_raises(tmp_path: Path):
    with pytest.raises(InvalidProfileError) as exc:
        run_configure(
            from_json=None,
            out_dir=tmp_path,
            overrides={"use_case": "telepathy", "deployment_target": "local-standalone"},
        )
    assert "telepathy" in exc.value.message


def test_no_collect_json_skips_modality_check(tmp_path: Path):
    """Configure can run before Collect; modality check is best-effort."""
    data = run_configure(
        from_json=None,
        out_dir=tmp_path,
        overrides={"use_case": "image-search", "deployment_target": "local-standalone"},
    )
    assert data["use_case"] == "image-search"
