"""Phase 5 Evaluate — image branch.

Covers:
- vision-judge allow-list
- image-qrels parsing
- derived-eval requires a vision judge (fails clean otherwise)
- caption cache reuse
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from lib.errors import InvalidProfileError, JudgeUnavailableError
from lib.evaluator import JudgeConfig
from lib.phases.evaluate import (
    DERIVED_IMAGE_QUERIES_FILE,
    _derive_image_queries,
    _load_image_qrels,
    _resolve_query_set,
)
from lib.vision_judge import is_vision_capable, require_vision

FIXTURE_PHOTOS = Path(__file__).parent / "fixtures" / "photos"


# --- Allow-list ------------------------------------------------------------


@pytest.mark.parametrize(
    "provider,model,expected",
    [
        ("openai", "gpt-4o", True),
        ("openai", "gpt-4o-mini", True),
        ("openai", "gpt-5", True),
        ("openai", "gpt-5-pro", True),
        ("anthropic", "claude-3-5-sonnet-20241022", True),
        ("anthropic", "claude-opus-4-7", True),
        ("openai", "gpt-3.5-turbo", False),
        ("openai", "text-embedding-3-small", False),
        ("cohere", "command-r-plus", False),
        ("anthropic", "haiku-1", False),
    ],
)
def test_vision_allow_list(provider: str, model: str, expected: bool):
    assert is_vision_capable(provider, model) is expected


def test_require_vision_rejects_non_vision():
    with pytest.raises(JudgeUnavailableError) as exc:
        require_vision("openai", "gpt-3.5-turbo")
    assert exc.value.payload["provider"] == "openai:gpt-3.5-turbo"


# --- Image qrels parsing ---------------------------------------------------


def test_image_qrels_loads_query_text_image_path(tmp_path: Path):
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(
        "\n".join(
            [
                json.dumps({"query_text": "sunset", "image_path": "a.jpg"}),
                json.dumps({"query_text": "cat", "image_paths": ["c1.jpg", "c2.jpg"]}),
            ]
        )
    )
    rows = _load_image_qrels(qrels)
    assert len(rows) == 2
    assert rows[0].query == "sunset"
    assert rows[0].relevant_ids == ("a.jpg",)
    assert rows[1].relevant_ids == ("c1.jpg", "c2.jpg")


def test_image_qrels_rejects_missing_query(tmp_path: Path):
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(json.dumps({"image_path": "a.jpg"}))
    with pytest.raises(InvalidProfileError):
        _load_image_qrels(qrels)


def test_image_qrels_rejects_missing_path(tmp_path: Path):
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(json.dumps({"query_text": "sunset"}))
    with pytest.raises(InvalidProfileError):
        _load_image_qrels(qrels)


# --- Derived eval gating ---------------------------------------------------


def _image_plan() -> dict[str, object]:
    return {
        "embedding": {"provider": "clip-local", "model": "ViT-B-32", "modality": "image"},
        "schema": {"primary_key": "image_path", "vector_field": "embedding"},
    }


def _image_collect(paths: list[Path]) -> dict[str, object]:
    return {
        "data_shape": "image_dir",
        "rows": [{"image_path": str(p)} for p in paths],
    }


def test_derive_image_queries_without_judge_raises(tmp_path: Path):
    (tmp_path / "collect.json").write_text(
        json.dumps(_image_collect(sorted(FIXTURE_PHOTOS.glob("*.jpg"))[:3]))
    )
    with pytest.raises(JudgeUnavailableError):
        _derive_image_queries(out_dir=tmp_path, plan=_image_plan(), judge=None)


def test_derive_image_queries_with_non_vision_judge_raises(tmp_path: Path):
    (tmp_path / "collect.json").write_text(
        json.dumps(_image_collect(sorted(FIXTURE_PHOTOS.glob("*.jpg"))[:3]))
    )
    judge = JudgeConfig(provider="openai", model="text-embedding-3-small")
    with pytest.raises(JudgeUnavailableError) as exc:
        _derive_image_queries(out_dir=tmp_path, plan=_image_plan(), judge=judge)
    assert "openai:text-embedding-3-small" in exc.value.payload["provider"]


def test_resolve_query_set_image_no_qrels_no_vision_judge_raises(tmp_path: Path):
    """The integration point that the spec calls out."""
    (tmp_path / "collect.json").write_text(
        json.dumps(_image_collect(sorted(FIXTURE_PHOTOS.glob("*.jpg"))[:3]))
    )
    with pytest.raises(JudgeUnavailableError):
        _resolve_query_set(
            out_dir=tmp_path,
            plan=_image_plan(),
            qrels_path=None,
            queries_path=None,
            judge=None,
        )


# --- Caption cache reuse ---------------------------------------------------


def test_caption_cache_skips_already_captioned_paths(tmp_path: Path):
    paths = sorted(FIXTURE_PHOTOS.glob("*.jpg"))[:5]
    (tmp_path / "collect.json").write_text(json.dumps(_image_collect(paths)))

    judge = JudgeConfig(provider="openai", model="gpt-4o-mini")
    call_count = {"n": 0}

    def fake_caption(provider: str, model: str, ps):
        call_count["n"] += len(list(ps))
        return [f"caption-{Path(p).stem}" for p in ps]

    with patch("lib.phases.evaluate.caption_images", side_effect=fake_caption):
        first = _derive_image_queries(out_dir=tmp_path, plan=_image_plan(), judge=judge)
    assert len(first) == 5
    assert call_count["n"] == 5

    # Cache file written
    cache = tmp_path / DERIVED_IMAGE_QUERIES_FILE
    assert cache.exists()
    cached_rows = [json.loads(line) for line in cache.read_text().splitlines() if line.strip()]
    assert len(cached_rows) == 5

    # Re-run with the same plan + collect → no new captions called
    call_count["n"] = 0
    with patch("lib.phases.evaluate.caption_images", side_effect=fake_caption):
        second = _derive_image_queries(out_dir=tmp_path, plan=_image_plan(), judge=judge)
    assert len(second) == 5
    assert call_count["n"] == 0  # cache hit
    assert {q.query for q in second} == {q.query for q in first}
