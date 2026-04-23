"""Phase 3 Plan — image-search routing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from lib.phases.plan import (
    DEFAULT_IMAGE_EMBEDDING,
    VOYAGE_MULTIMODAL_EMBEDDING,
    plan_from_profile,
    run_plan,
)


def _image_profile(**configure_overrides: object) -> dict[str, object]:
    collect = {
        "data_shape": "image_dir",
        "fields": [
            {"name": "image_path", "type": "string"},
            {"name": "width", "type": "int"},
            {"name": "height", "type": "int"},
            {"name": "bytes", "type": "int"},
        ],
        "suggested_primary_key": "image_path",
        "suggested_text_field": None,
        "record_count_estimate": 20,
    }
    configure = {
        "use_case": "image-search",
        "query_patterns": ["long-natural-language"],
        "dataset_size": 20,
        "deployment_target": "local-standalone",
        "hybrid_preference": False,
        "reranker_preference": "none",
    }
    configure.update(configure_overrides)
    return {"collect": collect, "configure": configure}


def test_default_image_plan_picks_clip_local():
    plan = plan_from_profile(_image_profile()).to_dict()
    assert plan["embedding"]["provider"] == DEFAULT_IMAGE_EMBEDDING["provider"]
    assert plan["embedding"]["model"] == DEFAULT_IMAGE_EMBEDDING["model"]
    assert plan["embedding"]["dim"] == DEFAULT_IMAGE_EMBEDDING["dim"]
    assert plan["embedding"]["modality"] == "image"
    assert "device_hint" in plan["embedding"]
    assert plan["sparse_enabled"] is False
    assert plan["schema"]["sparse_field"] is None
    assert plan["reranker"] is None


def test_voyage_multimodal_override_flows_through():
    plan = plan_from_profile(
        _image_profile(embedding_preference="voyage-multimodal-3")
    ).to_dict()
    assert plan["embedding"]["provider"] == VOYAGE_MULTIMODAL_EMBEDDING["provider"]
    assert plan["embedding"]["model"] == "voyage-multimodal-3"
    assert plan["embedding"]["dim"] == 1024


def test_image_schema_has_no_text_field():
    plan = plan_from_profile(_image_profile()).to_dict()
    assert plan["schema"]["text_field"] is None
    assert plan["schema"]["primary_key"] == "image_path"
    extras = {f["name"] for f in plan["schema"]["extra_fields"]}
    assert {"width", "height", "bytes"} <= extras


def test_image_plan_md_contains_required_lines(tmp_path: Path):
    """End-to-end: write collect + configure, run_plan, inspect plan.md."""
    profile = _image_profile()
    (tmp_path / "collect.json").write_text(json.dumps(profile["collect"]))
    (tmp_path / "configure.json").write_text(json.dumps(profile["configure"]))
    run_plan(out_dir=tmp_path)
    md = (tmp_path / "plan.md").read_text()
    assert "Modality: `image`" in md
    assert "Device hint:" in md
    assert "Sparse field: disabled (image collection)" in md
    assert "Text field: (none — image collection)" in md
    assert "ViT-B-32" in md


def test_text_path_unchanged():
    """Plan still works for the text-flow movies sample shape."""
    profile = {
        "collect": {
            "data_shape": "jsonl",
            "fields": [
                {"name": "id", "type": "string"},
                {"name": "body", "type": "string", "avg_length": 200},
                {"name": "year", "type": "int"},
            ],
            "suggested_primary_key": "id",
            "suggested_text_field": "body",
            "record_count_estimate": 20,
        },
        "configure": {
            "use_case": "rag",
            "query_patterns": ["long-natural-language"],
            "dataset_size": 20,
            "deployment_target": "local-standalone",
            "hybrid_preference": "auto",
            "reranker_preference": "auto",
        },
    }
    plan = plan_from_profile(profile).to_dict()
    assert plan["embedding"]["provider"] == "openai"
    assert plan["embedding"]["modality"] == "text"
    assert plan["sparse_enabled"] is True
    assert plan["schema"]["text_field"] == "body"


def test_image_plan_dict_provider_override():
    """Dict-form embedding_preference also routes correctly."""
    profile = _image_profile(
        embedding_preference={"provider": "voyage", "model": "voyage-multimodal-3", "dim": 1024}
    )
    plan = plan_from_profile(profile).to_dict()
    assert plan["embedding"]["provider"] == "voyage"
    assert plan["embedding"]["dim"] == 1024
    assert plan["embedding"]["modality"] == "image"


def test_device_hint_is_one_of_known_values(monkeypatch: pytest.MonkeyPatch):
    plan = plan_from_profile(_image_profile()).to_dict()
    assert plan["embedding"]["device_hint"] in {"cpu", "mps", "cuda"}
