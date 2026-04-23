"""Variant-parser + cap-enforcement tests for comparison mode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from lib.errors import InvalidProfileError
from lib.phases.evaluate import _apply_variant, _parse_variants_file, _resolve_variants


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "variants.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_parse_accepts_json_when_yaml_missing(tmp_path: Path):
    # JSON is a YAML subset, so this also passes the yaml-loading path
    path = _write(
        tmp_path,
        json.dumps(
            {
                "variants": [
                    {"name": "no-hybrid", "overrides": {"hybrid": False}},
                    {"name": "big-m", "overrides": {"index": {"params": {"M": 32}}}},
                ]
            }
        ),
    )
    variants = _parse_variants_file(path)
    assert [v["name"] for v in variants] == ["no-hybrid", "big-m"]


def test_parse_rejects_missing_top_level_list(tmp_path: Path):
    path = _write(tmp_path, json.dumps({"not_variants": []}))
    with pytest.raises(InvalidProfileError, match="top-level `variants:`"):
        _parse_variants_file(path)


def test_parse_rejects_empty_list(tmp_path: Path):
    path = _write(tmp_path, json.dumps({"variants": []}))
    with pytest.raises(InvalidProfileError, match="at least one entry"):
        _parse_variants_file(path)


def test_parse_rejects_duplicate_names(tmp_path: Path):
    path = _write(
        tmp_path,
        json.dumps(
            {
                "variants": [
                    {"name": "same", "overrides": {}},
                    {"name": "same", "overrides": {}},
                ]
            }
        ),
    )
    with pytest.raises(InvalidProfileError, match="duplicate variant name"):
        _parse_variants_file(path)


def test_parse_rejects_unknown_override_axis(tmp_path: Path):
    path = _write(
        tmp_path,
        json.dumps({"variants": [{"name": "x", "overrides": {"sharding": 4}}]}),
    )
    with pytest.raises(InvalidProfileError, match="unsupported override axes"):
        _parse_variants_file(path)


def test_parse_rejects_unsupported_nested_key(tmp_path: Path):
    path = _write(
        tmp_path,
        json.dumps({"variants": [{"name": "x", "overrides": {"embedding": {"temperature": 0.5}}}]}),
    )
    with pytest.raises(InvalidProfileError, match="unsupported nested keys"):
        _parse_variants_file(path)


def test_parse_requires_string_name(tmp_path: Path):
    path = _write(tmp_path, json.dumps({"variants": [{"name": "", "overrides": {}}]}))
    with pytest.raises(InvalidProfileError, match="non-empty string"):
        _parse_variants_file(path)


def test_resolve_variants_enforces_cap(tmp_path: Path):
    path = _write(
        tmp_path,
        json.dumps(
            {"variants": [{"name": f"v{i}", "overrides": {"hybrid": False}} for i in range(7)]}
        ),
    )
    with pytest.raises(InvalidProfileError, match="exceeds cap"):
        _resolve_variants(compare_path=str(path), allow_large=False)


def test_resolve_variants_allows_large_override(tmp_path: Path):
    path = _write(
        tmp_path,
        json.dumps(
            {"variants": [{"name": f"v{i}", "overrides": {"hybrid": False}} for i in range(7)]}
        ),
    )
    variants = _resolve_variants(compare_path=str(path), allow_large=True)
    assert len(variants) == 7


def test_resolve_variants_returns_empty_when_no_path():
    assert _resolve_variants(compare_path=None, allow_large=False) == []


def test_resolve_variants_errors_on_missing_file(tmp_path: Path):
    with pytest.raises(InvalidProfileError, match="variants file not found"):
        _resolve_variants(compare_path=str(tmp_path / "absent.yaml"), allow_large=False)


# --- _apply_variant --------------------------------------------------------


def _plan() -> dict:
    return {
        "embedding": {"provider": "openai", "model": "text-embedding-3-small", "dim": 1536},
        "index": {"type": "HNSW", "params": {"M": 16, "efConstruction": 200}},
        "sparse_enabled": True,
        "reranker": "cohere-rerank-3",
        "target_uri": "http://localhost:19530",
    }


def test_apply_variant_merges_embedding_model_only():
    variant = {"name": "voyage", "overrides": {"embedding": {"model": "voyage-3"}}}
    merged = _apply_variant(_plan(), variant)
    assert merged["embedding"]["model"] == "voyage-3"
    # Other embedding fields must survive the merge
    assert merged["embedding"]["provider"] == "openai"
    assert merged["embedding"]["dim"] == 1536


def test_apply_variant_merges_index_params_deeply():
    variant = {"name": "big-m", "overrides": {"index": {"params": {"M": 32}}}}
    merged = _apply_variant(_plan(), variant)
    assert merged["index"]["params"]["M"] == 32
    # efConstruction must survive
    assert merged["index"]["params"]["efConstruction"] == 200
    assert merged["index"]["type"] == "HNSW"


def test_apply_variant_toggles_hybrid():
    merged = _apply_variant(_plan(), {"name": "no-hybrid", "overrides": {"hybrid": False}})
    assert merged["sparse_enabled"] is False


def test_apply_variant_can_disable_reranker():
    merged = _apply_variant(_plan(), {"name": "no-rerank", "overrides": {"reranker": None}})
    assert merged["reranker"] is None


def test_apply_variant_leaves_base_plan_untouched():
    base = _plan()
    _apply_variant(base, {"name": "x", "overrides": {"hybrid": False}})
    assert base["sparse_enabled"] is True
