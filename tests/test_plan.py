"""Golden-file tests for the plan decision tree."""

from __future__ import annotations

import pytest
from lib.phases.plan import plan_from_profile


def _profile(
    *,
    dataset_size: int,
    deployment: str,
    use_case: str = "rag",
    hybrid_pref: str = "auto",
    reranker_pref: str = "auto",
    fields: list[dict] | None = None,
) -> dict:
    return {
        "collect": {
            "data_shape": "jsonl",
            "fields": fields
            or [
                {"name": "id", "type": "string", "avg_length": 5, "sample_value": "x"},
                {"name": "body", "type": "string", "avg_length": 500, "sample_value": "…"},
            ],
            "suggested_primary_key": "id",
            "suggested_text_field": "body",
            "record_count_estimate": dataset_size,
            "source_path": None,
        },
        "configure": {
            "use_case": use_case,
            "query_patterns": ["long-natural-language"],
            "dataset_size": dataset_size,
            "deployment_target": deployment,
            "latency_target_ms": None,
            "embedding_preference": None,
            "hybrid_preference": hybrid_pref,
            "reranker_preference": reranker_pref,
        },
    }


def test_small_local_picks_hnsw_no_quantization():
    plan = plan_from_profile(_profile(dataset_size=5_000, deployment="local-standalone"))
    assert plan.index.type == "HNSW"
    assert plan.index.quantization is None


def test_medium_dataset_picks_hnsw_larger_m():
    plan = plan_from_profile(_profile(dataset_size=500_000, deployment="local-standalone"))
    assert plan.index.type == "HNSW"
    assert plan.index.params["M"] >= 24


def test_large_cloud_picks_diskann():
    plan = plan_from_profile(_profile(dataset_size=50_000_000, deployment="zilliz-serverless"))
    assert plan.index.type == "DISKANN"
    assert plan.index.backend_compatibility == "Cloud-only"


def test_large_local_falls_back_to_hnsw_with_bigger_m():
    plan = plan_from_profile(_profile(dataset_size=50_000_000, deployment="local-standalone"))
    assert plan.index.type == "HNSW"
    assert plan.index.params["M"] == 32


def test_rag_auto_enables_sparse():
    plan = plan_from_profile(
        _profile(dataset_size=100, deployment="local-standalone", use_case="rag")
    )
    assert plan.sparse_enabled is True


def test_recommendations_auto_disables_sparse():
    plan = plan_from_profile(
        _profile(dataset_size=100, deployment="local-standalone", use_case="recommendations")
    )
    assert plan.sparse_enabled is False


def test_explicit_dense_overrides_auto():
    plan = plan_from_profile(
        _profile(dataset_size=100, deployment="local-standalone", hybrid_pref="dense")
    )
    assert plan.sparse_enabled is False


def test_reranker_opt_in_cohere():
    plan = plan_from_profile(
        _profile(dataset_size=100, deployment="local-standalone", reranker_pref="cohere")
    )
    assert plan.reranker == "cohere-rerank-3"


def test_plan_is_deterministic():
    p1 = plan_from_profile(_profile(dataset_size=100, deployment="local-standalone"))
    p2 = plan_from_profile(_profile(dataset_size=100, deployment="local-standalone"))
    assert p1.to_dict() == p2.to_dict()


@pytest.mark.parametrize("deployment", ["zilliz-serverless", "zilliz-dedicated", "zilliz-byoc"])
def test_cloud_targets_produce_https_uri(deployment: str):
    plan = plan_from_profile(_profile(dataset_size=100, deployment=deployment))
    assert plan.target_uri.startswith("https://")


def test_local_produces_http_uri():
    plan = plan_from_profile(_profile(dataset_size=100, deployment="local-standalone"))
    assert plan.target_uri.startswith("http://localhost:")


def test_plan_emits_bulk_import_threshold_default():
    plan = plan_from_profile(_profile(dataset_size=100, deployment="local-standalone"))
    assert plan.bulk_import_threshold == 100_000
    assert plan.to_dict()["bulk_import_threshold"] == 100_000
