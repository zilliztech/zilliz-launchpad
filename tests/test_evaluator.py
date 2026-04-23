"""Pure-math tests for lib/evaluator.py — no Milvus, no LLM."""

from __future__ import annotations

import time

import pytest
from lib.errors import JudgeUnavailableError
from lib.evaluator import (
    JudgeConfig,
    QueryWithExpectedIds,
    compute_latency,
    compute_rag_quality,
    compute_retrieval_metrics,
    derive_queries_from_corpus,
)

# --- Retrieval metrics -----------------------------------------------------


def _q(query: str, *ids: str, grade: int = 1) -> QueryWithExpectedIds:
    return QueryWithExpectedIds(query=query, relevant_ids=tuple(ids), grade=grade)


def test_recall_perfect_when_all_relevant_in_topk():
    qrels = [_q("dogs", "d1", "d2")]
    hits = [["d1", "d2", "x3"]]
    out = compute_retrieval_metrics(qrels, hits, k=3)
    assert out["recall@3"] == pytest.approx(1.0)
    assert out["MRR@10"] == pytest.approx(1.0)  # first hit is relevant


def test_recall_partial_when_only_some_relevant():
    qrels = [_q("dogs", "d1", "d2", "d3")]
    hits = [["d1", "x2", "d2"]]
    out = compute_retrieval_metrics(qrels, hits, k=3)
    assert out["recall@3"] == pytest.approx(2 / 3)


def test_mrr_picks_rank_of_first_relevant():
    qrels = [_q("dogs", "d9")]
    hits = [["x1", "x2", "d9"]]
    out = compute_retrieval_metrics(qrels, hits, k=10)
    assert out["MRR@10"] == pytest.approx(1 / 3)


def test_ndcg_zero_when_no_relevant_hit_appears():
    qrels = [_q("dogs", "d9")]
    hits = [["x1", "x2", "x3"]]
    out = compute_retrieval_metrics(qrels, hits, k=10)
    assert out["NDCG@10"] == 0.0
    assert out["recall@10"] == 0.0


def test_ndcg_uses_grade_weight():
    # Single relevant doc at rank 1 with grade=3 and grade=1 should produce
    # the same NDCG (both normalised by their own IDCG), so NDCG is stable.
    out1 = compute_retrieval_metrics([_q("dogs", "d9", grade=1)], [["d9"]], k=10)
    out3 = compute_retrieval_metrics([_q("dogs", "d9", grade=3)], [["d9"]], k=10)
    assert out1["NDCG@10"] == pytest.approx(out3["NDCG@10"])


def test_empty_qrels_returns_empty_dict():
    # A query with no labels should be dropped; if all are dropped, return {}
    out = compute_retrieval_metrics([_q("dogs")], [["x1", "x2"]], k=10)
    assert out == {}


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="qrels/hits length mismatch"):
        compute_retrieval_metrics([_q("a", "1")], [], k=10)


def test_query_count_reflects_only_labelled_queries():
    qrels = [_q("a", "x1"), _q("b"), _q("c", "y1")]
    hits = [["x1"], ["ignored"], ["y1"]]
    out = compute_retrieval_metrics(qrels, hits, k=10)
    assert out["query_count"] == 2


# --- Latency ---------------------------------------------------------------


def test_latency_percentiles_single_threaded():
    # Fixed synthetic latencies via a fake search that sleeps a known amount
    delays_ms = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    def _search(q: str) -> None:
        idx = int(q)
        time.sleep(delays_ms[idx] / 1000.0)

    queries = [str(i) for i in range(len(delays_ms))]
    out = compute_latency(_search, queries, concurrency=1)
    # Nearest-rank percentile: p50 of 10 samples = index 5 (1-based) = 50ms
    assert out["count"] == 10
    assert out["p50_ms"] >= 40  # generous lower bounds for CI jitter
    assert out["p95_ms"] >= 80
    assert out["p99_ms"] >= 90


def test_latency_empty_queries_returns_zeros():
    out = compute_latency(lambda q: None, [], concurrency=1)
    assert out == {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "count": 0}


def test_latency_concurrent_still_records_all_samples():
    def _search(q: str) -> None:
        time.sleep(0.005)

    out = compute_latency(_search, [str(i) for i in range(20)], concurrency=8)
    assert out["count"] == 20
    assert out["p50_ms"] > 0


# --- RAG quality -----------------------------------------------------------


def test_judge_config_parses_provider_and_model():
    cfg = JudgeConfig.parse("openai:gpt-4o-mini")
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o-mini"
    assert cfg.env_var == "OPENAI_API_KEY"


def test_judge_config_rejects_bad_format():
    with pytest.raises(ValueError, match="'<provider>:<model>'"):
        JudgeConfig.parse("openai-gpt-4o")


def test_judge_config_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported judge provider"):
        JudgeConfig.parse("grok:xai-1")


def test_rag_quality_raises_when_credential_missing():
    cfg = JudgeConfig(provider="openai", model="gpt-4o-mini")
    with pytest.raises(JudgeUnavailableError) as info:
        compute_rag_quality(
            ["What is X?"],
            [["ctx"]],
            ["ans"],
            cfg,
            credential_resolver=lambda _: None,
        )
    assert info.value.payload["env_var"] == "OPENAI_API_KEY"


def test_rag_quality_returns_none_for_empty_queries():
    cfg = JudgeConfig(provider="openai", model="gpt-4o-mini")
    assert compute_rag_quality([], [], [], cfg, credential_resolver=lambda _: "secret") is None


# --- Derived query set -----------------------------------------------------


def test_derive_pulls_first_sentence_and_doc_id():
    docs = [
        {"id": "d1", "text": "A small dog barks. More text ignored."},
        {"id": "d2", "text": "Cats purr when happy. Also other things."},
    ]
    queries = derive_queries_from_corpus(docs, text_field="text", id_field="id", sample_size=5)
    assert [q.query for q in queries] == [
        "A small dog barks.",
        "Cats purr when happy.",
    ]
    assert queries[0].relevant_ids == ("d1",)
    assert queries[1].relevant_ids == ("d2",)


def test_derive_respects_sample_size():
    docs = [{"id": f"d{i}", "text": f"Sentence {i}."} for i in range(50)]
    queries = derive_queries_from_corpus(docs, text_field="text", id_field="id", sample_size=3)
    assert len(queries) == 3


def test_derive_skips_empty_text():
    docs = [
        {"id": "d1", "text": ""},
        {"id": "d2", "text": "Valid content here."},
    ]
    queries = derive_queries_from_corpus(docs, text_field="text", id_field="id", sample_size=5)
    assert len(queries) == 1
    assert queries[0].relevant_ids == ("d2",)


def test_derive_stringifies_numeric_ids():
    docs = [{"id": 42, "text": "Whatever."}]
    queries = derive_queries_from_corpus(docs, text_field="text", id_field="id", sample_size=5)
    assert queries[0].relevant_ids == ("42",)
