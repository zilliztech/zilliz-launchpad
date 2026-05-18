"""Unit tests for the per-query cost estimator (lib.pricing).

Prices are read from the knowledge markdown tables, so these assertions
track the advisory numbers seeded in:
  - references/knowledge/dense_embedding_models.md
  - references/knowledge/reranker_guide.md
"""

from __future__ import annotations

import logging

import pytest
from lib import pricing


def test_estimate_tokens_is_ceil_of_len_over_4():
    assert pricing.estimate_tokens("") == 0
    assert pricing.estimate_tokens("a") == 1
    assert pricing.estimate_tokens("abcd") == 1
    assert pricing.estimate_tokens("abcde") == 2
    assert pricing.estimate_tokens("a" * 8) == 2


def test_embedding_price_lookup_hit():
    assert pricing.embedding_price_per_1m("openai", "text-embedding-3-small") == 0.02
    assert pricing.embedding_price_per_1m("voyage", "voyage-3") == 0.06
    # provider casing must not matter
    assert pricing.embedding_price_per_1m("OpenAI", "text-embedding-3-small") == 0.02


def test_embedding_price_lookup_miss_returns_none():
    assert pricing.embedding_price_per_1m("openai", "no-such-model") is None
    assert pricing.embedding_price_per_1m("acme", "whatever") is None


def test_embedding_price_byom_wildcard_row():
    # The `zilliz-byom | (user-chosen)` row is a provider-level wildcard
    # priced 0 (self-hosted, no per-call API spend).
    assert pricing.embedding_price_per_1m("zilliz-byom", "anything-at-all") == 0.0


def test_reranker_price_lookup():
    assert pricing.reranker_price_per_1k("cohere-rerank-3") == 2.0
    assert pricing.reranker_price_per_1k("bge-reranker-v2-m3") == 0.0
    assert pricing.reranker_price_per_1k("mystery-reranker") is None


def test_cost_per_query_embed_only():
    # estimate_tokens: "abcd"->1, "abcdefgh"->2  => mean 1.5 tokens/query
    cost = pricing.cost_per_query(
        queries=["abcd", "abcdefgh"],
        provider="openai",
        model="text-embedding-3-small",
        reranker=None,
        top_k=10,
    )
    assert cost == pytest.approx(1.5 / 1e6 * 0.02)


def test_cost_per_query_with_reranker_adds_search_cost():
    # rerank fan-out = top_k * 3 = 30 candidates => 30/1000 * 2.0 = 0.06
    cost = pricing.cost_per_query(
        queries=["abcd", "abcdefgh"],
        provider="openai",
        model="text-embedding-3-small",
        reranker="cohere-rerank-3",
        top_k=10,
    )
    assert cost == pytest.approx(1.5 / 1e6 * 0.02 + 30 / 1000 * 2.0)


def test_cost_per_query_self_hosted_reranker_adds_nothing():
    embed_only = pricing.cost_per_query(
        queries=["hello world"],
        provider="voyage",
        model="voyage-3",
        reranker=None,
        top_k=10,
    )
    with_bge = pricing.cost_per_query(
        queries=["hello world"],
        provider="voyage",
        model="voyage-3",
        reranker="bge-reranker-v2-m3",
        top_k=10,
    )
    assert embed_only == with_bge


def test_cost_per_query_unpriced_embedding_returns_none():
    assert (
        pricing.cost_per_query(
            queries=["abcd"],
            provider="openai",
            model="ghost-model",
            reranker=None,
            top_k=10,
        )
        is None
    )


def test_cost_per_query_unpriced_reranker_returns_none():
    assert (
        pricing.cost_per_query(
            queries=["abcd"],
            provider="openai",
            model="text-embedding-3-small",
            reranker="ghost-reranker",
            top_k=10,
        )
        is None
    )


def test_unknown_model_warns_exactly_once(caplog):
    pricing.reset_missing_warnings()
    with caplog.at_level(logging.WARNING, logger="lib.pricing"):
        pricing.cost_per_query(
            queries=["abcd"], provider="openai", model="ghost-x", reranker=None, top_k=10
        )
        pricing.cost_per_query(
            queries=["efgh"], provider="openai", model="ghost-x", reranker=None, top_k=10
        )
    matching = [r for r in caplog.records if "ghost-x" in r.getMessage()]
    assert len(matching) == 1
