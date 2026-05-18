"""Tests for the opt-in judge rewrite of text-derived queries.

Exercises `_rewrite_derived_queries` (cache + rewrite) and the
`_resolve_query_set` contract directly — no Milvus, no real LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

from lib.evaluator import JudgeConfig, QueryWithExpectedIds
from lib.phases import evaluate as ev


def _base() -> list[QueryWithExpectedIds]:
    return [
        QueryWithExpectedIds(query="The Matrix is a 1999 film.", relevant_ids=("m1",)),
        QueryWithExpectedIds(query="Inception bends dreams.", relevant_ids=("m2",)),
    ]


def _judge() -> JudgeConfig:
    return JudgeConfig(provider="openai", model="gpt-4o-mini")


def test_rewrite_applied_ids_preserved(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "lib.query_rewrite.rewrite_query",
        lambda provider, model, original: f"rw::{original}",
    )
    out = ev._rewrite_derived_queries(_base(), out_dir=tmp_path, judge=_judge())

    assert [q.query for q in out] == [
        "rw::The Matrix is a 1999 film.",
        "rw::Inception bends dreams.",
    ]
    # ground-truth ids untouched
    assert [q.relevant_ids for q in out] == [("m1",), ("m2",)]
    # cache file written, one line per doc id
    cache = (tmp_path / ev.DERIVED_QUERIES_FILE).read_text(encoding="utf-8").splitlines()
    recs = [json.loads(line) for line in cache]
    assert {r["doc_id"] for r in recs} == {"m1", "m2"}
    assert recs[0]["original"] == "The Matrix is a 1999 film."


def test_cache_hit_on_rerun_zero_calls(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def _rw(provider, model, original):
        calls.append(original)
        return f"rw::{original}"

    monkeypatch.setattr("lib.query_rewrite.rewrite_query", _rw)

    ev._rewrite_derived_queries(_base(), out_dir=tmp_path, judge=_judge())
    assert len(calls) == 2  # first run rewrites both

    calls.clear()
    out = ev._rewrite_derived_queries(_base(), out_dir=tmp_path, judge=_judge())
    assert calls == []  # second run is a pure cache hit
    assert [q.query for q in out] == [
        "rw::The Matrix is a 1999 film.",
        "rw::Inception bends dreams.",
    ]


def test_partial_cache_only_calls_missing(tmp_path: Path, monkeypatch):
    # Pre-seed the cache with just m1.
    (tmp_path / ev.DERIVED_QUERIES_FILE).write_text(
        json.dumps({"doc_id": "m1", "original": "x", "rewritten": "cached-m1"}) + "\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    def _rw(provider, model, original):
        calls.append(original)
        return f"rw::{original}"

    monkeypatch.setattr("lib.query_rewrite.rewrite_query", _rw)
    out = ev._rewrite_derived_queries(_base(), out_dir=tmp_path, judge=_judge())

    assert calls == ["Inception bends dreams."]  # only the uncached doc
    assert out[0].query == "cached-m1"  # reused from cache
    assert out[1].query == "rw::Inception bends dreams."


def test_no_judge_is_verbatim_and_writes_no_cache(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        ev,
        "_iter_documents",
        lambda **kw: iter([{"id": "m1", "text": "The Matrix is a 1999 film. It stars Keanu."}]),
    )
    plan = {"schema": {"text_field": "text", "primary_key": "id"}, "embedding": {}}

    out = ev._derive_queries(out_dir=tmp_path, plan=plan, judge=None)

    assert out[0].query == "The Matrix is a 1999 film."  # verbatim first sentence
    assert out[0].relevant_ids == ("m1",)
    assert not (tmp_path / ev.DERIVED_QUERIES_FILE).exists()  # no cache file


def test_resolve_query_set_derived_flag_true_with_judge(tmp_path: Path, monkeypatch):
    seen: dict[str, object] = {}

    def _fake_derive(*, out_dir, plan, judge=None):
        seen["judge"] = judge
        return [QueryWithExpectedIds(query="q", relevant_ids=("d1",))]

    monkeypatch.setattr(ev, "_derive_queries", _fake_derive)
    plan = {"schema": {}, "embedding": {}}
    judge = _judge()

    queries, derived = ev._resolve_query_set(
        out_dir=tmp_path, plan=plan, qrels_path=None, queries_path=None, judge=judge
    )
    assert derived is True
    assert seen["judge"] is judge  # judge threaded into the text-derived path
