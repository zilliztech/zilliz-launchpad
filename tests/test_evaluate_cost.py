"""Report-shape tests for the populated cost/query column (issue #10).

These exercise the pure report-assembly functions without Milvus.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lib import pricing
from lib.evaluator import QueryWithExpectedIds
from lib.phases.evaluate import (
    _build_report,
    _decision_row,
    _render_markdown,
    _RowResult,
)


def _row(label: str, *, cost: dict[str, float] | None) -> dict:
    return _RowResult(
        label=label,
        retrieval={"recall@10": 0.5},
        latency={"p95_ms": 12.0},
        rag=None,
        cost=cost,
    ).to_dict()


def test_rowresult_to_dict_includes_cost():
    d = _RowResult(label="base", cost={"cost_per_query": 0.01}).to_dict()
    assert d["cost"] == {"cost_per_query": 0.01}


def test_rowresult_cost_defaults_to_none():
    assert _RowResult(label="base").to_dict()["cost"] is None


def test_decision_row_renders_cost_when_present():
    line = _decision_row(
        "base", {"recall@10": 0.5}, {"p95_ms": 12.0}, None, {"cost_per_query": 0.06}
    )
    assert "$0.060000" in line


def test_decision_row_renders_dash_when_cost_none():
    line = _decision_row("base", {"recall@10": 0.5}, {"p95_ms": 12.0}, None, None)
    assert line.rstrip().endswith("| — |")


def test_build_report_and_markdown_carry_base_and_variant_cost(tmp_path: Path):
    queries = [QueryWithExpectedIds(query="hi", relevant_ids=("1",))]
    base = _row("base", cost={"cost_per_query": 0.01})
    variant = _row("voyage", cost={"cost_per_query": 0.02})
    variant["overrides"] = {"embedding": {"provider": "voyage"}}

    report = _build_report(
        out_dir=tmp_path,
        queries=queries,
        base_row=base,
        variant_rows=[variant],
        derived=False,
    )

    assert report["cost_metrics"] == {"cost_per_query": 0.01}
    assert report["variants"][0]["cost"] == {"cost_per_query": 0.02}

    md = _render_markdown(report)
    assert "cost/query" in md
    assert "$0.010000" in md
    assert "$0.020000" in md


def test_markdown_renders_dash_for_unpriced_base(tmp_path: Path):
    queries = [QueryWithExpectedIds(query="hi", relevant_ids=("1",))]
    base = _row("base", cost=None)
    report = _build_report(
        out_dir=tmp_path,
        queries=queries,
        base_row=base,
        variant_rows=[],
        derived=False,
    )
    assert report["cost_metrics"] is None
    md = _render_markdown(report)
    # the base decision row's cost cell is the em-dash placeholder
    base_line = next(ln for ln in md.splitlines() if ln.startswith("| base "))
    assert base_line.rstrip().endswith("| — |")


def test_comparison_table_differentiates_providers_by_cost(tmp_path: Path):
    pricing.reset_missing_warnings()
    texts = ["a fairly typical search query about movies"] * 5
    openai_cost = pricing.cost_per_query(
        queries=texts,
        provider="openai",
        model="text-embedding-3-small",
        reranker=None,
        top_k=10,
    )
    voyage_cost = pricing.cost_per_query(
        queries=texts, provider="voyage", model="voyage-3", reranker=None, top_k=10
    )
    assert openai_cost is not None and voyage_cost is not None
    assert openai_cost != voyage_cost

    queries = [QueryWithExpectedIds(query=t, relevant_ids=("1",)) for t in texts]
    base = _row("base", cost={"cost_per_query": openai_cost})
    variant = _row("voyage", cost={"cost_per_query": voyage_cost})
    variant["overrides"] = {"embedding": {"provider": "voyage", "model": "voyage-3"}}
    report = _build_report(
        out_dir=tmp_path,
        queries=queries,
        base_row=base,
        variant_rows=[variant],
        derived=False,
    )
    md = _render_markdown(report)
    base_cell = (
        next(ln for ln in md.splitlines() if ln.startswith("| base ")).rsplit("|", 2)[1].strip()
    )
    voyage_cell = (
        next(ln for ln in md.splitlines() if ln.startswith("| voyage ")).rsplit("|", 2)[1].strip()
    )
    assert base_cell.startswith("$") and voyage_cell.startswith("$")
    assert base_cell != voyage_cell


def test_unpriced_variant_dashes_and_warns_once(caplog):
    pricing.reset_missing_warnings()
    with caplog.at_level(logging.WARNING, logger="lib.pricing"):
        c1 = pricing.cost_per_query(
            queries=["x"], provider="acme", model="unknown-emb", reranker=None, top_k=10
        )
        c2 = pricing.cost_per_query(
            queries=["y"], provider="acme", model="unknown-emb", reranker=None, top_k=10
        )
    assert c1 is None and c2 is None
    line = _decision_row("acme-variant", {"recall@10": 0.1}, {"p95_ms": 1.0}, None, None)
    assert line.rstrip().endswith("| — |")
    warnings = [r for r in caplog.records if "unknown-emb" in r.getMessage()]
    assert len(warnings) == 1
