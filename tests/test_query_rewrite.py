"""Unit tests for lib/query_rewrite.py — no real LLM, no network."""

from __future__ import annotations

import pytest
from lib import query_rewrite
from lib.errors import JudgeUnavailableError


def test_rewrite_returns_cleaned_model_output(monkeypatch):
    monkeypatch.setattr(query_rewrite, "resolve", lambda *a, **k: "secret")
    monkeypatch.setattr(
        query_rewrite, "_rewrite_openai", lambda model, original: '  "best sci-fi films"  '
    )
    out = query_rewrite.rewrite_query("openai", "gpt-4o-mini", "The Matrix is a 1999 film.")
    assert out == "best sci-fi films"  # stripped of quotes/whitespace


def test_empty_model_output_falls_back_to_original(monkeypatch):
    monkeypatch.setattr(query_rewrite, "resolve", lambda *a, **k: "secret")
    monkeypatch.setattr(query_rewrite, "_rewrite_openai", lambda model, original: "   ")
    original = "The Matrix is a 1999 film."
    assert query_rewrite.rewrite_query("openai", "gpt-4o-mini", original) == original


def test_missing_credential_raises_judge_unavailable(monkeypatch):
    monkeypatch.setattr(query_rewrite, "resolve", lambda *a, **k: None)
    called = False

    def _boom(model, original):  # pragma: no cover - must not run
        nonlocal called
        called = True
        return "x"

    monkeypatch.setattr(query_rewrite, "_rewrite_openai", _boom)
    with pytest.raises(JudgeUnavailableError) as info:
        query_rewrite.rewrite_query("openai", "gpt-4o-mini", "q")
    assert info.value.payload["env_var"] == "OPENAI_API_KEY"
    assert called is False  # gated before any network call


def test_unsupported_provider_raises_judge_unavailable(monkeypatch):
    monkeypatch.setattr(query_rewrite, "resolve", lambda *a, **k: "secret")
    with pytest.raises(JudgeUnavailableError):
        query_rewrite.rewrite_query("cohere", "command-r", "q")


def test_rewrite_queries_batches(monkeypatch):
    monkeypatch.setattr(query_rewrite, "resolve", lambda *a, **k: "secret")
    monkeypatch.setattr(query_rewrite, "_rewrite_openai", lambda model, original: f"rw:{original}")
    out = query_rewrite.rewrite_queries("openai", "gpt-4o-mini", ["a", "b"])
    assert out == ["rw:a", "rw:b"]
