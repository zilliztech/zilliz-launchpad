"""Phase 5 metric math.

Pure functions — no Milvus, no LLM calls at import time. The evaluate
driver threads a `search_fn` into `compute_latency` so this module stays
testable against canned data.

Metric families:
  * retrieval (needs qrels): recall@k, MRR@10, NDCG@10
  * latency:                 p50, p95, p99 under a configurable concurrency
  * RAG quality (opt-in):    faithfulness, answer_relevance via ragas
"""

from __future__ import annotations

import math
import re
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from .errors import JudgeUnavailableError

# --- Types -----------------------------------------------------------------


@dataclass(frozen=True)
class QueryWithExpectedIds:
    """Single query plus the ids that are considered relevant.

    `relevant_ids` may be an empty tuple when the query has no labels — that
    still flows through `compute_latency` but is dropped from retrieval math.
    `grade` is an optional graded-relevance value (NDCG uses it; recall/MRR
    treat any appearance as relevant).

    When `query_image_path` is set, the evaluator routes this row through the
    image→image search path: `query` carries a display label (e.g. the query
    filename) and the real payload is the bytes at the named path.
    """

    query: str
    relevant_ids: tuple[str, ...] = ()
    grade: int = 1
    query_image_path: str | None = None


# --- Retrieval metrics ------------------------------------------------------


def compute_retrieval_metrics(
    qrels: Sequence[QueryWithExpectedIds],
    hits: Sequence[Sequence[str]],
    *,
    k: int = 10,
) -> dict[str, float]:
    """Compute recall@k, MRR@10, NDCG@10 over aligned qrels/hits.

    `hits[i]` is the ranked list of hit ids for `qrels[i]`. Queries without
    any `relevant_ids` are skipped (their rank contribution is zero).

    Returns empty dict when no qrels carry labels so the caller can decide
    whether to omit the section from the report.
    """
    if len(qrels) != len(hits):
        raise ValueError(f"qrels/hits length mismatch: {len(qrels)} qrels vs {len(hits)} hit lists")

    labelled = [(q, h) for q, h in zip(qrels, hits, strict=True) if q.relevant_ids]
    if not labelled:
        return {}

    recalls: list[float] = []
    rrs: list[float] = []
    ndcgs: list[float] = []
    for q, ranked in labelled:
        topk = list(ranked[:k])
        relevant = set(q.relevant_ids)
        recalls.append(sum(1 for h in topk if h in relevant) / len(relevant))
        rrs.append(_reciprocal_rank(topk[:10], relevant))
        ndcgs.append(_ndcg(topk[:10], relevant, grade=q.grade))

    return {
        f"recall@{k}": statistics.mean(recalls),
        "MRR@10": statistics.mean(rrs),
        "NDCG@10": statistics.mean(ndcgs),
        "query_count": float(len(labelled)),
    }


def _reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, hid in enumerate(ranked, start=1):
        if hid in relevant:
            return 1.0 / i
    return 0.0


def _ndcg(ranked: list[str], relevant: set[str], *, grade: int) -> float:
    dcg = 0.0
    for i, hid in enumerate(ranked, start=1):
        if hid in relevant:
            dcg += grade / math.log2(i + 1)
    ideal_hits = min(len(relevant), len(ranked))
    idcg = sum(grade / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


# --- Latency ---------------------------------------------------------------


def compute_latency(
    search_fn: Callable[[str], Any],
    queries: Sequence[str],
    *,
    concurrency: int = 1,
) -> dict[str, float]:
    """Measure p50/p95/p99 wall-clock latency of `search_fn` over `queries`.

    The caller owns what `search_fn` does — it might call `client.search(...)`
    or a higher-level wrapper. We only time it.

    Latencies are reported in milliseconds. Results include `count` so the
    report can flag suspiciously small samples.
    """
    if not queries:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "count": 0}

    samples: list[float] = []
    if concurrency <= 1:
        for q in queries:
            samples.append(_time_one(search_fn, q))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_time_one, search_fn, q) for q in queries]
            for fut in as_completed(futures):
                samples.append(fut.result())

    return {
        "p50_ms": _percentile(samples, 50),
        "p95_ms": _percentile(samples, 95),
        "p99_ms": _percentile(samples, 99),
        "count": len(samples),
    }


def _time_one(search_fn: Callable[[str], Any], query: str) -> float:
    start = time.perf_counter()
    search_fn(query)
    return (time.perf_counter() - start) * 1000.0


def _percentile(samples: list[float], pct: int) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    # Nearest-rank method — matches what most eng teams mean by "p95"
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


# --- RAG quality (opt-in, ragas) -------------------------------------------


# Map of judge-LLM provider → env var that must be set for ragas to work.
# Kept here (not imported from embeddings.py) because the ragas judge is a
# generator-model credential, not the embedder credential.
_JUDGE_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "cohere": "COHERE_API_KEY",
}


@dataclass(frozen=True)
class JudgeConfig:
    provider: str
    model: str

    @classmethod
    def parse(cls, spec: str) -> JudgeConfig:
        """Accept the `<provider>:<model>` CLI format."""
        if ":" not in spec:
            raise ValueError(f"--judge-llm must be '<provider>:<model>' (got {spec!r})")
        provider, model = spec.split(":", 1)
        provider = provider.strip().lower()
        model = model.strip()
        if provider not in _JUDGE_ENV:
            raise ValueError(
                f"Unsupported judge provider {provider!r}; known: {', '.join(sorted(_JUDGE_ENV))}"
            )
        if not model:
            raise ValueError("--judge-llm model segment is empty")
        return cls(provider=provider, model=model)

    @property
    def env_var(self) -> str:
        return _JUDGE_ENV[self.provider]


def compute_rag_quality(
    queries: Sequence[str],
    contexts: Sequence[Sequence[str]],
    answers: Sequence[str],
    judge: JudgeConfig,
    *,
    credential_resolver: Callable[[str], str | None] | None = None,
) -> dict[str, float] | None:
    """Compute faithfulness + answer_relevance via ragas.

    Returns None when there's nothing to score (empty query list). Raises
    `JudgeUnavailableError` when the judge provider's env var is unset;
    the Phase 5 driver surfaces that as a CLI error envelope.

    `credential_resolver` lets tests inject a stub without touching the
    real env. Default resolves through `lib.credentials`.
    """
    if not queries:
        return None

    if credential_resolver is None:
        from .credentials import resolve as _resolve

        def credential_resolver(name: str) -> str | None:
            return _resolve(name, optional=True, allow_cli=False)

    token = credential_resolver(judge.env_var)
    if not token:
        raise JudgeUnavailableError(provider=judge.provider, env_var=judge.env_var)

    # Import lazily — ragas is a heavy dep we only want to touch when used
    from ragas import evaluate  # type: ignore[import-not-found]
    from ragas.metrics import answer_relevance, faithfulness  # type: ignore[import-not-found]

    dataset = [
        {"question": q, "contexts": list(c), "answer": a}
        for q, c, a in zip(queries, contexts, answers, strict=True)
    ]
    result = evaluate(dataset, metrics=[faithfulness, answer_relevance])

    # ragas returns a DatasetDict-like object with per-metric means
    out: dict[str, float] = {}
    for name in ("faithfulness", "answer_relevance"):
        value = result.get(name) if hasattr(result, "get") else None
        if value is not None:
            out[name] = float(value)
    return out


# --- Derived query set -----------------------------------------------------


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def derive_queries_from_corpus(
    documents: Iterable[dict[str, Any]],
    *,
    text_field: str,
    id_field: str,
    sample_size: int = 25,
) -> list[QueryWithExpectedIds]:
    """Sample K docs, take the first sentence as a smoke query.

    The expected id for each query is the source doc's id. This is a
    *smoke* eval, not a real retrieval eval — the report must mark
    `derived: true` so the user knows.
    """
    out: list[QueryWithExpectedIds] = []
    for doc in documents:
        if len(out) >= sample_size:
            break
        raw_text = doc.get(text_field)
        raw_id = doc.get(id_field)
        if not raw_text or raw_id is None:
            continue
        text = str(raw_text).strip()
        if not text:
            continue
        first_sentence = _SENTENCE_SPLIT.split(text, maxsplit=1)[0].strip()
        if not first_sentence:
            continue
        out.append(
            QueryWithExpectedIds(
                query=first_sentence[:512],  # cap to keep queries reasonable
                relevant_ids=(str(raw_id),),
            )
        )
    return out
