"""Per-query cost estimation for the Phase 5 decision table.

Prices live in the user-facing knowledge markdown so adding or repricing a
model is a one-file edit (issue #10):
  - references/knowledge/dense_embedding_models.md  (`USD / 1M tokens`)
  - references/knowledge/reranker_guide.md           (`USD / 1k searches`)

Token counts are estimated with the project's character heuristic
(`len(text) / 4`, matching chunking.py) rather than per-SDK usage — uniform
across providers, dependency-free, and precise enough to rank variants.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "references" / "knowledge"
_EMBEDDING_FILE = _KNOWLEDGE_DIR / "dense_embedding_models.md"
_RERANKER_FILE = _KNOWLEDGE_DIR / "reranker_guide.md"

_CHARS_PER_TOKEN = 4  # matches chunking.py's approximation

# Deduplicates the "model not in price table" warning so many variants /
# repeated runs warn once per unknown model rather than spamming the log.
_WARNED: set[str] = set()


def reset_missing_warnings() -> None:
    """Clear the warn-once dedup set (used by tests)."""
    _WARNED.clear()


def _warn_missing(what: str) -> None:
    if what in _WARNED:
        return
    _WARNED.add(what)
    logger.warning(
        "%s is not in the price table; cost/query will show '—'. "
        "Add a row to the knowledge markdown to price it.",
        what,
    )


def estimate_tokens(text: str) -> int:
    """Approximate input tokens as ceil(len(text) / 4)."""
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


def _split_row(line: str) -> list[str]:
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def _is_separator(cells: Sequence[str]) -> bool:
    return all(set(c) <= {"-", ":", " "} and c for c in cells)


def _backtick_code(cell: str) -> str | None:
    """Return the code inside the first backtick pair, or None if unquoted.

    An unquoted model cell (e.g. `(user-chosen)`) marks a provider-level
    wildcard row.
    """
    m = re.search(r"`([^`]+)`", cell)
    return m.group(1) if m else None


def _parse_price_table(
    path: Path, *, key_header: str, price_header: str
) -> tuple[dict[tuple[str, str], float], dict[str, float], dict[str, float]]:
    """Parse one markdown price table.

    Returns (exact, provider_wildcard, by_code):
      - exact:    {(provider_lower, code): price}
      - wildcard: {provider_lower: price}  (rows with an unquoted key cell)
      - by_code:  {code: price}            (single-key tables, e.g. rerankers)
    """
    exact: dict[tuple[str, str], float] = {}
    wildcard: dict[str, float] = {}
    by_code: dict[str, float] = {}
    if not path.exists():
        return exact, wildcard, by_code

    header: list[str] | None = None
    key_idx = price_idx = provider_idx = -1
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip().startswith("|"):
            continue
        cells = _split_row(raw)
        if header is None:
            if any(key_header in c for c in cells) and any(price_header in c for c in cells):
                header = cells
                key_idx = next(i for i, c in enumerate(cells) if key_header in c)
                price_idx = next(i for i, c in enumerate(cells) if price_header in c)
                provider_idx = next((i for i, c in enumerate(cells) if c.lower() == "provider"), -1)
            continue
        if _is_separator(cells):
            continue
        if max(key_idx, price_idx, provider_idx) >= len(cells):
            continue
        try:
            price = float(cells[price_idx])
        except ValueError:
            continue
        code = _backtick_code(cells[key_idx])
        if provider_idx >= 0:
            provider = cells[provider_idx].lower()
            if code is None:
                wildcard[provider] = price
            else:
                exact[(provider, code)] = price
        elif code is not None:
            by_code[code] = price
    return exact, wildcard, by_code


@lru_cache(maxsize=1)
def _embedding_prices() -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    exact, wildcard, _ = _parse_price_table(
        _EMBEDDING_FILE, key_header="Recommended model", price_header="USD / 1M tokens"
    )
    return exact, wildcard


@lru_cache(maxsize=1)
def _reranker_prices() -> dict[str, float]:
    _, _, by_code = _parse_price_table(
        _RERANKER_FILE, key_header="Adapter id", price_header="USD / 1k searches"
    )
    return by_code


def embedding_price_per_1m(provider: str, model: str) -> float | None:
    """USD per 1M input tokens for provider/model, or None if unpriced."""
    exact, wildcard = _embedding_prices()
    key = (provider.lower(), model)
    if key in exact:
        return exact[key]
    return wildcard.get(provider.lower())


def reranker_price_per_1k(adapter: str) -> float | None:
    """USD per 1k rerank searches for adapter, or None if unpriced."""
    return _reranker_prices().get(adapter)


def cost_per_query(
    *,
    queries: Sequence[str],
    provider: str,
    model: str,
    reranker: str | None,
    top_k: int,
) -> float | None:
    """Estimated USD cost of one query for this resolved plan.

    Embedding cost uses mean estimated tokens across the query set; reranker
    cost uses the `top_k * 3` candidate fan-out priced per 1k searches.
    Returns None (→ '—' cell) if the embedding model or an enabled reranker
    is missing from the price table, warning once per unknown model.
    """
    embed_price = embedding_price_per_1m(provider, model)
    if embed_price is None:
        _warn_missing(f"embedding {provider}/{model}")
        return None

    mean_tokens = sum(estimate_tokens(q) for q in queries) / len(queries) if queries else 0.0
    cost = mean_tokens / 1_000_000 * embed_price

    if reranker:
        rr_price = reranker_price_per_1k(reranker)
        if rr_price is None:
            _warn_missing(f"reranker {reranker}")
            return None
        cost += (top_k * 3) / 1000 * rr_price

    return cost
