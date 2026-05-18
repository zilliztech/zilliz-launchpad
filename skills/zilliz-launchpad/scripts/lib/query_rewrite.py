"""Judge-LLM query rewriter for Phase 5 text-derived eval.

Derived mode samples the corpus and uses the first sentence of each doc
verbatim as the query. Because that query is a literal substring of the
indexed text, recall@10 saturates and the smoke metric loses signal. When
the user opts in with `--judge-llm <provider>:<model>`, this module
paraphrases each derived query so it stays on-intent but no longer copies
the source sentence.

Mirrors `vision_judge.py`: lazy provider import, `resolve_required` for
the credential, `_clean()` of the response, sequential batching (callers
own caching). Kept out of `evaluator.py` so that module keeps its
"no LLM/Milvus at import time" contract.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from .credentials import resolve
from .errors import JudgeUnavailableError

logger = logging.getLogger(__name__)

# Providers we trust for the text rewrite. Kept tight on purpose (matches
# what `vision_judge.py` supports); other providers raise judge_unavailable.
_PROVIDER_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

_REWRITE_PROMPT = (
    "Rewrite the sentence below as one natural search query (5-12 words) "
    "that a real user might type to find the document it came from. "
    "Preserve the original intent, but do NOT reuse any verbatim phrase "
    "from the sentence — paraphrase it. No quotes, no preamble, just the "
    "query.\n\nSentence: "
)


def _env_var(provider: str) -> str:
    return _PROVIDER_ENV.get(provider.lower(), "OPENAI_API_KEY")


def rewrite_query(provider: str, model: str, original: str) -> str:
    """Return a paraphrased search query for `original` via the named LLM.

    Raises `JudgeUnavailableError` when the provider is unsupported or its
    credential is unresolved — checked *before* any network call. When the
    model returns blank output, degrades to the original verbatim string
    rather than crashing the eval.
    """
    prov = provider.lower()
    if prov not in _PROVIDER_ENV:
        raise JudgeUnavailableError(provider=f"{provider}:{model}", env_var=_env_var(prov))

    env_var = _PROVIDER_ENV[prov]
    if not resolve(env_var, optional=True):
        raise JudgeUnavailableError(provider=f"{provider}:{model}", env_var=env_var)

    if prov == "openai":
        rewritten = _rewrite_openai(model, original)
    else:
        rewritten = _rewrite_anthropic(model, original)

    cleaned = _clean(rewritten)
    return cleaned or original


def rewrite_queries(provider: str, model: str, originals: Iterable[str]) -> list[str]:
    """Sequential rewrite. LLM APIs are rate-limited and Phase 5 only
    rewrites ~25 derived queries; parallelism is left to the caller.
    """
    return [rewrite_query(provider, model, q) for q in originals]


# --- Provider-specific clients --------------------------------------------


def _rewrite_openai(model: str, original: str) -> str:
    from openai import OpenAI  # noqa: PLC0415

    from .credentials import resolve_required  # noqa: PLC0415

    client = OpenAI(api_key=resolve_required("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        max_tokens=64,
        messages=[{"role": "user", "content": _REWRITE_PROMPT + original}],
    )
    msg = resp.choices[0].message.content if resp.choices else None
    return msg or ""


def _rewrite_anthropic(model: str, original: str) -> str:
    try:
        from anthropic import Anthropic  # noqa: PLC0415
    except ImportError as exc:
        raise JudgeUnavailableError(
            provider=f"anthropic:{model}", env_var="ANTHROPIC_API_KEY"
        ) from exc

    from .credentials import resolve_required  # noqa: PLC0415

    client = Anthropic(api_key=resolve_required("ANTHROPIC_API_KEY"))
    msg: Any = client.messages.create(
        model=model,
        max_tokens=64,
        messages=[{"role": "user", "content": _REWRITE_PROMPT + original}],
    )
    text_parts: list[str] = []
    for block in msg.content or []:
        if getattr(block, "type", None) == "text":
            text_parts.append(getattr(block, "text", ""))
    return " ".join(text_parts)


def _clean(s: str) -> str:
    return s.strip().strip('"').strip("'").strip()
