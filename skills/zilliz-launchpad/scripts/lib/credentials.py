"""Env-first credential resolution.

Usage:
    token = resolve("ZILLIZ_TOKEN")              # raises if missing
    key = resolve("OPENAI_API_KEY", optional=True)  # returns None if missing

The skill layer catches `MissingCredentialError`, prompts the user in
dialogue, exports the value, and re-invokes. The CLI never prompts on
stdin — keeps it scriptable outside an agent.
"""

from __future__ import annotations

import os

from .errors import MissingCredentialError

HINTS: dict[str, str] = {
    "ZILLIZ_TOKEN": (
        "Get a token from https://cloud.zilliz.com → Project → API Keys, "
        "then `export ZILLIZ_TOKEN=<value>`"
    ),
    "OPENAI_API_KEY": "Get a key from https://platform.openai.com/api-keys",
    "COHERE_API_KEY": "Get a key from https://dashboard.cohere.com/api-keys",
    "VOYAGE_API_KEY": "Get a key from https://dash.voyageai.com",
    "ZILLIZ_BYOM_URL": "Endpoint URL of your Zilliz BYOM embedding service",
    "ZILLIZ_BYOM_KEY": "API key for your Zilliz BYOM embedding service",
}


def resolve(key: str, *, optional: bool = False) -> str | None:
    value = os.environ.get(key)
    if value:
        return value
    if optional:
        return None
    raise MissingCredentialError(env_var=key, hint=HINTS.get(key))


def resolve_required(key: str) -> str:
    value = resolve(key, optional=False)
    assert value is not None  # for type-checker; resolve() raises otherwise
    return value
