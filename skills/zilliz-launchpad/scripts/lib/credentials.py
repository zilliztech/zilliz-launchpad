"""Env-first credential resolution.

Usage:
    token = resolve("ZILLIZ_TOKEN")              # raises if missing
    key = resolve("OPENAI_API_KEY", optional=True)  # returns None if missing
    token = resolve("ZILLIZ_TOKEN", allow_cli=True)  # fall back to zilliz CLI

The skill layer catches `MissingCredentialError`, prompts the user in
dialogue, exports the value, and re-invokes. The CLI never prompts on
stdin — keeps it scriptable outside an agent.
"""

from __future__ import annotations

import os

from .errors import LaunchpadError, MissingCredentialError

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


def _cli_token() -> str | None:
    """Best-effort token lookup via the optional zilliz CLI.

    Returns None when the CLI is absent, not logged in, or errors. Callers
    must tolerate a None here and fall through to the normal missing-cred
    error path.
    """
    try:
        from . import zilliz_cli  # local import keeps unit tests subprocess-free
    except Exception:
        return None
    try:
        if not zilliz_cli.is_available():
            return None
        data = zilliz_cli.auth_whoami()
    except LaunchpadError:
        return None
    token = data.get("token") or data.get("api_key") or data.get("apiKey")
    return str(token) if token else None


def resolve(key: str, *, optional: bool = False, allow_cli: bool = False) -> str | None:
    value = os.environ.get(key)
    if value:
        return value
    if allow_cli and key == "ZILLIZ_TOKEN":
        cli_val = _cli_token()
        if cli_val:
            return cli_val
    if optional:
        return None
    raise MissingCredentialError(env_var=key, hint=HINTS.get(key))


def resolve_required(key: str, *, allow_cli: bool = False) -> str:
    value = resolve(key, optional=False, allow_cli=allow_cli)
    assert value is not None  # for type-checker; resolve() raises otherwise
    return value
