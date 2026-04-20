"""URI-aware connection factory.

Business code calls `MilvusClient(uri=...)` and never branches on backend.
The factory detects:
  - Zilliz Cloud   → *.zillizcloud.com host
  - Local / OSS    → anything else (no token required by default)
and resolves a token from the environment when required.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import urlparse

from pymilvus import MilvusClient as _PyMilvusClient

from .credentials import resolve
from .errors import MissingCredentialError


class Backend(str, Enum):
    ZILLIZ_CLOUD = "zilliz-cloud"
    LOCAL = "local"


@dataclass(frozen=True)
class ConnectionTarget:
    uri: str
    backend: Backend
    requires_token: bool


def detect_target(uri: str) -> ConnectionTarget:
    host = (urlparse(uri).hostname or "").lower()
    if host.endswith("zillizcloud.com"):
        return ConnectionTarget(uri=uri, backend=Backend.ZILLIZ_CLOUD, requires_token=True)
    return ConnectionTarget(uri=uri, backend=Backend.LOCAL, requires_token=False)


def MilvusClient(uri: str, token: str | None = None, **kwargs: Any) -> _PyMilvusClient:
    """Return a configured pymilvus client.

    `token` resolution:
      1. explicit argument
      2. `ZILLIZ_TOKEN` env var (Zilliz Cloud only)
      3. raises `MissingCredentialError` if Cloud target has no token
    """
    target = detect_target(uri)
    resolved = token
    if target.requires_token and resolved is None:
        resolved = resolve("ZILLIZ_TOKEN", optional=True, allow_cli=True)
        if resolved is None:
            raise MissingCredentialError(env_var="ZILLIZ_TOKEN")

    client_kwargs: dict[str, Any] = {"uri": uri, **kwargs}
    if resolved is not None:
        client_kwargs["token"] = resolved
    return _PyMilvusClient(**client_kwargs)
