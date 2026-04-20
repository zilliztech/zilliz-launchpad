"""Structured error classes with a common CLI-serializable form.

All launchpad errors inherit `LaunchpadError` so the CLI can catch a single
type and render a consistent JSON payload — agents parse this payload to
decide whether to prompt the user, retry, or abort.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


class LaunchpadError(Exception):
    """Base class. Subclasses fill `code` and `payload`."""

    code: str = "launchpad_error"

    def __init__(self, message: str, **payload: Any) -> None:
        super().__init__(message)
        self.message = message
        self.payload: dict[str, Any] = payload

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.payload}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


class MissingCredentialError(LaunchpadError):
    code = "missing_credential"

    def __init__(self, env_var: str, *, hint: str | None = None) -> None:
        super().__init__(
            f"Required credential not set: {env_var}",
            env_var=env_var,
            export_hint=hint or f"export {env_var}=<value>",
        )


class SchemaConflictError(LaunchpadError):
    code = "schema_conflict"

    def __init__(self, collection: str, mismatches: list[str]) -> None:
        super().__init__(
            f"Existing collection '{collection}' has incompatible schema",
            collection=collection,
            mismatches=mismatches,
        )


class SparseUnavailable(LaunchpadError):
    code = "sparse_unavailable"

    def __init__(self, collection: str) -> None:
        super().__init__(
            f"Collection '{collection}' has no sparse vector field",
            collection=collection,
            remediation="Re-ingest with `sparse.enabled=true` in the plan.",
        )


class InvalidProfileError(LaunchpadError):
    code = "invalid_profile"

    def __init__(self, pointer: str, reason: str) -> None:
        super().__init__(
            f"Requirement profile invalid at {pointer}: {reason}",
            pointer=pointer,
            reason=reason,
        )


class BackendUnsupportedError(LaunchpadError):
    code = "backend_unsupported"

    def __init__(self, uri: str, feature: str) -> None:
        super().__init__(
            f"Backend at {uri} does not support {feature}",
            uri=uri,
            feature=feature,
        )


@dataclass
class CliErrorEnvelope:
    """Uniform envelope emitted to stderr on failure."""

    code: str
    message: str
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_error(cls, err: LaunchpadError) -> "CliErrorEnvelope":
        d = err.to_dict()
        code = d.pop("code")
        message = d.pop("message")
        return cls(code=code, message=message, extra=d)

    def to_json(self) -> str:
        return json.dumps(
            {"code": self.code, "message": self.message, **self.extra},
            ensure_ascii=False,
            sort_keys=True,
        )
