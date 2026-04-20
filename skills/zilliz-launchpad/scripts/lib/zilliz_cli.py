"""Optional `zilliz` CLI subprocess wrapper.

The launchpad's local-Milvus path never needs this module. Cloud flows use it
for cluster auto-discovery, pre-flight checks, and bulk imports. Every call
site first checks `is_available()`; the result is cached per-process.

Never log the raw output of `auth_whoami` — it contains an API token. Any
helper that echoes stderr/stdout for debugging MUST scrub `za_*` substrings.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from typing import Any

from .errors import (
    LaunchpadError,
    ZillizCliAuthError,
    ZillizCliMissingError,
)

BINARY = "zilliz"
MIN_VERSION = "0.3.0"
_TOKEN_PATTERN = re.compile(r"za_[A-Za-z0-9]+")

logger = logging.getLogger(__name__)

_cached_available: bool | None = None


def _mask(text: str) -> str:
    """Replace any `za_...` token substrings with `za_***`."""
    return _TOKEN_PATTERN.sub("za_***", text)


def invalidate() -> None:
    """Clear cached availability. Test-only hook."""
    global _cached_available
    _cached_available = None


def _parse_version(text: str) -> tuple[int, ...]:
    cleaned = text.strip().lstrip("v")
    parts: list[int] = []
    for chunk in cleaned.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _version_ok(got: str, wanted: str) -> bool:
    return _parse_version(got) >= _parse_version(wanted)


def _run(
    args: list[str],
    *,
    allow_nonzero: bool = False,
    log_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Invoke `zilliz <args>` capturing stdout/stderr as text."""
    result = subprocess.run(  # noqa: S603 — fixed binary name, args come from our wrappers
        [BINARY, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 and not allow_nonzero:
        stderr = _mask(result.stderr or "").strip()
        raise LaunchpadError(
            f"zilliz {' '.join(args)} failed (exit {result.returncode}): {stderr}",
            stderr=stderr,
            returncode=result.returncode,
        )
    if log_output and logger.isEnabledFor(logging.DEBUG):
        logger.debug("zilliz %s stdout: %s", " ".join(args), _mask(result.stdout or ""))
    return result


def _json_stdout(result: subprocess.CompletedProcess[str]) -> Any:
    text = (result.stdout or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LaunchpadError(
            f"zilliz CLI returned non-JSON output: {exc}",
            stdout_preview=_mask(text[:200]),
        ) from exc


def is_available() -> bool:
    """Return True only when the CLI is present, a supported version, and logged in.

    Cached per-process. Call `invalidate()` to reset (tests only).
    """
    global _cached_available
    if _cached_available is not None:
        return _cached_available

    if shutil.which(BINARY) is None:
        _cached_available = False
        return False

    try:
        version_raw = _run(["version", "--output", "json"])
    except LaunchpadError:
        _cached_available = False
        return False
    parsed = _json_stdout(version_raw) or {}
    version_text = ""
    if isinstance(parsed, dict):
        version_text = str(parsed.get("version") or parsed.get("Version") or "")
    elif isinstance(parsed, str):
        version_text = parsed
    if not version_text or not _version_ok(version_text, MIN_VERSION):
        _cached_available = False
        return False

    auth_result = _run(["auth", "whoami", "--output", "json"], allow_nonzero=True, log_output=False)
    if auth_result.returncode != 0:
        auth_result = _run(["auth", "status"], allow_nonzero=True, log_output=False)
    if auth_result.returncode != 0:
        _cached_available = False
        return False

    _cached_available = True
    return True


def _require_available() -> None:
    if shutil.which(BINARY) is None:
        raise ZillizCliMissingError()
    if not is_available():
        raise ZillizCliAuthError()


def auth_whoami() -> dict[str, Any]:
    """Return the authenticated session payload (token + context).

    Raises `ZillizCliAuthError` when the CLI is present but not logged in,
    and `ZillizCliMissingError` when it is absent.

    Output intentionally NEVER hits the debug log.
    """
    if shutil.which(BINARY) is None:
        raise ZillizCliMissingError()
    result = _run(["auth", "whoami", "--output", "json"], allow_nonzero=True, log_output=False)
    if result.returncode != 0:
        raise ZillizCliAuthError()
    data = _json_stdout(result)
    if not isinstance(data, dict):
        raise ZillizCliAuthError("auth whoami returned an unexpected shape")
    return data


def cluster_list() -> list[dict[str, Any]]:
    _require_available()
    result = _run(["cluster", "list", "--output", "json"])
    data = _json_stdout(result)
    if data is None:
        return []
    if isinstance(data, dict):
        data = data.get("clusters") or data.get("items") or []
    if not isinstance(data, list):
        raise LaunchpadError("zilliz cluster list returned an unexpected shape")
    return [c for c in data if isinstance(c, dict)]


def cluster_describe(cluster_id: str) -> dict[str, Any]:
    _require_available()
    result = _run(["cluster", "describe", "--cluster-id", cluster_id, "--output", "json"])
    data = _json_stdout(result)
    if not isinstance(data, dict):
        raise LaunchpadError("zilliz cluster describe returned an unexpected shape")
    return data


def cluster_create_stub(*_args: Any, **_kwargs: Any) -> None:
    """Placeholder for Phase 6 Deploy.

    Intentionally raises so any accidental call surfaces the fact that
    Deploy is not yet implemented.
    """
    raise NotImplementedError(
        "cluster_create is part of the future Phase 6 Deploy change; not implemented yet."
    )


def import_create(
    *,
    cluster_id: str,
    collection_name: str,
    files: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _require_available()
    args = [
        "import",
        "create",
        "--cluster-id",
        cluster_id,
        "--collection-name",
        collection_name,
        "--output",
        "json",
    ]
    for f in files:
        args += ["--file", f]
    if extra:
        for k, v in extra.items():
            args += [f"--{k}", str(v)]
    result = _run(args)
    data = _json_stdout(result)
    if not isinstance(data, dict):
        raise LaunchpadError("zilliz import create returned an unexpected shape")
    return data


def import_describe(job_id: str) -> dict[str, Any]:
    _require_available()
    result = _run(["import", "describe", "--job-id", job_id, "--output", "json"])
    data = _json_stdout(result)
    if not isinstance(data, dict):
        raise LaunchpadError("zilliz import describe returned an unexpected shape")
    return data
