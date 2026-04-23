"""Per-invocation run directory helpers.

Each phase writes into `scripts/runs/<utc-iso>/`. UTC timestamps are
lexicographically sortable, stable, and timezone-free.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import InvalidProfileError

_RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs"


def runs_root() -> Path:
    _RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    return _RUNS_ROOT


def new_run_dir(label: str | None = None) -> Path:
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    name = f"{ts}-{label}" if label else ts
    path = runs_root() / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def latest_run_dir() -> Path | None:
    candidates = sorted(p for p in runs_root().iterdir() if p.is_dir())
    return candidates[-1] if candidates else None


def resolve_run_dir(arg: str | None) -> Path:
    """`arg` may be `None` (→ latest), a relative path, or an absolute path."""
    if arg is None:
        latest = latest_run_dir()
        if latest is None:
            raise FileNotFoundError("No run directory yet — run `collect` first.")
        return latest
    p = Path(arg)
    if not p.is_absolute():
        p = runs_root() / p
    if not p.exists():
        raise FileNotFoundError(f"Run directory not found: {p}")
    return p


def _load_json(path: Path, *, artifact: str) -> dict[str, Any]:
    if not path.exists():
        raise InvalidProfileError(
            pointer=str(path),
            reason=f"Required artifact '{artifact}' is missing from the run directory",
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InvalidProfileError(pointer=str(path), reason=f"not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise InvalidProfileError(
            pointer=str(path),
            reason=f"expected a JSON object, got {type(data).__name__}",
        )
    return data


def load_collect(run_dir: Path) -> dict[str, Any]:
    return _load_json(run_dir / "collect.json", artifact="collect.json")


def load_configure(run_dir: Path) -> dict[str, Any]:
    return _load_json(run_dir / "configure.json", artifact="configure.json")


def load_plan(run_dir: Path) -> dict[str, Any]:
    return _load_json(run_dir / "plan.json", artifact="plan.json")


def load_execute(run_dir: Path) -> dict[str, Any]:
    return _load_json(run_dir / "execute.json", artifact="execute.json")


def preflight_execute_artifact(run_dir: Path) -> dict[str, Any]:
    """Return parsed execute.json; raise invalid_profile if it is not present.

    Phases 5 and 6 call this before any work — they need a completed Execute
    run to target. Surfacing the missing artifact as `invalid_profile` lets
    the skill prompt the user to re-run Execute before retrying.
    """
    return load_execute(run_dir)
