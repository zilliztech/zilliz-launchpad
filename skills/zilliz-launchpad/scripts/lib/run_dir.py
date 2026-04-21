"""Per-invocation run directory helpers.

Each phase writes into `scripts/runs/<utc-iso>/`. UTC timestamps are
lexicographically sortable, stable, and timezone-free.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

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
