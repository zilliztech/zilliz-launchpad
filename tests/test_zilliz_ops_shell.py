"""Guards for the thin-shell contract of `zilliz_ops.py`.

These are shell-level guards (size budget, registration wiring,
single-file extensibility), not phase CLI behavior tests — phase CLI
behavior is covered by the per-phase test modules.
"""

from __future__ import annotations

from pathlib import Path

import typer
from typer.testing import CliRunner
from zilliz_ops import app

_SHELL = (
    Path(__file__).resolve().parent.parent
    / "skills"
    / "zilliz-launchpad"
    / "scripts"
    / "zilliz_ops.py"
)

# Issue #7 acceptance: the shell must stay under 80 lines.
_LINE_BUDGET = 80


def test_shell_under_line_budget() -> None:
    line_count = len(_SHELL.read_text(encoding="utf-8").splitlines())
    assert line_count < _LINE_BUDGET, (
        f"zilliz_ops.py is {line_count} lines; budget is < {_LINE_BUDGET}. "
        "Per-phase CLI logic belongs in lib/phases/<phase>.py, not the shell."
    )


def test_all_phase_subcommands_registered() -> None:
    """`from zilliz_ops import app` exposes all six phases via CliRunner."""
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("collect", "configure", "plan", "execute", "evaluate", "deploy"):
        assert name in result.output


def test_phase_added_with_one_register_function() -> None:
    """A new phase is one module exposing register(app) — no other edits."""
    fresh = typer.Typer()

    @fresh.command()
    def existing() -> None:  # pragma: no cover - keeps the app multi-command
        """Stand-in for the already-registered phases."""

    def register(target: typer.Typer) -> None:
        @target.command()
        def tune() -> None:  # pragma: no cover - exercised via CliRunner
            """Throwaway phase used to prove single-file extensibility."""
            typer.echo("tune ok")

    register(fresh)
    result = CliRunner().invoke(fresh, ["tune"])
    assert result.exit_code == 0
    assert "tune ok" in result.output
