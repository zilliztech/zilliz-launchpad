"""CLI tests for the `plan diff` subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import lib.run_dir as run_dir
import pytest
from typer.testing import CliRunner
from zilliz_ops import app

_RUNNER = CliRunner()
_PLAN_MD = (
    "# Launchpad Plan\n\n"
    "- **Collection**: `launchpad_collection`\n\n"
    "## Embedding\n"
    "- Model: `text-embedding-3-small`\n"
)


@pytest.fixture
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(run_dir, "_RUNS_ROOT", root)
    return root


def _mk_run(root: Path, name: str, plan_md: str | None = _PLAN_MD) -> Path:
    p = root / name
    p.mkdir()
    if plan_md is not None:
        (p / "plan.md").write_text(plan_md, encoding="utf-8")
    return p


def test_identical_plans_exit_0_no_output(runs_root: Path) -> None:
    _mk_run(runs_root, "2026-05-18T09-00-00Z")
    _mk_run(runs_root, "2026-05-18T10-00-00Z")
    result = _RUNNER.invoke(app, ["plan", "diff"])
    assert result.exit_code == 0
    assert result.stdout == ""


def test_differing_plans_exit_1_with_unified_markers(runs_root: Path) -> None:
    _mk_run(runs_root, "2026-05-18T09-00-00Z")
    _mk_run(
        runs_root,
        "2026-05-18T10-00-00Z",
        _PLAN_MD.replace("text-embedding-3-small", "voyage-3"),
    )
    result = _RUNNER.invoke(app, ["plan", "diff"])
    assert result.exit_code == 1
    out = result.stdout
    assert "--- 2026-05-18T09-00-00Z/plan.md" in out
    assert "+++ 2026-05-18T10-00-00Z/plan.md" in out
    assert "-- Model: `text-embedding-3-small`" in out
    assert "+- Model: `voyage-3`" in out


def test_no_arg_resolves_latest_vs_previous(runs_root: Path) -> None:
    _mk_run(runs_root, "2026-05-18T08-00-00Z", _PLAN_MD.replace("small", "OLDEST"))
    _mk_run(runs_root, "2026-05-18T09-00-00Z")  # previous (old side)
    _mk_run(
        runs_root,
        "2026-05-18T10-00-00Z",
        _PLAN_MD.replace("text-embedding-3-small", "voyage-3"),
    )  # latest (new side)
    result = _RUNNER.invoke(app, ["plan", "diff"])
    assert result.exit_code == 1
    # Diff is latest vs the immediately-preceding run, not the oldest.
    assert "--- 2026-05-18T09-00-00Z/plan.md" in result.stdout
    assert "+++ 2026-05-18T10-00-00Z/plan.md" in result.stdout
    assert "OLDEST" not in result.stdout


def test_single_arg_is_run_a_with_defaulted_predecessor(runs_root: Path) -> None:
    _mk_run(runs_root, "2026-05-18T09-00-00Z")  # predecessor → run-b
    _mk_run(
        runs_root,
        "2026-05-18T10-00-00Z",
        _PLAN_MD.replace("text-embedding-3-small", "voyage-3"),
    )
    _mk_run(runs_root, "2026-05-18T11-00-00Z", _PLAN_MD.replace("small", "NEWER"))
    result = _RUNNER.invoke(app, ["plan", "diff", "2026-05-18T10-00-00Z"])
    assert result.exit_code == 1
    assert "--- 2026-05-18T09-00-00Z/plan.md" in result.stdout
    assert "+++ 2026-05-18T10-00-00Z/plan.md" in result.stdout
    assert "NEWER" not in result.stdout


def test_two_explicit_args_no_defaulting(runs_root: Path) -> None:
    _mk_run(runs_root, "2026-05-18T09-00-00Z")
    _mk_run(
        runs_root,
        "2026-05-18T10-00-00Z",
        _PLAN_MD.replace("text-embedding-3-small", "voyage-3"),
    )
    result = _RUNNER.invoke(app, ["plan", "diff", "2026-05-18T10-00-00Z", "2026-05-18T09-00-00Z"])
    assert result.exit_code == 1
    assert "+++ 2026-05-18T10-00-00Z/plan.md" in result.stdout


def _assert_error_envelope(result, code: str) -> None:
    assert result.exit_code == 2  # non-1: distinct from the "differ" signal
    payload = json.loads(result.stderr.strip().splitlines()[-1])
    assert payload["code"] == code


def test_missing_run_dir_errors(runs_root: Path) -> None:
    _mk_run(runs_root, "2026-05-18T09-00-00Z")
    result = _RUNNER.invoke(app, ["plan", "diff", "does-not-exist"])
    _assert_error_envelope(result, "invalid_profile")


def test_missing_plan_md_errors(runs_root: Path) -> None:
    _mk_run(runs_root, "2026-05-18T09-00-00Z", plan_md=None)
    _mk_run(runs_root, "2026-05-18T10-00-00Z")
    result = _RUNNER.invoke(app, ["plan", "diff"])
    _assert_error_envelope(result, "invalid_profile")


def test_single_run_no_previous_errors(runs_root: Path) -> None:
    _mk_run(runs_root, "2026-05-18T09-00-00Z")
    result = _RUNNER.invoke(app, ["plan", "diff"])
    _assert_error_envelope(result, "invalid_profile")


def test_bare_plan_still_runs_phase_3(runs_root: Path) -> None:
    """Sub-app conversion must not change `plan`'s standalone behavior."""
    from lib.phases.collect import run_collect
    from lib.phases.configure import run_configure

    run = runs_root / "2026-05-18T12-00-00Z"
    run.mkdir()
    run_collect(input_path=None, sample="movies", out_dir=run)
    run_configure(
        from_json=None,
        out_dir=run,
        overrides={"dataset_size": 20, "deployment_target": "local-standalone"},
    )
    result = _RUNNER.invoke(app, ["plan", "--run-dir", "2026-05-18T12-00-00Z"])
    assert result.exit_code == 0
    assert "index:" in result.stdout
    assert "sparse:" in result.stdout
    assert (run / "plan.json").exists()
    assert (run / "plan.md").exists()
