"""Unit tests for run-dir ordering helpers."""

from __future__ import annotations

from pathlib import Path

import lib.run_dir as run_dir
import pytest


@pytest.fixture
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(run_dir, "_RUNS_ROOT", root)
    return root


def _mk(root: Path, name: str) -> Path:
    p = root / name
    p.mkdir()
    return p


def test_previous_run_dir_returns_predecessor(runs_root: Path) -> None:
    a = _mk(runs_root, "2026-05-18T09-00-00Z")
    b = _mk(runs_root, "2026-05-18T10-00-00Z")
    c = _mk(runs_root, "2026-05-18T11-00-00Z")
    assert run_dir.previous_run_dir(c) == b
    assert run_dir.previous_run_dir(b) == a


def test_previous_run_dir_first_run_returns_none(runs_root: Path) -> None:
    a = _mk(runs_root, "2026-05-18T09-00-00Z")
    _mk(runs_root, "2026-05-18T10-00-00Z")
    assert run_dir.previous_run_dir(a) is None


def test_previous_run_dir_single_run_returns_none(runs_root: Path) -> None:
    only = _mk(runs_root, "2026-05-18T09-00-00Z")
    assert run_dir.previous_run_dir(only) is None


def test_previous_run_dir_unknown_run_returns_none(runs_root: Path, tmp_path: Path) -> None:
    _mk(runs_root, "2026-05-18T09-00-00Z")
    stranger = tmp_path / "not-a-run"
    stranger.mkdir()
    assert run_dir.previous_run_dir(stranger) is None
