"""Smoke tests for the `./start_milvus.sh attu ...` subcommand dispatcher.

We stub `docker` (and `lsof`) on PATH so the script never talks to the real
Docker daemon. This lets us assert exit codes and printed text without
needing containers running.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "skills"
    / "zilliz-launchpad"
    / "scripts"
    / "start_milvus.sh"
)


@pytest.fixture()
def fake_bin(tmp_path: Path) -> Path:
    """Provide a PATH directory with fake `docker` and `lsof` shims."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "info" ]; then exit 0; fi\n'
        'if [ "$1" = "compose" ] && [ "$2" = "version" ]; then exit 0; fi\n'
        'echo "DOCKER-STUB: $*"\n'
    )
    docker.chmod(0o755)

    lsof = bin_dir / "lsof"
    lsof.write_text("#!/usr/bin/env bash\nexit 1\n")  # port always free
    lsof.chmod(0o755)

    return bin_dir


def _run(args: list[str], fake_bin: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_unknown_attu_subcommand_exits_2(fake_bin: Path) -> None:
    result = _run(["attu", "foo"], fake_bin)
    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert "attu" in result.stderr


def test_attu_no_subcommand_defaults_to_up(fake_bin: Path) -> None:
    result = _run(["attu"], fake_bin)
    assert result.returncode == 0, result.stderr
    assert "DOCKER-STUB: compose --profile ops up -d attu" in result.stdout
    assert "http://localhost:8000" in result.stdout


def test_attu_down_invokes_profile_rm(fake_bin: Path) -> None:
    result = _run(["attu", "down"], fake_bin)
    assert result.returncode == 0, result.stderr
    assert "DOCKER-STUB: compose --profile ops rm -sf attu" in result.stdout


def test_attu_status_invokes_profile_ps(fake_bin: Path) -> None:
    result = _run(["attu", "status"], fake_bin)
    assert result.returncode == 0, result.stderr
    assert "DOCKER-STUB: compose --profile ops ps attu" in result.stdout


def test_top_level_usage_mentions_attu(fake_bin: Path) -> None:
    result = _run(["bogus"], fake_bin)
    assert result.returncode == 2
    assert "attu" in result.stderr
