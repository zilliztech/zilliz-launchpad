"""Unit tests for `lib.zilliz_cli` — all subprocess calls are mocked."""

from __future__ import annotations

import json
import subprocess
from typing import Any
from unittest.mock import patch

import pytest
from lib import zilliz_cli
from lib.errors import LaunchpadError, ZillizCliAuthError, ZillizCliMissingError


def _completed(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["zilliz"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


@pytest.fixture(autouse=True)
def _reset_cache():
    zilliz_cli.invalidate()
    yield
    zilliz_cli.invalidate()


def test_is_available_false_when_binary_missing():
    with patch("lib.zilliz_cli.shutil.which", return_value=None):
        assert zilliz_cli.is_available() is False


def test_is_available_false_when_unauthed():
    def fake_run(args, **kwargs):
        if args[:2] == ["zilliz", "version"]:
            return _completed(stdout=json.dumps({"version": "0.3.5"}))
        if args[1:3] == ["auth", "whoami"]:
            return _completed(returncode=1, stderr="not logged in")
        return _completed(returncode=1)

    with (
        patch("lib.zilliz_cli.shutil.which", return_value="/usr/bin/zilliz"),
        patch("lib.zilliz_cli.subprocess.run", side_effect=fake_run),
    ):
        assert zilliz_cli.is_available() is False


def test_is_available_false_when_version_stale():
    def fake_run(args, **kwargs):
        if args[1:2] == ["version"]:
            return _completed(stdout=json.dumps({"version": "0.1.0"}))
        return _completed()

    with (
        patch("lib.zilliz_cli.shutil.which", return_value="/usr/bin/zilliz"),
        patch("lib.zilliz_cli.subprocess.run", side_effect=fake_run),
    ):
        assert zilliz_cli.is_available() is False


def test_is_available_true_when_authed_and_version_ok():
    def fake_run(args, **kwargs):
        if args[1:2] == ["version"]:
            return _completed(stdout=json.dumps({"version": "0.3.0"}))
        if args[1:3] == ["auth", "whoami"]:
            return _completed(stdout=json.dumps({"token": "za_secret"}))
        return _completed(returncode=1)

    with (
        patch("lib.zilliz_cli.shutil.which", return_value="/usr/bin/zilliz"),
        patch("lib.zilliz_cli.subprocess.run", side_effect=fake_run),
    ):
        assert zilliz_cli.is_available() is True


def test_is_available_caches_result():
    calls = {"n": 0}

    def fake_run(args, **kwargs):
        calls["n"] += 1
        if args[1:2] == ["version"]:
            return _completed(stdout=json.dumps({"version": "0.3.0"}))
        return _completed(stdout=json.dumps({"token": "za_x"}))

    with (
        patch("lib.zilliz_cli.shutil.which", return_value="/usr/bin/zilliz"),
        patch("lib.zilliz_cli.subprocess.run", side_effect=fake_run),
    ):
        assert zilliz_cli.is_available() is True
        first = calls["n"]
        assert zilliz_cli.is_available() is True
        assert calls["n"] == first  # no new subprocess calls


def test_auth_whoami_raises_missing_when_binary_absent():
    with (
        patch("lib.zilliz_cli.shutil.which", return_value=None),
        pytest.raises(ZillizCliMissingError),
    ):
        zilliz_cli.auth_whoami()


def test_auth_whoami_raises_auth_when_not_logged_in():
    with (
        patch("lib.zilliz_cli.shutil.which", return_value="/usr/bin/zilliz"),
        patch(
            "lib.zilliz_cli.subprocess.run",
            return_value=_completed(returncode=1, stderr="no session"),
        ),
        pytest.raises(ZillizCliAuthError),
    ):
        zilliz_cli.auth_whoami()


def test_auth_whoami_returns_payload():
    with (
        patch("lib.zilliz_cli.shutil.which", return_value="/usr/bin/zilliz"),
        patch(
            "lib.zilliz_cli.subprocess.run",
            return_value=_completed(stdout=json.dumps({"token": "za_abc", "org": "o"})),
        ),
    ):
        data = zilliz_cli.auth_whoami()
    assert data == {"token": "za_abc", "org": "o"}


def test_cluster_list_invokes_correct_args():
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[1:2] == ["version"]:
            return _completed(stdout=json.dumps({"version": "0.3.0"}))
        if args[1:3] == ["auth", "whoami"]:
            return _completed(stdout=json.dumps({"token": "za_x"}))
        if args[1:3] == ["cluster", "list"]:
            return _completed(stdout=json.dumps([{"clusterId": "c1"}, {"clusterId": "c2"}]))
        return _completed(returncode=1)

    with (
        patch("lib.zilliz_cli.shutil.which", return_value="/usr/bin/zilliz"),
        patch("lib.zilliz_cli.subprocess.run", side_effect=fake_run),
    ):
        clusters = zilliz_cli.cluster_list()
    assert clusters == [{"clusterId": "c1"}, {"clusterId": "c2"}]
    assert ["zilliz", "cluster", "list", "--output", "json"] in calls


def test_cluster_describe_passes_cluster_id():
    captured: dict[str, list[str]] = {}

    def fake_run(args, **kwargs):
        if args[1:2] == ["version"]:
            return _completed(stdout=json.dumps({"version": "0.3.0"}))
        if args[1:3] == ["auth", "whoami"]:
            return _completed(stdout=json.dumps({"token": "za_x"}))
        if args[1:3] == ["cluster", "describe"]:
            captured["args"] = args
            return _completed(stdout=json.dumps({"state": "RUNNING"}))
        return _completed(returncode=1)

    with (
        patch("lib.zilliz_cli.shutil.which", return_value="/usr/bin/zilliz"),
        patch("lib.zilliz_cli.subprocess.run", side_effect=fake_run),
    ):
        out = zilliz_cli.cluster_describe("c-xyz")
    assert out == {"state": "RUNNING"}
    assert "c-xyz" in captured["args"]


def test_nonzero_exit_raises_launchpad_error():
    def fake_run(args, **kwargs):
        if args[1:2] == ["version"]:
            return _completed(stdout=json.dumps({"version": "0.3.0"}))
        if args[1:3] == ["auth", "whoami"]:
            return _completed(stdout=json.dumps({"token": "za_x"}))
        if args[1:3] == ["cluster", "list"]:
            return _completed(returncode=2, stderr="boom")
        return _completed(returncode=1)

    with (
        patch("lib.zilliz_cli.shutil.which", return_value="/usr/bin/zilliz"),
        patch("lib.zilliz_cli.subprocess.run", side_effect=fake_run),
        pytest.raises(LaunchpadError) as exc,
    ):
        zilliz_cli.cluster_list()
    assert "boom" in str(exc.value)


def test_import_create_builds_file_args():
    captured: dict[str, list[str]] = {}

    def fake_run(args, **kwargs):
        if args[1:2] == ["version"]:
            return _completed(stdout=json.dumps({"version": "0.3.0"}))
        if args[1:3] == ["auth", "whoami"]:
            return _completed(stdout=json.dumps({"token": "za_x"}))
        if args[1:3] == ["import", "create"]:
            captured["args"] = args
            return _completed(stdout=json.dumps({"jobId": "j-1"}))
        return _completed(returncode=1)

    with (
        patch("lib.zilliz_cli.shutil.which", return_value="/usr/bin/zilliz"),
        patch("lib.zilliz_cli.subprocess.run", side_effect=fake_run),
    ):
        out = zilliz_cli.import_create(
            cluster_id="c1", collection_name="col", files=["a.jsonl", "b.jsonl"]
        )
    assert out == {"jobId": "j-1"}
    assert captured["args"].count("--file") == 2


def test_import_describe_parses_state():
    def fake_run(args, **kwargs):
        if args[1:2] == ["version"]:
            return _completed(stdout=json.dumps({"version": "0.3.0"}))
        if args[1:3] == ["auth", "whoami"]:
            return _completed(stdout=json.dumps({"token": "za_x"}))
        if args[1:3] == ["import", "describe"]:
            return _completed(stdout=json.dumps({"state": "DONE"}))
        return _completed(returncode=1)

    with (
        patch("lib.zilliz_cli.shutil.which", return_value="/usr/bin/zilliz"),
        patch("lib.zilliz_cli.subprocess.run", side_effect=fake_run),
    ):
        assert zilliz_cli.import_describe("j1")["state"] == "DONE"


def test_cluster_create_invokes_cli_with_plan_and_region():
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[1:2] == ["version"]:
            return _completed(stdout=json.dumps({"version": "0.3.0"}))
        if args[1:3] == ["auth", "whoami"]:
            return _completed(stdout=json.dumps({"token": "za_x"}))
        if args[1:3] == ["cluster", "create"]:
            return _completed(stdout=json.dumps({"cluster_id": "c-new"}))
        return _completed(returncode=1)

    with (
        patch("lib.zilliz_cli.shutil.which", return_value="/usr/bin/zilliz"),
        patch("lib.zilliz_cli.subprocess.run", side_effect=fake_run),
    ):
        out = zilliz_cli.cluster_create(
            cluster_name="launchpad-x",
            plan="Serverless",
            region="gcp-us-west1",
            project_id=None,
            extra=None,
        )
    assert out["cluster_id"] == "c-new"
    # The create invocation must pass the three key arguments
    create_args = [a for a in calls if a[1:3] == ["cluster", "create"]][0]
    assert "--cluster-name" in create_args and "launchpad-x" in create_args
    assert "--plan" in create_args and "Serverless" in create_args
    assert "--region" in create_args and "gcp-us-west1" in create_args


def test_cluster_create_nonzero_raises_cluster_create_failed():
    from lib.errors import ClusterCreateFailedError

    def fake_run(args, **kwargs):
        if args[1:2] == ["version"]:
            return _completed(stdout=json.dumps({"version": "0.3.0"}))
        if args[1:3] == ["auth", "whoami"]:
            return _completed(stdout=json.dumps({"token": "za_x"}))
        if args[1:3] == ["cluster", "create"]:
            return _completed(returncode=2, stderr="quota exceeded; za_TOKEN embedded")
        return _completed(returncode=1)

    with (
        patch("lib.zilliz_cli.shutil.which", return_value="/usr/bin/zilliz"),
        patch("lib.zilliz_cli.subprocess.run", side_effect=fake_run),
        pytest.raises(ClusterCreateFailedError) as info,
    ):
        zilliz_cli.cluster_create(
            cluster_name="x",
            plan="Serverless",
            region="r",
            project_id=None,
            extra=None,
        )
    assert info.value.payload["exit_code"] == 2
    # Tokens must be scrubbed from stderr before bubbling up
    assert "za_TOKEN" not in info.value.payload["stderr"]
    assert "za_***" in info.value.payload["stderr"]


def test_extract_cluster_id_finds_top_level():
    assert zilliz_cli.extract_cluster_id({"cluster_id": "c-1"}) == "c-1"
    assert zilliz_cli.extract_cluster_id({"clusterId": "c-2"}) == "c-2"
    assert zilliz_cli.extract_cluster_id({"id": "c-3"}) == "c-3"


def test_extract_cluster_id_finds_nested():
    assert zilliz_cli.extract_cluster_id({"cluster": {"id": "c-9"}}) == "c-9"
    assert zilliz_cli.extract_cluster_id({"data": {"clusterId": "c-10"}}) == "c-10"


def test_extract_cluster_id_missing_returns_none():
    assert zilliz_cli.extract_cluster_id({"unrelated": "x"}) is None


def test_whoami_output_not_logged(caplog: Any):
    caplog.set_level("DEBUG")
    with (
        patch("lib.zilliz_cli.shutil.which", return_value="/usr/bin/zilliz"),
        patch(
            "lib.zilliz_cli.subprocess.run",
            return_value=_completed(stdout=json.dumps({"token": "za_SECRET_XYZ"})),
        ),
    ):
        zilliz_cli.auth_whoami()
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "za_SECRET_XYZ" not in joined


def test_mask_scrubs_token_patterns():
    out = zilliz_cli._mask("error with za_Secret123 inside")
    assert "za_Secret123" not in out
    assert "za_***" in out
