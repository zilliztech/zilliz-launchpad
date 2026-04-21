"""Tests for Phase 2 Configure — CLI-present vs CLI-absent paths."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from lib.phases.configure import run_configure


def test_local_target_never_invokes_cli(tmp_path: Path):
    with patch("lib.zilliz_cli.is_available") as is_avail:
        data = run_configure(
            from_json=None,
            out_dir=tmp_path,
            overrides={"deployment_target": "local-standalone"},
        )
    is_avail.assert_not_called()
    assert "cluster_id" not in data
    assert "resolved_from_cli" not in data


def test_cloud_target_without_cli_falls_back_silently(tmp_path: Path):
    with patch("lib.zilliz_cli.is_available", return_value=False):
        data = run_configure(
            from_json=None,
            out_dir=tmp_path,
            overrides={"deployment_target": "zilliz-serverless"},
        )
    assert data["deployment_target"] == "zilliz-serverless"
    assert "cluster_id" not in data


def test_cloud_target_with_cli_populates_cluster_id(tmp_path: Path):
    fake_clusters = [
        {
            "clusterId": "c-serverless-1",
            "tier": "SERVERLESS",
            "connectAddress": "https://x.zillizcloud.com",
        }
    ]
    with (
        patch("lib.zilliz_cli.is_available", return_value=True),
        patch("lib.zilliz_cli.cluster_list", return_value=fake_clusters),
    ):
        data = run_configure(
            from_json=None,
            out_dir=tmp_path,
            overrides={"deployment_target": "zilliz-serverless"},
        )
    assert data["cluster_id"] == "c-serverless-1"
    assert data["resolved_from_cli"] is True
    assert data["target_uri"] == "https://x.zillizcloud.com"


def test_cloud_target_empty_list_falls_back(tmp_path: Path):
    with (
        patch("lib.zilliz_cli.is_available", return_value=True),
        patch("lib.zilliz_cli.cluster_list", return_value=[]),
    ):
        data = run_configure(
            from_json=None,
            out_dir=tmp_path,
            overrides={"deployment_target": "zilliz-serverless"},
        )
    assert "cluster_id" not in data


def test_configure_json_written_to_disk(tmp_path: Path):
    with patch("lib.zilliz_cli.is_available", return_value=False):
        run_configure(from_json=None, out_dir=tmp_path, overrides=None)
    written = json.loads((tmp_path / "configure.json").read_text())
    assert written["deployment_target"] == "local-standalone"
