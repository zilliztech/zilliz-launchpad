"""Tests for the Phase 6 driver — precedence, guards, observability snapshot.

Interactions with Milvus + `zilliz` CLI are mocked via a fake `ClusterCli`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from lib.errors import DestructiveWithoutConfirmError, InvalidProfileError
from lib.phases.deploy import _resolve_cluster_precedence, _write_observability_snapshot


def test_precedence_rejects_cluster_id_and_create_together():
    with pytest.raises(InvalidProfileError, match="exactly one of"):
        _resolve_cluster_precedence(
            cluster_id_flag="c-1", create_flag=True, confirm=True, configure={}
        )


def test_precedence_requires_confirm_with_create():
    with pytest.raises(DestructiveWithoutConfirmError) as info:
        _resolve_cluster_precedence(
            cluster_id_flag=None, create_flag=True, confirm=False, configure={}
        )
    assert info.value.payload["action"] == "zilliz cluster create"


def test_precedence_create_with_confirm_returns_none():
    # None means "about to provision a new cluster"
    resolved = _resolve_cluster_precedence(
        cluster_id_flag=None, create_flag=True, confirm=True, configure={}
    )
    assert resolved is None


def test_precedence_prefers_explicit_cluster_id_flag():
    resolved = _resolve_cluster_precedence(
        cluster_id_flag="c-flag",
        create_flag=False,
        confirm=False,
        configure={"cluster_id": "c-configure"},
    )
    assert resolved == "c-flag"


def test_precedence_falls_back_to_configure_cluster_id():
    resolved = _resolve_cluster_precedence(
        cluster_id_flag=None,
        create_flag=False,
        confirm=False,
        configure={"cluster_id": "c-configure"},
    )
    assert resolved == "c-configure"


def test_precedence_no_target_raises():
    with pytest.raises(InvalidProfileError, match="no target cluster"):
        _resolve_cluster_precedence(
            cluster_id_flag=None, create_flag=False, confirm=False, configure={}
        )


# --- observability.json snapshot -----------------------------------------


def test_observability_snapshot_appends_deploy_entry(tmp_path: Path):
    from lib.deployer import DeployState

    state = DeployState(out_dir=tmp_path)
    state.cluster_id = "c-1"
    state.collection_name = "coll"
    state.ingest_mode = "client"
    state.ingest_row_count = 42
    state.observability = {
        "prometheus_url": None,
        "grafana_dashboard": "https://grafana/x",
        "query_log_sample_path": "observability.json",
    }

    _write_observability_snapshot(tmp_path, state)
    data = json.loads((tmp_path / "observability.json").read_text(encoding="utf-8"))
    assert len(data["deploy_snapshots"]) == 1
    snap = data["deploy_snapshots"][0]
    assert snap["cluster_id"] == "c-1"
    assert snap["post_ingest_row_count"] == 42
    assert snap["ingest_mode"] == "client"
    assert data["pointers"]["grafana_dashboard"] == "https://grafana/x"


def test_observability_snapshot_extends_existing_file(tmp_path: Path):
    from lib.deployer import DeployState

    (tmp_path / "observability.json").write_text(
        json.dumps(
            {
                "deploy_snapshots": [{"cluster_id": "c-old"}],
                "latency_samples": [{"source": "evaluate"}],
            }
        ),
        encoding="utf-8",
    )
    state = DeployState(out_dir=tmp_path)
    state.cluster_id = "c-new"
    _write_observability_snapshot(tmp_path, state)
    data = json.loads((tmp_path / "observability.json").read_text(encoding="utf-8"))
    assert len(data["deploy_snapshots"]) == 2
    assert data["deploy_snapshots"][0]["cluster_id"] == "c-old"
    assert data["deploy_snapshots"][1]["cluster_id"] == "c-new"
    # latency_samples from an earlier evaluate run must survive
    assert data["latency_samples"] == [{"source": "evaluate"}]


def test_observability_snapshot_recovers_from_corrupt_file(tmp_path: Path):
    (tmp_path / "observability.json").write_text("{not valid json", encoding="utf-8")
    from lib.deployer import DeployState

    state = DeployState(out_dir=tmp_path)
    state.cluster_id = "c-new"
    _write_observability_snapshot(tmp_path, state)
    # Rewrites with a valid shape rather than crashing
    data = json.loads((tmp_path / "observability.json").read_text(encoding="utf-8"))
    assert data["deploy_snapshots"][0]["cluster_id"] == "c-new"
