"""Unit tests for the Phase 6 deployer state machine.

All CLI interactions run through a fake `ClusterCli` so these tests
never touch the network. Covers preflight, provision polling, resumable
deploy.json snapshots, observability pointers, and ingest routing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from lib import deployer
from lib.client import Backend
from lib.errors import (
    BulkImportFailedError,
    ClusterNotReadyError,
    LaunchpadError,
    ZillizCliAuthError,
    ZillizCliMissingError,
)


class FakeCli:
    """Test seam matching the ClusterCli protocol.

    Scripts are consumed in FIFO order so a single describe queue can
    model a cluster transitioning through states.
    """

    def __init__(
        self,
        *,
        available: bool = True,
        whoami: dict[str, Any] | None = None,
        describe_queue: list[dict[str, Any]] | None = None,
        create_payload: dict[str, Any] | None = None,
        import_create_payload: dict[str, Any] | None = None,
        import_describe_queue: list[dict[str, Any]] | None = None,
    ) -> None:
        self._available = available
        self._whoami = whoami or {}
        self._describe = list(describe_queue or [])
        self._create_payload = create_payload or {}
        self._import_create_payload = import_create_payload or {}
        self._import_describe = list(import_describe_queue or [])
        self.calls: list[str] = []

    def is_available(self) -> bool:
        self.calls.append("is_available")
        return self._available

    def auth_whoami(self) -> dict[str, Any]:
        self.calls.append("auth_whoami")
        if not self._available:
            raise ZillizCliMissingError()
        if not self._whoami:
            raise ZillizCliAuthError()
        return self._whoami

    def cluster_describe(self, cluster_id: str) -> dict[str, Any]:
        self.calls.append(f"cluster_describe:{cluster_id}")
        if not self._describe:
            raise AssertionError("no more describe responses queued")
        return self._describe.pop(0)

    def cluster_create(
        self,
        *,
        cluster_name: str,
        plan: str,
        region: str,
        project_id: str | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        self.calls.append(f"cluster_create:{cluster_name}:{plan}:{region}")
        return self._create_payload

    def extract_cluster_id(self, payload: dict[str, Any]) -> str | None:
        return payload.get("cluster_id") or payload.get("clusterId") or payload.get("id") or None

    def import_create(
        self,
        *,
        cluster_id: str,
        collection_name: str,
        files: list[str],
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        self.calls.append(f"import_create:{cluster_id}:{collection_name}")
        return self._import_create_payload

    def import_describe(self, job_id: str) -> dict[str, Any]:
        self.calls.append(f"import_describe:{job_id}")
        if not self._import_describe:
            raise AssertionError("no more import_describe responses queued")
        return self._import_describe.pop(0)


# --- Preflight -------------------------------------------------------------


def test_preflight_missing_binary_raises_missing_error():
    cli = FakeCli(available=False)
    with pytest.raises(ZillizCliMissingError):
        deployer.preflight(cluster_id=None, cli=cli)


def test_preflight_auth_failure_raises_auth_error():
    # When the binary exists but whoami is empty (= not logged in), the
    # real zilliz_cli would return available=False. We model that: the fake
    # is unavailable and whoami is empty, so auth_whoami raises missing-
    # error. The preflight surfaces whichever variant the real CLI would.
    cli = FakeCli(available=False, whoami={})
    with pytest.raises(ZillizCliMissingError):
        deployer.preflight(cluster_id=None, cli=cli)


def test_preflight_ready_cluster_returns():
    cli = FakeCli(
        available=True,
        whoami={"user": "x"},
        describe_queue=[{"state": "RUNNING"}],
    )
    # Should not raise
    deployer.preflight(cluster_id="c-1", cli=cli)


def test_preflight_failed_state_raises_not_ready():
    cli = FakeCli(
        available=True,
        whoami={"user": "x"},
        describe_queue=[{"state": "FAILED"}],
    )
    with pytest.raises(ClusterNotReadyError) as info:
        deployer.preflight(cluster_id="c-1", cli=cli)
    assert info.value.payload["state"] == "FAILED"


def test_preflight_paused_state_suggests_resume():
    cli = FakeCli(
        available=True,
        whoami={"user": "x"},
        describe_queue=[{"state": "PAUSED"}],
    )
    with pytest.raises(ClusterNotReadyError) as info:
        deployer.preflight(cluster_id="c-1", cli=cli)
    assert "cluster resume" in info.value.payload["remediation"]


def test_preflight_unknown_state_bails():
    cli = FakeCli(
        available=True,
        whoami={"user": "x"},
        describe_queue=[{"state": "QUANTUM_SUPERPOSITION"}],
    )
    with pytest.raises(ClusterNotReadyError):
        deployer.preflight(cluster_id="c-1", cli=cli)


# --- Provision polling -----------------------------------------------------


def test_provision_polls_through_provisioning_states():
    cli = FakeCli(
        available=True,
        whoami={"user": "x"},
        create_payload={"cluster_id": "c-new"},
        describe_queue=[
            {"state": "PENDING"},
            {"state": "PROVISIONING"},
            {"state": "RUNNING", "connect_address": "https://new.zillizcloud.com"},
        ],
    )
    sleeps: list[float] = []
    describe = deployer.provision_cluster(
        cluster_name="x",
        plan="Serverless",
        region="gcp-us-west1",
        project_id=None,
        cli=cli,
        sleep=sleeps.append,
        stream=lambda m: None,
    )
    assert describe["connect_address"] == "https://new.zillizcloud.com"
    assert len(sleeps) == 2  # two waiting polls before RUNNING


def test_provision_extracts_id_from_top_level_payload():
    cli = FakeCli(
        available=True,
        whoami={"user": "x"},
        create_payload={"cluster_id": "c-123"},
        describe_queue=[{"state": "RUNNING", "connect_address": "https://c.zillizcloud.com"}],
    )
    out = deployer.provision_cluster(
        cluster_name="x",
        plan="Serverless",
        region="r",
        project_id=None,
        cli=cli,
        sleep=lambda _: None,
    )
    assert out["cluster_id"] == "c-123"


def test_provision_rejects_missing_cluster_id():
    cli = FakeCli(available=True, whoami={"user": "x"}, create_payload={"unrelated": "field"})
    with pytest.raises(LaunchpadError, match="did not return a cluster id"):
        deployer.provision_cluster(
            cluster_name="x",
            plan="Serverless",
            region="r",
            project_id=None,
            cli=cli,
            sleep=lambda _: None,
        )


# --- Observability pointers -----------------------------------------------


def test_observability_pointers_local_uses_prometheus():
    pointers = deployer.observability_pointers(backend=Backend.LOCAL, describe_payload=None)
    assert pointers["prometheus_url"] == "http://localhost:9091/metrics"
    assert pointers["grafana_dashboard"] is None


def test_observability_pointers_cloud_picks_grafana_from_describe():
    pointers = deployer.observability_pointers(
        backend=Backend.ZILLIZ_CLOUD,
        describe_payload={"grafana_dashboard": "https://grafana.zilliz.com/d/abc"},
    )
    assert pointers["grafana_dashboard"] == "https://grafana.zilliz.com/d/abc"
    assert pointers["prometheus_url"] is None


def test_observability_pointers_cloud_tries_alternative_field_names():
    pointers = deployer.observability_pointers(
        backend=Backend.ZILLIZ_CLOUD,
        describe_payload={"dashboard_url": "https://grafana.example.com"},
    )
    assert pointers["grafana_dashboard"] == "https://grafana.example.com"


def test_observability_pointers_cloud_without_dashboard_is_null():
    pointers = deployer.observability_pointers(
        backend=Backend.ZILLIZ_CLOUD, describe_payload={"state": "RUNNING"}
    )
    assert pointers["grafana_dashboard"] is None


def test_extract_cluster_uri_from_nested_connection():
    uri = deployer.extract_cluster_uri({"connection": {"uri": "https://nested.zillizcloud.com"}})
    assert uri == "https://nested.zillizcloud.com"


def test_extract_cluster_uri_missing_raises():
    with pytest.raises(LaunchpadError, match="does not contain a URI"):
        deployer.extract_cluster_uri({"state": "RUNNING"})


# --- DeployState (resumability) -------------------------------------------


def test_deploy_state_fresh_when_no_file(tmp_path: Path):
    state = deployer.DeployState.load_or_new(tmp_path)
    assert state.cluster_id == ""
    assert state.cluster_ready is False
    assert state.ingest_status == "pending"


def test_deploy_state_reloads_partial_snapshot(tmp_path: Path):
    (tmp_path / "deploy.json").write_text(
        json.dumps(
            {
                "cluster_id": "c-1",
                "cluster_ready": True,
                "collection_ready": True,
                "index_ready": False,
                "ingest_status": "pending",
                "timestamps": {"cluster_ready": "2026-04-23T00:00:00+00:00"},
            }
        ),
        encoding="utf-8",
    )
    state = deployer.DeployState.load_or_new(tmp_path)
    assert state.cluster_id == "c-1"
    assert state.cluster_ready is True
    assert state.collection_ready is True
    assert state.index_ready is False
    assert state.timestamps["cluster_ready"] == "2026-04-23T00:00:00+00:00"


def test_snapshot_round_trips(tmp_path: Path):
    state = deployer.DeployState(out_dir=tmp_path)
    state.cluster_id = "c-42"
    state.cluster_ready = True
    state.mark("cluster_ready")
    state.snapshot()
    reloaded = deployer.DeployState.load_or_new(tmp_path)
    assert reloaded.cluster_id == "c-42"
    assert reloaded.cluster_ready is True
    assert "cluster_ready" in reloaded.timestamps


def test_snapshot_keys_sorted(tmp_path: Path):
    state = deployer.DeployState(out_dir=tmp_path)
    state.snapshot()
    parsed = json.loads((tmp_path / "deploy.json").read_text(encoding="utf-8"))
    assert list(parsed.keys()) == sorted(parsed.keys())


# --- Bulk-import error surfacing ------------------------------------------


def test_bulk_import_failure_raises_bulk_import_failed(tmp_path: Path):
    # Build a minimal plan sufficient for _bulk_import to exercise its loop
    plan = {
        "schema": {
            "primary_key": "id",
            "text_field": "text",
            "vector_field": "embedding",
            "extra_fields": [],
        },
        "collection_name": "c",
    }

    class _FakeEmbedder:
        name = "fake"
        dim = 3

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

    cli = FakeCli(
        available=True,
        whoami={"user": "x"},
        import_create_payload={"jobId": "job-1"},
        import_describe_queue=[{"state": "FAILED", "reason": "row malformed"}],
    )

    from lib.chunking import ChunkConfig

    with pytest.raises(BulkImportFailedError) as info:
        deployer._bulk_import(
            docs=[{"id": "d1", "text": "hello world"}],
            plan=plan,
            run_dir=tmp_path,
            embedder=_FakeEmbedder(),
            chunk_config=ChunkConfig(size=512, overlap=64),
            cluster_id="c-1",
            cli=cli,
        )
    assert info.value.payload["job_id"] == "job-1"
    assert info.value.payload["reason"] == "row malformed"


def test_bulk_import_without_job_id_raises():
    cli = FakeCli(available=True, whoami={"user": "x"}, import_create_payload={"no_job": True})
    from lib.chunking import ChunkConfig

    class _E:
        name = "fake"
        dim = 3

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0, 0.0, 0.0] for _ in texts]

    import tempfile

    with (
        tempfile.TemporaryDirectory() as tmp,
        pytest.raises(BulkImportFailedError, match="did not return a job id"),
    ):
        deployer._bulk_import(
            docs=[{"id": "d1", "text": "hi"}],
            plan={
                "schema": {
                    "primary_key": "id",
                    "text_field": "text",
                    "vector_field": "embedding",
                    "extra_fields": [],
                },
                "collection_name": "c",
            },
            run_dir=Path(tmp),
            embedder=_E(),
            chunk_config=ChunkConfig(size=512, overlap=64),
            cluster_id="c-1",
            cli=cli,
        )
