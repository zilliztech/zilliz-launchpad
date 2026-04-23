"""Phase 6 deploy helpers.

The deployer promotes a completed Execute run to Zilliz Cloud. It owns:

  * preflight (CLI present, authenticated, target cluster running)
  * optional cluster provisioning via `zilliz cluster create` with polling
  * idempotent collection + index reuse of the Phase 4 primitives
  * ingest routing (client-side vs `zilliz import`) against the threshold
  * a resumable state machine — `DeployState` persists `deploy.json` after
    every transition so a rerun with `--cluster-id` skips finished steps

All imperative side effects go through injectable seams (`ClusterCli`,
`ingest_fn`) so the state machine is unit-testable without a live CLI.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pymilvus import MilvusClient as _PyMilvusClient

from . import zilliz_cli
from .chunking import ChunkConfig
from .client import Backend
from .embeddings import EmbeddingProvider, make_embedder
from .errors import (
    BulkImportFailedError,
    ClusterNotReadyError,
    LaunchpadError,
    ZillizCliAuthError,
    ZillizCliMissingError,
)
from .ingest import IngestStats, ingest_documents
from .operations import build_basic_schema, create_collection, create_index, load_collection

logger = logging.getLogger(__name__)

READY_STATE = "RUNNING"
WAIT_STATES = {"PROVISIONING", "MODIFYING", "CREATING", "PENDING"}
FAIL_STATES = {"PAUSED", "DELETING", "FAILED", "ERROR"}
PROVISION_MAX_WAIT_SEC = 15 * 60
PREFLIGHT_MAX_WAIT_SEC = 60
IMPORT_POLL_CAP_SEC = 30 * 60
GRAFANA_DASHBOARD_FIELDS = ("grafana_dashboard", "grafanaDashboard", "dashboard_url")
URI_FIELDS = ("connect_address", "connectAddress", "uri", "URI", "endpoint")


# --- Injectable seams ------------------------------------------------------


class ClusterCli(Protocol):
    """Minimal surface of `zilliz_cli` the deployer uses.

    Tests pass a fake matching this shape; production code uses the module
    as-is via `DefaultClusterCli`.
    """

    def is_available(self) -> bool: ...
    def auth_whoami(self) -> dict[str, Any]: ...
    def cluster_describe(self, cluster_id: str) -> dict[str, Any]: ...
    def cluster_create(
        self,
        *,
        cluster_name: str,
        plan: str,
        region: str,
        project_id: str | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]: ...
    def extract_cluster_id(self, payload: dict[str, Any]) -> str | None: ...
    def import_create(
        self,
        *,
        cluster_id: str,
        collection_name: str,
        files: list[str],
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]: ...
    def import_describe(self, job_id: str) -> dict[str, Any]: ...


class DefaultClusterCli:
    """Thin adapter delegating to the `zilliz_cli` module."""

    def is_available(self) -> bool:
        return zilliz_cli.is_available()

    def auth_whoami(self) -> dict[str, Any]:
        return zilliz_cli.auth_whoami()

    def cluster_describe(self, cluster_id: str) -> dict[str, Any]:
        return zilliz_cli.cluster_describe(cluster_id)

    def cluster_create(
        self,
        *,
        cluster_name: str,
        plan: str,
        region: str,
        project_id: str | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return zilliz_cli.cluster_create(
            cluster_name=cluster_name,
            plan=plan,
            region=region,
            project_id=project_id,
            extra=extra,
        )

    def extract_cluster_id(self, payload: dict[str, Any]) -> str | None:
        return zilliz_cli.extract_cluster_id(payload)

    def import_create(
        self,
        *,
        cluster_id: str,
        collection_name: str,
        files: list[str],
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return zilliz_cli.import_create(
            cluster_id=cluster_id,
            collection_name=collection_name,
            files=files,
            extra=extra,
        )

    def import_describe(self, job_id: str) -> dict[str, Any]:
        return zilliz_cli.import_describe(job_id)


# --- Deploy state (resumable) ----------------------------------------------


@dataclass
class DeployState:
    """Tracks every transition; snapshot-to-disk on each mutation.

    The state dataclass is the single source of truth for `deploy.json`.
    On any exception, the caller surfaces the error but the partial state
    is already persisted, so a rerun with `--cluster-id <id>` can resume.
    """

    out_dir: Path
    cluster_id: str = ""
    cluster_uri: str = ""
    token_source: str = ""
    collection_name: str = ""
    ingest_mode: str = ""  # "bulk" | "client"
    ingest_row_count: int = 0
    ingest_status: str = "pending"  # pending | in_progress | complete | failed
    cluster_ready: bool = False
    collection_ready: bool = False
    index_ready: bool = False
    observability: dict[str, Any] = field(
        default_factory=lambda: {
            "prometheus_url": None,
            "grafana_dashboard": None,
            "query_log_sample_path": None,
        }
    )
    timestamps: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load_or_new(cls, out_dir: Path) -> DeployState:
        path = out_dir / "deploy.json"
        if not path.exists():
            return cls(out_dir=out_dir)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            out_dir=out_dir,
            cluster_id=str(data.get("cluster_id") or ""),
            cluster_uri=str(data.get("cluster_uri") or ""),
            token_source=str(data.get("token_source") or ""),
            collection_name=str(data.get("collection_name") or ""),
            ingest_mode=str(data.get("ingest_mode") or ""),
            ingest_row_count=int(data.get("ingest_row_count") or 0),
            ingest_status=str(data.get("ingest_status") or "pending"),
            cluster_ready=bool(data.get("cluster_ready")),
            collection_ready=bool(data.get("collection_ready")),
            index_ready=bool(data.get("index_ready")),
            observability=dict(data.get("observability") or {}),
            timestamps=dict(data.get("timestamps") or {}),
        )

    def mark(self, resource: str) -> None:
        """Stamp a transition with an ISO-8601 timestamp."""
        self.timestamps[resource] = datetime.now(UTC).isoformat(timespec="seconds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "cluster_ready": self.cluster_ready,
            "cluster_uri": self.cluster_uri,
            "collection_name": self.collection_name,
            "collection_ready": self.collection_ready,
            "index_ready": self.index_ready,
            "ingest_mode": self.ingest_mode,
            "ingest_row_count": self.ingest_row_count,
            "ingest_status": self.ingest_status,
            "observability": self.observability,
            "timestamps": self.timestamps,
            "token_source": self.token_source,
        }

    def snapshot(self) -> None:
        path = self.out_dir / "deploy.json"
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


# --- Preflight -------------------------------------------------------------


def preflight(
    *,
    cluster_id: str | None,
    cli: ClusterCli,
) -> None:
    """Ensure the CLI is installed, authenticated, and target cluster is ready.

    When `cluster_id` is None the RUNNING check is skipped — the caller is
    about to provision a new cluster via `provision_cluster`.
    """
    if not cli.is_available():
        # Ambiguous between missing binary and not logged in. The underlying
        # call path raises the precise error, so re-invoke through whoami
        # which discriminates. `auth_whoami` throws the right variant.
        cli.auth_whoami()  # raises ZillizCliMissingError / ZillizCliAuthError
        # If somehow `is_available()` returned False but whoami passed,
        # fail closed.
        raise ZillizCliAuthError("zilliz CLI is not available despite auth success")

    if cluster_id is None:
        return

    state = _poll_until_ready(cli, cluster_id, max_wait_sec=PREFLIGHT_MAX_WAIT_SEC)
    if state.get("state", "").upper() != READY_STATE:
        raise ClusterNotReadyError(
            cluster_id=cluster_id,
            state=str(state.get("state") or "UNKNOWN"),
            remediation=f"zilliz cluster describe --cluster-id {cluster_id}",
        )


# --- Cluster provisioning --------------------------------------------------


def provision_cluster(
    *,
    cluster_name: str,
    plan: str,
    region: str,
    project_id: str | None,
    cli: ClusterCli,
    sleep: Callable[[float], None] = time.sleep,
    stream: Callable[[str], None] = lambda msg: print(msg, file=__import__("sys").stderr),
) -> dict[str, Any]:
    """Create a cluster and poll until it's RUNNING.

    Returns the describe payload of the fully-provisioned cluster so the
    caller can extract the connection URI and Grafana dashboard pointer.
    `sleep` and `stream` are injectable for deterministic tests.
    """
    create_payload = cli.cluster_create(
        cluster_name=cluster_name,
        plan=plan,
        region=region,
        project_id=project_id,
        extra=None,
    )
    cluster_id = cli.extract_cluster_id(create_payload)
    if not cluster_id:
        raise LaunchpadError(
            "zilliz cluster create did not return a cluster id",
            payload=create_payload,
        )
    stream(f"[deploy] cluster {cluster_id} requested — waiting for RUNNING")
    describe = _poll_until_ready(
        cli,
        cluster_id,
        max_wait_sec=PROVISION_MAX_WAIT_SEC,
        sleep=sleep,
        stream=stream,
    )
    describe.setdefault("cluster_id", cluster_id)
    return describe


def _poll_until_ready(
    cli: ClusterCli,
    cluster_id: str,
    *,
    max_wait_sec: int,
    sleep: Callable[[float], None] = time.sleep,
    stream: Callable[[str], None] = lambda msg: None,
) -> dict[str, Any]:
    """Poll `cluster describe` with exponential backoff until RUNNING.

    Raises `ClusterNotReadyError` on FAIL_STATES or timeout.
    """
    deadline = time.monotonic() + max_wait_sec
    delay = 2.0
    while True:
        payload = cli.cluster_describe(cluster_id)
        state = str(payload.get("state") or payload.get("status") or "").upper()
        if state == READY_STATE:
            return payload
        if state in FAIL_STATES:
            raise ClusterNotReadyError(
                cluster_id=cluster_id,
                state=state,
                remediation=(
                    f"zilliz cluster resume --cluster-id {cluster_id}"
                    if state == "PAUSED"
                    else f"Investigate cluster {cluster_id} at https://cloud.zilliz.com"
                ),
            )
        if state not in WAIT_STATES and state:
            raise ClusterNotReadyError(
                cluster_id=cluster_id,
                state=state,
                remediation=f"zilliz cluster describe --cluster-id {cluster_id}",
            )
        if time.monotonic() >= deadline:
            raise ClusterNotReadyError(
                cluster_id=cluster_id,
                state=state or "UNKNOWN",
                remediation=(
                    f"Cluster {cluster_id} did not reach RUNNING within "
                    f"{max_wait_sec}s (last state: {state or 'UNKNOWN'})"
                ),
            )
        stream(f"[deploy] cluster {cluster_id} state={state or '?'}, waiting {delay:.0f}s")
        sleep(delay)
        delay = min(delay * 1.5, 30.0)


# --- Observability pointers ------------------------------------------------


def observability_pointers(
    *,
    backend: Backend,
    describe_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the `observability` block for `deploy.json`.

    Standalone targets get a local Prometheus scrape URL. Cloud targets get
    the pre-baked Grafana dashboard URL the describe payload carries (CLI
    versions name the field differently; we probe a handful).
    """
    pointers: dict[str, Any] = {
        "prometheus_url": None,
        "grafana_dashboard": None,
        "query_log_sample_path": "observability.json",
    }
    if backend is Backend.LOCAL:
        pointers["prometheus_url"] = "http://localhost:9091/metrics"
        return pointers
    if describe_payload:
        for field_name in GRAFANA_DASHBOARD_FIELDS:
            val = describe_payload.get(field_name)
            if isinstance(val, str) and val:
                pointers["grafana_dashboard"] = val
                break
    return pointers


def extract_cluster_uri(describe_payload: dict[str, Any]) -> str:
    for field_name in URI_FIELDS:
        val = describe_payload.get(field_name)
        if isinstance(val, str) and val:
            return val
    # Nested under `connection` in some CLI versions
    conn = describe_payload.get("connection")
    if isinstance(conn, dict):
        for field_name in URI_FIELDS:
            val = conn.get(field_name)
            if isinstance(val, str) and val:
                return val
    raise LaunchpadError(
        "zilliz cluster describe payload does not contain a URI",
        payload_keys=sorted(describe_payload.keys()),
    )


# --- Collection + index ----------------------------------------------------


def recreate_collection_and_index(
    *,
    client: _PyMilvusClient,
    plan: dict[str, Any],
) -> dict[str, str]:
    """Idempotent: reuses create_collection + create_index from `operations`.

    Returns a report of what was `created` / `reused` / `rebuilt` per the
    existing `operations.py` contract. Raises `SchemaConflictError` when
    the cluster already carries an incompatible collection (that error is
    already part of the spec-facing envelope contract).
    """
    # Lazy import — `phases.execute` transitively imports this module
    from .phases.execute import DTYPE_MAP

    schema_spec = plan["schema"]
    extras: list[tuple[str, Any, int | None]] = []
    for f in schema_spec["extra_fields"]:
        dt, maxlen = DTYPE_MAP.get(f["type"], (None, 256))
        if dt is None:
            continue
        extras.append((f["name"], dt, maxlen if maxlen is not None else f.get("max_length")))
    schema = build_basic_schema(
        primary_field=schema_spec["primary_key"],
        text_field=schema_spec["text_field"],
        vector_field=schema_spec["vector_field"],
        dim=schema_spec["dim"],
        enable_sparse=schema_spec.get("sparse_field") is not None,
        sparse_field=schema_spec.get("sparse_field") or "sparse",
        extra_fields=extras,
    )
    coll_status = create_collection(client, plan["collection_name"], schema)
    idx_status = create_index(
        client,
        plan["collection_name"],
        schema_spec["vector_field"],
        index_type=plan["index"]["type"],
        metric_type=plan["index"]["metric"],
        params=plan["index"]["params"],
    )
    if plan.get("sparse_enabled"):
        create_index(
            client,
            plan["collection_name"],
            schema_spec["sparse_field"],
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={},
        )
    load_collection(client, plan["collection_name"])
    return {"collection": coll_status, "index": idx_status}


# --- Ingest routing --------------------------------------------------------


def ingest(
    *,
    client: _PyMilvusClient,
    plan: dict[str, Any],
    docs: Iterable[dict[str, Any]],
    run_dir: Path,
    cluster_id: str | None,
    backend: Backend,
    bulk_threshold: int,
    cli: ClusterCli,
) -> tuple[str, int]:
    """Route to bulk import (Cloud + CLI + above threshold) or client upsert.

    Returns `(mode, row_count)` where mode is "bulk" or "client" (or
    "client-fallback" when bulk was tried and failed). The row count is
    the number of *source documents* ingested; chunks are an internal
    detail of the ingest path.
    """
    doc_list = list(docs)
    embedder = make_embedder(
        plan["embedding"]["provider"],
        plan["embedding"]["model"],
        plan["embedding"]["dim"],
    )
    chunk_config = ChunkConfig(size=plan["chunking"]["size"], overlap=plan["chunking"]["overlap"])
    extra_keys = tuple(f["name"] for f in plan["schema"]["extra_fields"])

    use_bulk = (
        len(doc_list) > bulk_threshold
        and backend is Backend.ZILLIZ_CLOUD
        and cluster_id is not None
    )
    if use_bulk:
        try:
            if not cli.is_available():
                raise ZillizCliMissingError()
            _bulk_import(
                docs=doc_list,
                plan=plan,
                run_dir=run_dir,
                embedder=embedder,
                chunk_config=chunk_config,
                cluster_id=str(cluster_id),
                cli=cli,
            )
            return "bulk", len(doc_list)
        except (LaunchpadError, ZillizCliMissingError) as exc:
            logger.warning("bulk import unavailable/failed, falling back to client: %s", exc)
            # Fall through to client path below

    stats = _client_ingest(
        client=client,
        plan=plan,
        docs=doc_list,
        embedder=embedder,
        chunk_config=chunk_config,
        extra_keys=extra_keys,
    )
    return "client", stats.documents


def _client_ingest(
    *,
    client: _PyMilvusClient,
    plan: dict[str, Any],
    docs: list[dict[str, Any]],
    embedder: EmbeddingProvider,
    chunk_config: ChunkConfig,
    extra_keys: tuple[str, ...],
) -> IngestStats:
    return ingest_documents(
        client,
        plan["collection_name"],
        docs,
        embedder,
        text_field=plan["schema"]["text_field"],
        id_field=plan["schema"]["primary_key"],
        vector_field=plan["schema"]["vector_field"],
        chunk_config=chunk_config,
        extra_field_keys=extra_keys,
    )


def _bulk_import(
    *,
    docs: list[dict[str, Any]],
    plan: dict[str, Any],
    run_dir: Path,
    embedder: EmbeddingProvider,
    chunk_config: ChunkConfig,
    cluster_id: str,
    cli: ClusterCli,
) -> None:
    """Precompute chunks+embeddings, submit to `zilliz import`, poll to done.

    Lifted from the Phase 4 bulk path; centralising here lets Phase 6 use
    the same file format without a circular import.
    """
    import hashlib

    from .chunking import chunk_text

    schema = plan["schema"]
    text_field = schema["text_field"]
    id_field = schema["primary_key"]
    vector_field = schema["vector_field"]
    extra_keys = tuple(f["name"] for f in schema["extra_fields"])

    def _pk(source_id: str, idx: int) -> str:
        return hashlib.sha256(f"{source_id}::{idx}".encode()).hexdigest()[:32]

    jsonl_path = run_dir / "bulk_import.jsonl"
    total_chunks = 0
    with jsonl_path.open("w", encoding="utf-8") as out:
        batch_texts: list[str] = []
        batch_rows: list[dict[str, Any]] = []
        for doc_idx, doc in enumerate(docs):
            source_id = str(doc.get(id_field) or f"_synth_{doc_idx}")
            text = str(doc.get(text_field) or "").strip()
            if not text:
                continue
            for idx, chunk in enumerate(chunk_text(text, chunk_config)):
                row: dict[str, Any] = {
                    id_field: _pk(source_id, idx),
                    text_field: chunk,
                }
                for k in extra_keys:
                    if k in doc:
                        row[k] = doc[k]
                batch_texts.append(chunk)
                batch_rows.append(row)
                total_chunks += 1
                if len(batch_rows) >= 64:
                    for r, v in zip(batch_rows, embedder.embed(batch_texts), strict=True):
                        r[vector_field] = v
                        out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    batch_texts.clear()
                    batch_rows.clear()
        if batch_rows:
            for r, v in zip(batch_rows, embedder.embed(batch_texts), strict=True):
                r[vector_field] = v
                out.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("bulk-import: wrote %d chunks to %s", total_chunks, jsonl_path)
    job = cli.import_create(
        cluster_id=cluster_id,
        collection_name=plan["collection_name"],
        files=[str(jsonl_path)],
        extra=None,
    )
    job_id = str(job.get("jobId") or job.get("job_id") or job.get("id") or "") or None
    if not job_id:
        raise BulkImportFailedError(
            job_id=None,
            reason="zilliz import create did not return a job id",
        )

    deadline = time.monotonic() + IMPORT_POLL_CAP_SEC
    delay = 2.0
    while True:
        status = cli.import_describe(job_id)
        state = str(status.get("state") or status.get("status") or "").upper()
        if state == "DONE":
            return
        if state == "FAILED":
            reason = str(status.get("reason") or status.get("failReason") or "unknown")
            raise BulkImportFailedError(job_id=job_id, reason=reason)
        if time.monotonic() >= deadline:
            raise BulkImportFailedError(
                job_id=job_id,
                reason=f"job did not finish within {IMPORT_POLL_CAP_SEC}s (last state: {state})",
            )
        time.sleep(delay)
        delay = min(delay * 1.5, 30)


# --- Document source -------------------------------------------------------


def load_documents(*, plan: dict[str, Any], run_dir: Path) -> Iterable[dict[str, Any]]:
    """Reuse Phase 4's document iterator so Phase 6 sees the same corpus."""
    from .phases.execute import _iter_documents

    return _iter_documents(plan=plan, run_dir=run_dir, sample=None, input_path=None)
