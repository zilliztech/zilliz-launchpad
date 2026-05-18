"""Phase 6: Deploy — promote a local Execute run to Zilliz Cloud.

Replaces the scaffolding stub. Flow:

  1. preflight: zilliz CLI present + authenticated; optionally target
     existing cluster in RUNNING state
  2. provision (when --create): `zilliz cluster create` + poll
  3. recreate collection + index on the target cluster (idempotent)
  4. ingest: client-side or `zilliz import` depending on corpus size
  5. emit deploy.json + observability.json deploy_snapshot
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from pymilvus import MilvusClient as _PyMilvusClient

from .. import zilliz_cli
from ..cli import fail as _cli_fail
from ..client import Backend, MilvusClient, detect_target
from ..credentials import resolve as resolve_credential
from ..deployer import (
    ClusterCli,
    DefaultClusterCli,
    DeployState,
    extract_cluster_uri,
    ingest,
    load_documents,
    observability_pointers,
    preflight,
    provision_cluster,
    recreate_collection_and_index,
)
from ..errors import (
    DestructiveWithoutConfirmError,
    InvalidProfileError,
    LaunchpadError,
)
from ..run_dir import (
    load_configure,
    load_plan,
    preflight_execute_artifact,
    resolve_run_dir,
)

logger = logging.getLogger(__name__)

DEFAULT_CLUSTER_NAME_PREFIX = "launchpad"
DEPLOYMENT_TARGET_PLAN = {
    "zilliz-serverless": "Serverless",
    "zilliz-dedicated": "Standard",
    "zilliz-byoc": "Enterprise",
}


def run_deploy(
    *,
    out_dir: Path,
    cluster_id: str | None,
    create: bool,
    confirm: bool,
    stop_local: bool,
    cli: ClusterCli | None = None,
) -> dict[str, Any]:
    """Entry point for Phase 6.

    Precedence for cluster targeting:
      1. --cluster-id <id>  (explicit)
      2. --create           (provision a new cluster)
      3. configure.json#cluster_id  (reuse from Configure)

    The `cli` argument lets tests inject a fake `ClusterCli`. Production
    callers omit it; the default delegates to the `zilliz_cli` module.
    """
    preflight_execute_artifact(out_dir)
    plan = load_plan(out_dir)
    configure = load_configure(out_dir)
    cli_impl: ClusterCli = cli if cli is not None else DefaultClusterCli()

    resolved_id = _resolve_cluster_precedence(
        cluster_id_flag=cluster_id,
        create_flag=create,
        confirm=confirm,
        configure=configure,
    )

    state = DeployState.load_or_new(out_dir)
    _hydrate_token_source(state)

    if create:
        # Cluster doesn't exist yet; preflight only validates CLI + auth
        preflight(cluster_id=None, cli=cli_impl)
        describe = _provision_new(state=state, configure=configure, plan=plan, cli=cli_impl)
    else:
        preflight(cluster_id=resolved_id, cli=cli_impl)
        describe = cli_impl.cluster_describe(str(resolved_id))

    _record_cluster(state=state, describe=describe, plan=plan)

    # Point the Milvus client at the (now confirmed) cluster URI
    client = MilvusClient(uri=state.cluster_uri)
    target = detect_target(state.cluster_uri)

    _do_collection(state=state, client=client, plan=plan)
    _do_ingest(
        state=state,
        client=client,
        plan=plan,
        out_dir=out_dir,
        target_backend=target.backend,
        cli=cli_impl,
    )

    if stop_local:
        _stop_local_sidecar(out_dir)

    _write_observability_snapshot(out_dir, state)
    return state.to_dict()


# --- Precedence + resolution ----------------------------------------------


def _resolve_cluster_precedence(
    *,
    cluster_id_flag: str | None,
    create_flag: bool,
    confirm: bool,
    configure: dict[str, Any],
) -> str | None:
    if cluster_id_flag and create_flag:
        raise InvalidProfileError(
            pointer="cli",
            reason="pass exactly one of --cluster-id or --create, not both",
        )
    if create_flag:
        if not confirm:
            raise DestructiveWithoutConfirmError(
                action="zilliz cluster create",
                resources=[
                    f"cluster:{DEFAULT_CLUSTER_NAME_PREFIX}-* "
                    f"(region from configure.json / zilliz config)"
                ],
            )
        return None
    if cluster_id_flag:
        return cluster_id_flag
    configured = configure.get("cluster_id")
    if isinstance(configured, str) and configured:
        return configured
    raise InvalidProfileError(
        pointer="cli",
        reason=(
            "no target cluster — pass --cluster-id, --create (with --confirm), "
            "or set cluster_id in configure.json"
        ),
    )


# --- Provisioning ----------------------------------------------------------


def _provision_new(
    *,
    state: DeployState,
    configure: dict[str, Any],
    plan: dict[str, Any],
    cli: ClusterCli,
) -> dict[str, Any]:
    deployment_target = str(configure.get("deployment_target") or "zilliz-serverless")
    plan_name = DEPLOYMENT_TARGET_PLAN.get(deployment_target, "Serverless")
    region = str(configure.get("region") or "")
    if not region:
        raise InvalidProfileError(
            pointer="configure.json#region",
            reason=(
                "--create requires a region. Set configure.json.region "
                "(e.g. 'gcp-us-west1') or run `zilliz config set region <r>` "
                "and re-run Configure."
            ),
        )
    cluster_name = str(configure.get("cluster_name") or _default_cluster_name(plan))
    project_id = configure.get("project_id")
    describe = provision_cluster(
        cluster_name=cluster_name,
        plan=plan_name,
        region=region,
        project_id=project_id if isinstance(project_id, str) else None,
        cli=cli,
    )
    state.cluster_id = str(describe.get("cluster_id") or describe.get("clusterId") or "")
    state.cluster_ready = True
    state.mark("cluster_ready")
    state.snapshot()
    return describe


def _default_cluster_name(plan: dict[str, Any]) -> str:
    coll = str(plan.get("collection_name") or "collection")
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    return f"{DEFAULT_CLUSTER_NAME_PREFIX}-{coll}-{ts}"


# --- Per-step recording ----------------------------------------------------


def _record_cluster(*, state: DeployState, describe: dict[str, Any], plan: dict[str, Any]) -> None:
    if not state.cluster_id:
        state.cluster_id = str(describe.get("cluster_id") or describe.get("clusterId") or "")
    state.cluster_uri = extract_cluster_uri(describe)
    state.cluster_ready = True
    state.mark("cluster_ready")
    state.observability = observability_pointers(
        backend=detect_target(state.cluster_uri).backend,
        describe_payload=describe,
    )
    state.collection_name = str(plan.get("collection_name") or "")
    state.snapshot()


def _do_collection(*, state: DeployState, client: _PyMilvusClient, plan: dict[str, Any]) -> None:
    if state.collection_ready and state.index_ready:
        logger.info("collection + index already recorded ready; skipping")
        return
    result = recreate_collection_and_index(client=client, plan=plan)
    state.collection_ready = True
    state.index_ready = True
    state.mark("collection_ready")
    state.mark("index_ready")
    logger.info("collection=%s index=%s", result.get("collection"), result.get("index"))
    state.snapshot()


def _do_ingest(
    *,
    state: DeployState,
    client: _PyMilvusClient,
    plan: dict[str, Any],
    out_dir: Path,
    target_backend: Backend,
    cli: ClusterCli,
) -> None:
    if state.ingest_status == "complete":
        logger.info("ingest already complete; skipping")
        return
    state.ingest_status = "in_progress"
    state.snapshot()
    threshold = int(plan.get("bulk_import_threshold") or 0)
    try:
        mode, rows = ingest(
            client=client,
            plan=plan,
            docs=load_documents(plan=plan, run_dir=out_dir),
            run_dir=out_dir,
            cluster_id=state.cluster_id or None,
            backend=target_backend,
            bulk_threshold=threshold,
            cli=cli,
        )
    except Exception:
        state.ingest_status = "failed"
        state.snapshot()
        raise
    state.ingest_mode = mode
    state.ingest_row_count = rows
    state.ingest_status = "complete"
    state.mark("ingest_complete")
    state.snapshot()


# --- Observability snapshot ------------------------------------------------


def _write_observability_snapshot(out_dir: Path, state: DeployState) -> None:
    path = out_dir / "observability.json"
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("observability.json is not valid JSON; rewriting")
            data = {}
    snapshots = data.setdefault("deploy_snapshots", [])
    snapshots.append(
        {
            "cluster_id": state.cluster_id,
            "collection": state.collection_name,
            "post_ingest_row_count": state.ingest_row_count,
            "ingest_mode": state.ingest_mode,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    # Also record the pointers so downstream tooling doesn't have to re-read
    # deploy.json
    data.setdefault("pointers", state.observability)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


# --- Credentials ----------------------------------------------------------


def _hydrate_token_source(state: DeployState) -> None:
    """Record where the Cloud token came from (env vs CLI fallback).

    The actual token never lands in deploy.json — just the source tag.
    """
    env_token = resolve_credential("ZILLIZ_TOKEN", optional=True, allow_cli=False)
    if env_token:
        state.token_source = "env:ZILLIZ_TOKEN"
        return
    try:
        zilliz_cli.auth_whoami()
    except LaunchpadError:
        state.token_source = ""
        return
    state.token_source = "cli:whoami"


# --- Local-sidecar teardown (opt-in) --------------------------------------


def _stop_local_sidecar(run_dir: Path) -> None:
    pid_file = run_dir / "sidecar.pid"
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        return
    try:
        import os
        import signal

        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError) as exc:
        logger.warning("could not stop sidecar pid=%s: %s", pid, exc)
    pid_file.unlink(missing_ok=True)


def register(app: typer.Typer) -> None:
    """Attach the Phase 6 ``deploy`` subcommand to the shared app."""

    @app.command()
    def deploy(
        run_dir: str | None = typer.Option(None, "--run-dir"),
        cluster_id: str | None = typer.Option(
            None, "--cluster-id", help="Target an existing RUNNING cluster"
        ),
        create: bool = typer.Option(
            False, "--create", help="Provision a new cluster via `zilliz cluster create`"
        ),
        confirm: bool = typer.Option(
            False, "--confirm", help="Required with --create to acknowledge billing impact"
        ),
        stop_local: bool = typer.Option(
            False, "--stop-local", help="Stop the Phase 4 local UI sidecar after deploy"
        ),
    ) -> None:
        """Phase 6 — promote the Execute run to Zilliz Cloud."""
        out = resolve_run_dir(run_dir)
        try:
            report = run_deploy(
                out_dir=out,
                cluster_id=cluster_id,
                create=create,
                confirm=confirm,
                stop_local=stop_local,
            )
        except LaunchpadError as e:
            _cli_fail(e)
        typer.echo(f"run-dir: {out}")
        typer.echo(f"cluster: {report['cluster_id']} ({report['cluster_uri']})")
        typer.echo(f"collection: {report['collection_name']}")
        typer.echo(
            f"ingest: {report['ingest_mode']} "
            f"({report['ingest_row_count']} rows, status={report['ingest_status']})"
        )
        obs = report.get("observability") or {}
        if obs.get("grafana_dashboard"):
            typer.echo(f"grafana: {obs['grafana_dashboard']}")
        if obs.get("prometheus_url"):
            typer.echo(f"prometheus: {obs['prometheus_url']}")
        typer.echo(f"deploy.json: {out / 'deploy.json'}")
