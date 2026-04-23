"""Phase 4: Execute — apply plan.json to the target backend, start UI."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pymilvus import CollectionSchema, DataType
from pymilvus import MilvusClient as PyMilvusClient

from .. import zilliz_cli
from ..chunking import ChunkConfig, chunk_text
from ..client import Backend, MilvusClient, detect_target
from ..embeddings import (
    EmbeddingProvider,
    embed_text_with_clip,
    make_embedder,
    make_image_embedder,
)
from ..errors import ClusterNotReadyError, LaunchpadError
from ..ingest import ingest_documents
from ..operations import build_basic_schema, create_collection, create_index, load_collection
from ..samples import load as load_sample
from ..search import search_dense

DTYPE_MAP = {
    "string": (DataType.VARCHAR, 256),
    "int": (DataType.INT64, None),
    "float": (DataType.FLOAT, None),
    "bool": (DataType.BOOL, None),
}

READY_STATE = "RUNNING"
WAIT_STATES = {"PROVISIONING", "MODIFYING"}
FAIL_STATES = {"PAUSED", "DELETING", "FAILED"}
PREFLIGHT_MAX_WAIT_SEC = 60
IMPORT_POLL_CAP_SEC = 30 * 60
IMAGE_BATCH_SIZE = 16
IMAGE_PK_MAX_LENGTH = 512  # absolute paths can be long

logger = logging.getLogger(__name__)


def _schema_from_plan(plan_schema: dict[str, Any]) -> CollectionSchema:
    extras: list[tuple[str, DataType, int | None]] = []
    for f in plan_schema["extra_fields"]:
        dt, maxlen = DTYPE_MAP.get(f["type"], (DataType.VARCHAR, 256))
        extras.append((f["name"], dt, maxlen if maxlen is not None else f.get("max_length")))
    text_field = plan_schema.get("text_field")
    is_image = text_field is None
    return build_basic_schema(
        primary_field=plan_schema["primary_key"],
        text_field=text_field,
        vector_field=plan_schema["vector_field"],
        dim=plan_schema["dim"],
        enable_sparse=plan_schema.get("sparse_field") is not None,
        sparse_field=plan_schema.get("sparse_field") or "sparse",
        primary_max_length=IMAGE_PK_MAX_LENGTH if is_image else 128,
        extra_fields=extras,
    )


def _iter_documents(
    plan: dict[str, Any], run_dir: Path, sample: str | None, input_path: str | None
) -> Iterator[dict[str, Any]]:
    if sample:
        yield from load_sample(sample)
        return
    if input_path:
        path = Path(input_path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    # try to infer from collect.json
    collect = json.loads((run_dir / "collect.json").read_text(encoding="utf-8"))
    src = collect.get("source_sample")
    if src:
        yield from load_sample(src)
        return
    src_path = collect.get("source_path")
    if src_path:
        path = Path(src_path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    raise RuntimeError("No data source — provide --sample or --input, or run collect with one.")


def _cluster_preflight(cluster_id: str) -> dict[str, Any]:
    """Poll `zilliz cluster describe` until RUNNING, giving up on hard states."""
    deadline = time.monotonic() + PREFLIGHT_MAX_WAIT_SEC
    delay = 1.0
    last: dict[str, Any] = {}
    while True:
        last = zilliz_cli.cluster_describe(cluster_id)
        state = str(last.get("state") or last.get("status") or "").upper()
        if state == READY_STATE:
            return last
        if state in FAIL_STATES:
            remediation = (
                f"zilliz cluster resume --cluster-id {cluster_id}"
                if state == "PAUSED"
                else f"Investigate cluster {cluster_id} at https://cloud.zilliz.com"
            )
            raise ClusterNotReadyError(cluster_id=cluster_id, state=state, remediation=remediation)
        if state not in WAIT_STATES:
            raise ClusterNotReadyError(
                cluster_id=cluster_id,
                state=state or "UNKNOWN",
                remediation=f"zilliz cluster describe --cluster-id {cluster_id}",
            )
        if time.monotonic() >= deadline:
            raise ClusterNotReadyError(
                cluster_id=cluster_id,
                state=state,
                remediation=(
                    f"Wait longer or check cluster {cluster_id}; current state {state} "
                    f"did not reach RUNNING within {PREFLIGHT_MAX_WAIT_SEC}s"
                ),
            )
        time.sleep(delay)
        delay = min(delay * 2, 16)


def _start_sidecar(run_dir: Path, plan: dict[str, Any], port: int) -> int | None:
    """Launch the FastAPI sidecar as a background process. Returns PID."""
    env_file = run_dir / "sidecar.env.json"
    env_file.write_text(
        json.dumps(
            {
                "uri": plan["target_uri"],
                "collection": plan["collection_name"],
                "plan_dir": str(run_dir),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log = (run_dir / "sidecar.log").open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "lib.ui:app", "--host", "127.0.0.1", "--port", str(port)],
        env={"LAUNCHPAD_RUN_DIR": str(run_dir), **dict(__import__("os").environ)},
        stdout=log,
        stderr=log,
        cwd=str(Path(__file__).resolve().parent.parent.parent),  # scripts/
    )
    (run_dir / "sidecar.pid").write_text(str(proc.pid), encoding="utf-8")
    return proc.pid


def _should_bulk_import(
    *,
    row_count: int,
    threshold: int,
    target_backend: Backend,
    cluster_id: str | None,
) -> bool:
    if row_count <= threshold:
        return False
    if target_backend is not Backend.ZILLIZ_CLOUD or not cluster_id:
        return False
    try:
        if not zilliz_cli.is_available():
            return False
    except LaunchpadError:
        return False
    return True


def _bulk_import(
    *,
    docs: list[dict[str, Any]],
    plan: dict[str, Any],
    run_dir: Path,
    embedder: EmbeddingProvider,
    chunk_config: ChunkConfig,
    cluster_id: str,
) -> dict[str, Any]:
    """CLI-backed bulk import path.

    Precomputes chunks + embeddings, writes a JSONL with the `embedding`
    column populated, submits a `zilliz import create` job, and polls
    `zilliz import describe` until terminal or the 30-minute cap elapses.
    """
    schema = plan["schema"]
    text_field = schema["text_field"]
    id_field = schema["primary_key"]
    vector_field = schema["vector_field"]
    extra_keys = tuple(f["name"] for f in schema["extra_fields"])

    import hashlib

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
    job = zilliz_cli.import_create(
        cluster_id=cluster_id,
        collection_name=plan["collection_name"],
        files=[str(jsonl_path)],
    )
    job_id = str(job.get("jobId") or job.get("job_id") or job.get("id") or "")
    if not job_id:
        raise LaunchpadError("zilliz import create did not return a job id", payload=job)

    deadline = time.monotonic() + IMPORT_POLL_CAP_SEC
    delay = 2.0
    while True:
        status = zilliz_cli.import_describe(job_id)
        state = str(status.get("state") or status.get("status") or "").upper()
        if state == "DONE":
            return {
                "path": "bulk-import",
                "job_id": job_id,
                "chunks": total_chunks,
                "state": state,
            }
        if state == "FAILED":
            reason = status.get("reason") or status.get("failReason") or "unknown"
            raise LaunchpadError(
                f"zilliz import job {job_id} failed: {reason}",
                job_id=job_id,
                state=state,
            )
        if time.monotonic() >= deadline:
            raise LaunchpadError(
                f"zilliz import job {job_id} did not finish within {IMPORT_POLL_CAP_SEC}s",
                job_id=job_id,
                state=state,
            )
        time.sleep(delay)
        delay = min(delay * 1.5, 30)


def _image_collection_setup(
    plan: dict[str, Any], collect: dict[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    """Validate prereqs for the image branch and return (rows, primary_key)."""
    rows = collect.get("rows") or []
    if not rows:
        raise LaunchpadError(
            "image collection has no rows in collect.json",
            payload={"hint": "rerun `collect --input <dir>` to populate rows[]"},
        )
    pk_field = plan["schema"]["primary_key"]
    return list(rows), pk_field


def _ingest_image_rows(
    *,
    client: PyMilvusClient,
    collection: str,
    rows: list[dict[str, Any]],
    pk_field: str,
    vector_field: str,
    extra_keys: tuple[str, ...],
    embedder_fn: Any,
    out_dir: Path,
    processed: set[str],
) -> tuple[list[str], list[dict[str, str]], int]:
    """Returns (newly_processed_paths, skipped, batches_emitted)."""
    pending = [r for r in rows if str(r.get("image_path") or "") not in processed]
    newly_processed: list[str] = []
    skipped: list[dict[str, str]] = []
    batches = 0
    if not pending:
        return newly_processed, skipped, batches

    for start in range(0, len(pending), IMAGE_BATCH_SIZE):
        batch = pending[start : start + IMAGE_BATCH_SIZE]
        paths = [str(r["image_path"]) for r in batch]
        try:
            vectors = embedder_fn(paths)
        except Exception as exc:
            for p in paths:
                skipped.append({"image_path": p, "reason": f"batch-embed: {exc}"})
            logger.warning("image batch failed (%d files): %s", len(paths), exc)
            continue

        upsert_rows: list[dict[str, Any]] = []
        for r, v in zip(batch, vectors, strict=True):
            row: dict[str, Any] = {pk_field: str(r["image_path"]), vector_field: v}
            for k in extra_keys:
                if k in r:
                    row[k] = r[k]
            upsert_rows.append(row)
        try:
            client.upsert(collection_name=collection, data=upsert_rows)
        except Exception as exc:
            for p in paths:
                skipped.append({"image_path": p, "reason": f"upsert: {exc}"})
            logger.warning("image upsert failed (%d files): %s", len(paths), exc)
            continue
        newly_processed.extend(paths)
        batches += 1

        # Snapshot processed_files after every successful batch so a kill is
        # resumable. We rewrite execute.json so a partial run is observable.
        processed.update(paths)
        snapshot = {
            "phase": "ingesting",
            "processed_files": sorted(processed),
            "skipped_files": skipped,
        }
        (out_dir / "execute.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return newly_processed, skipped, batches


def _run_image_execute(
    *,
    out_dir: Path,
    plan: dict[str, Any],
    ui_port: int,
    start_ui: bool,
) -> dict[str, Any]:
    """Image branch — bypasses chunking, BM25, and bulk-import paths."""
    collect = json.loads((out_dir / "collect.json").read_text(encoding="utf-8"))
    rows, pk_field = _image_collection_setup(plan, collect)

    # Pre-flight credentials before touching Milvus.
    embedding = plan["embedding"]
    provider = str(embedding["provider"])
    model = embedding.get("model")
    device_hint = embedding.get("device_hint")
    embedder_fn = make_image_embedder(provider, model, device_hint)

    # Resume from existing snapshot if present.
    processed: set[str] = set()
    prior_skipped: list[dict[str, str]] = []
    snap_path = out_dir / "execute.json"
    if snap_path.exists():
        try:
            prior = json.loads(snap_path.read_text(encoding="utf-8"))
            processed = set(prior.get("processed_files") or [])
            prior_skipped = list(prior.get("skipped_files") or [])
        except (json.JSONDecodeError, OSError):
            processed = set()

    client = MilvusClient(uri=plan["target_uri"])
    schema = _schema_from_plan(plan["schema"])
    coll_status = create_collection(client, plan["collection_name"], schema)
    idx_status = create_index(
        client,
        plan["collection_name"],
        plan["schema"]["vector_field"],
        index_type=plan["index"]["type"],
        metric_type=plan["index"]["metric"],
        params=plan["index"]["params"],
    )
    load_collection(client, plan["collection_name"])

    extra_keys = tuple(f["name"] for f in plan["schema"]["extra_fields"])
    new_processed, skipped, batches = _ingest_image_rows(
        client=client,
        collection=plan["collection_name"],
        rows=rows,
        pk_field=pk_field,
        vector_field=plan["schema"]["vector_field"],
        extra_keys=extra_keys,
        embedder_fn=embedder_fn,
        out_dir=out_dir,
        processed=processed,
    )
    skipped = prior_skipped + skipped

    # Smoke text query: encode via CLIP if local, else skip (Voyage text path
    # would need an extra API call we don't strictly need at this stage).
    smoke_hits: list[dict[str, Any]] = []
    smoke_query = "a photograph"
    if provider == "clip-local":
        try:
            qvec = embed_text_with_clip(
                [smoke_query],
                model_id=str(model or "ViT-B-32"),
                device_hint=device_hint,
            )[0]
            res = client.search(
                collection_name=plan["collection_name"],
                data=[qvec],
                anns_field=plan["schema"]["vector_field"],
                limit=1,
                output_fields=[pk_field],
            )
            for batch_res in res or []:
                for hit in batch_res:
                    entity = getattr(hit, "entity", None)
                    pk_val = (
                        entity.get(pk_field) if entity is not None else hit.get(pk_field)
                    ) if hasattr(hit, "get") or entity is not None else None
                    smoke_hits.append(
                        {
                            "id": str(pk_val) if pk_val else "",
                            "score": float(getattr(hit, "score", 0.0) or hit.get("distance", 0.0)),
                        }
                    )
        except Exception as exc:
            logger.warning("image smoke query failed: %s", exc)

    sidecar_pid = _start_sidecar(out_dir, plan, ui_port) if start_ui else None

    report = {
        "phase": "complete",
        "collection_status": coll_status,
        "index_status": idx_status,
        "ingest_path": "image-batch",
        "ingest": {
            "images_total": len(rows),
            "images_ingested_this_run": len(new_processed),
            "images_skipped_this_run": len([s for s in skipped if s not in prior_skipped]),
            "batches": batches,
            "provider": provider,
            "model": model,
            "device_hint": device_hint,
        },
        "processed_files": sorted(processed),
        "skipped_files": skipped,
        "smoke_query": smoke_query,
        "smoke_hits": smoke_hits,
        "ui_port": ui_port if start_ui else None,
        "sidecar_pid": sidecar_pid,
    }
    snap_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def run_execute(
    *,
    out_dir: Path,
    sample: str | None,
    input_path: str | None,
    ui_port: int,
    start_ui: bool,
) -> dict[str, Any]:
    plan = json.loads((out_dir / "plan.json").read_text(encoding="utf-8"))

    target = detect_target(plan["target_uri"])
    cluster_id = plan.get("cluster_id")

    # Image branch: short-circuit before any text-flow work.
    if plan["embedding"].get("modality") == "image":
        return _run_image_execute(
            out_dir=out_dir, plan=plan, ui_port=ui_port, start_ui=start_ui
        )

    # Pre-flight when we have a CLI-resolved cluster_id
    preflight: dict[str, Any] | None = None
    if cluster_id and target.backend is Backend.ZILLIZ_CLOUD:
        preflight = _cluster_preflight(str(cluster_id))

    # Connect
    client = MilvusClient(uri=plan["target_uri"])

    # Collection + index
    schema = _schema_from_plan(plan["schema"])
    coll_status = create_collection(client, plan["collection_name"], schema)
    idx_status = create_index(
        client,
        plan["collection_name"],
        plan["schema"]["vector_field"],
        index_type=plan["index"]["type"],
        metric_type=plan["index"]["metric"],
        params=plan["index"]["params"],
    )
    if plan.get("sparse_enabled"):
        create_index(
            client,
            plan["collection_name"],
            plan["schema"]["sparse_field"],
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={},
        )
    load_collection(client, plan["collection_name"])

    # Ingest
    embedder = make_embedder(
        plan["embedding"]["provider"],
        plan["embedding"]["model"],
        plan["embedding"]["dim"],
    )
    docs = list(_iter_documents(plan, out_dir, sample, input_path))
    chunk_config = ChunkConfig(size=plan["chunking"]["size"], overlap=plan["chunking"]["overlap"])
    extra_keys = tuple(f["name"] for f in plan["schema"]["extra_fields"])

    threshold = int(plan.get("bulk_import_threshold") or 0)
    ingest_path = "client"
    import_report: dict[str, Any] | None = None
    if threshold and _should_bulk_import(
        row_count=len(docs),
        threshold=threshold,
        target_backend=target.backend,
        cluster_id=cluster_id,
    ):
        try:
            import_report = _bulk_import(
                docs=docs,
                plan=plan,
                run_dir=out_dir,
                embedder=embedder,
                chunk_config=chunk_config,
                cluster_id=str(cluster_id),
            )
            ingest_path = "bulk-import"
            stats = None
        except LaunchpadError as exc:
            logger.warning("bulk-import failed, falling back to client path: %s", exc)
            ingest_path = "client-fallback"
            stats = ingest_documents(
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
    else:
        if len(docs) > threshold and target.backend is Backend.ZILLIZ_CLOUD:
            logger.info(
                "bulk-import path unavailable (CLI missing or cluster_id not set); "
                "using client-side ingestion for %d rows",
                len(docs),
            )
        stats = ingest_documents(
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

    # Smoke test
    sample_query = docs[0].get(plan["schema"]["text_field"], "") if docs else ""
    smoke_hits = []
    if sample_query:
        smoke_hits = [
            h.to_dict()
            for h in search_dense(
                client,
                plan["collection_name"],
                str(sample_query),
                embedder,
                top_k=1,
                vector_field=plan["schema"]["vector_field"],
                output_fields=[plan["schema"]["text_field"]],
            )
        ]

    sidecar_pid = _start_sidecar(out_dir, plan, ui_port) if start_ui else None

    report: dict[str, Any] = {
        "collection_status": coll_status,
        "index_status": idx_status,
        "ingest_path": ingest_path,
        "ingest": (
            {
                "documents": stats.documents,
                "chunks": stats.chunks,
                "batches": stats.batches,
                "retries": stats.retries,
            }
            if stats is not None
            else import_report or {}
        ),
        "smoke_query": sample_query[:120],
        "smoke_hits": smoke_hits,
        "ui_port": ui_port if start_ui else None,
        "sidecar_pid": sidecar_pid,
    }
    if preflight is not None:
        report["preflight"] = {
            "cluster_id": cluster_id,
            "state": str(preflight.get("state") or preflight.get("status") or ""),
        }
    (out_dir / "execute.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report
