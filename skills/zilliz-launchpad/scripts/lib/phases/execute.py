"""Phase 4: Execute — apply plan.json to the target backend, start UI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from pymilvus import DataType

from ..chunking import ChunkConfig
from ..client import MilvusClient
from ..embeddings import make_embedder
from ..ingest import ingest_documents
from ..operations import build_basic_schema, create_collection, create_index, load_collection
from ..samples import load as load_sample
from ..search import search_dense

DTYPE_MAP = {"string": (DataType.VARCHAR, 256), "int": (DataType.INT64, None), "float": (DataType.FLOAT, None), "bool": (DataType.BOOL, None)}


def _schema_from_plan(plan_schema: dict[str, Any]):
    extras: list[tuple[str, DataType, int | None]] = []
    for f in plan_schema["extra_fields"]:
        dt, maxlen = DTYPE_MAP.get(f["type"], (DataType.VARCHAR, 256))
        extras.append((f["name"], dt, maxlen if maxlen is not None else f.get("max_length")))
    return build_basic_schema(
        primary_field=plan_schema["primary_key"],
        text_field=plan_schema["text_field"],
        vector_field=plan_schema["vector_field"],
        dim=plan_schema["dim"],
        enable_sparse=plan_schema.get("sparse_field") is not None,
        sparse_field=plan_schema.get("sparse_field") or "sparse",
        extra_fields=extras,
    )


def _iter_documents(plan: dict[str, Any], run_dir: Path, sample: str | None, input_path: str | None):
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


def _start_sidecar(run_dir: Path, plan: dict[str, Any], port: int) -> int | None:
    """Launch the FastAPI sidecar as a background process. Returns PID."""
    env_file = run_dir / "sidecar.env.json"
    env_file.write_text(
        json.dumps(
            {"uri": plan["target_uri"], "collection": plan["collection_name"], "plan_dir": str(run_dir)},
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


def run_execute(
    *,
    out_dir: Path,
    sample: str | None,
    input_path: str | None,
    ui_port: int,
    start_ui: bool,
) -> dict[str, Any]:
    plan = json.loads((out_dir / "plan.json").read_text(encoding="utf-8"))

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
    load_collection(client, plan["collection_name"])

    # Ingest
    embedder = make_embedder(
        plan["embedding"]["provider"],
        plan["embedding"]["model"],
        plan["embedding"]["dim"],
    )
    docs = list(_iter_documents(plan, out_dir, sample, input_path))
    extra_keys = tuple(f["name"] for f in plan["schema"]["extra_fields"])
    stats = ingest_documents(
        client,
        plan["collection_name"],
        docs,
        embedder,
        text_field=plan["schema"]["text_field"],
        id_field=plan["schema"]["primary_key"],
        vector_field=plan["schema"]["vector_field"],
        chunk_config=ChunkConfig(size=plan["chunking"]["size"], overlap=plan["chunking"]["overlap"]),
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

    report = {
        "collection_status": coll_status,
        "index_status": idx_status,
        "ingest": {
            "documents": stats.documents,
            "chunks": stats.chunks,
            "batches": stats.batches,
            "retries": stats.retries,
        },
        "smoke_query": sample_query[:120],
        "smoke_hits": smoke_hits,
        "ui_port": ui_port if start_ui else None,
        "sidecar_pid": sidecar_pid,
    }
    (out_dir / "execute.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report
