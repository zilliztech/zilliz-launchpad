"""FastAPI sidecar for the demo UI.

Environment:
  LAUNCHPAD_RUN_DIR: path to the active run directory (contains plan.json)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .client import MilvusClient
from .embeddings import embed_image_batch, embed_text_with_clip, make_embedder
from .errors import (
    ImageDecodeError,
    LaunchpadError,
    SparseUnavailable,
    UnsupportedImageProviderError,
)
from .search import search_dense, search_hybrid, search_image_to_image, search_sparse

MAX_IMAGE_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB cap documented in design.md
_SEARCH_IMAGE_TOP_K_MAX = 100

app = FastAPI(title="zilliz-launchpad sidecar")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _run_dir() -> Path:
    run_dir = os.environ.get("LAUNCHPAD_RUN_DIR")
    if not run_dir:
        raise RuntimeError("LAUNCHPAD_RUN_DIR is not set")
    return Path(run_dir)


def _load_plan() -> dict[str, Any]:
    plan_path = _run_dir() / "plan.json"
    data: dict[str, Any] = json.loads(plan_path.read_text(encoding="utf-8"))
    return data


def _load_thumbnail_index() -> dict[str, dict[str, Any]]:
    """Map image_path → row dict (with thumbnail_b64 and metadata) for image runs.

    Returns an empty dict for text runs or when collect.json is missing.
    """
    collect_path = _run_dir() / "collect.json"
    if not collect_path.exists():
        return {}
    try:
        collect = json.loads(collect_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if collect.get("data_shape") != "image_dir":
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in collect.get("rows") or []:
        path = row.get("image_path")
        if isinstance(path, str):
            out[path] = row
    return out


_plan_cache: dict[str, Any] | None = None
_client_cache: Any = None
_embedder_cache: Any = None
_thumb_cache: dict[str, dict[str, Any]] | None = None


def _plan() -> dict[str, Any]:
    global _plan_cache
    if _plan_cache is None:
        _plan_cache = _load_plan()
    return _plan_cache


def _client() -> Any:
    global _client_cache
    if _client_cache is None:
        _client_cache = MilvusClient(uri=_plan()["target_uri"])
    return _client_cache


def _is_image() -> bool:
    return bool(_plan().get("embedding", {}).get("modality") == "image")


def _embedder() -> Any:
    """Text embedder for the text-flow (image flow uses embed_text_with_clip)."""
    global _embedder_cache
    if _embedder_cache is None:
        p = _plan()["embedding"]
        _embedder_cache = make_embedder(p["provider"], p["model"], p["dim"])
    return _embedder_cache


def _thumbnails() -> dict[str, dict[str, Any]]:
    global _thumb_cache
    if _thumb_cache is None:
        _thumb_cache = _load_thumbnail_index()
    return _thumb_cache


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    mode: Literal["dense", "sparse", "hybrid"] = "dense"


class Hit(BaseModel):
    id: str
    score: float
    fields: dict[str, Any]


class SearchResponse(BaseModel):
    mode: str
    modality: Literal["text", "image"]
    hits: list[Hit]


class InfoResponse(BaseModel):
    collection_name: str
    modality: Literal["text", "image"]
    primary_key: str
    vector_field: str
    sparse_enabled: bool
    embedding: dict[str, Any]
    has_thumbnails: bool


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "collection": _plan()["collection_name"]}


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    plan = _plan()
    return InfoResponse(
        collection_name=plan["collection_name"],
        modality="image" if _is_image() else "text",
        primary_key=plan["schema"]["primary_key"],
        vector_field=plan["schema"]["vector_field"],
        sparse_enabled=bool(plan.get("sparse_enabled")),
        embedding={
            "provider": plan["embedding"]["provider"],
            "model": plan["embedding"]["model"],
            "dim": plan["embedding"]["dim"],
        },
        has_thumbnails=bool(_thumbnails()),
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    if _is_image():
        return _image_search(req)

    plan = _plan()
    client = _client()
    embedder = _embedder()
    text_field = plan["schema"]["text_field"]
    try:
        if req.mode == "dense":
            hits = search_dense(
                client,
                plan["collection_name"],
                req.query,
                embedder,
                top_k=req.top_k,
                vector_field=plan["schema"]["vector_field"],
                output_fields=[text_field],
            )
        elif req.mode == "sparse":
            hits = search_sparse(
                client,
                plan["collection_name"],
                req.query,
                top_k=req.top_k,
                sparse_field=plan["schema"].get("sparse_field") or "sparse",
                output_fields=[text_field],
            )
        else:
            hits = search_hybrid(
                client,
                plan["collection_name"],
                req.query,
                embedder,
                top_k=req.top_k,
                vector_field=plan["schema"]["vector_field"],
                sparse_field=plan["schema"].get("sparse_field") or "sparse",
                output_fields=[text_field],
            )
    except SparseUnavailable as e:
        raise HTTPException(status_code=400, detail=e.to_dict()) from e
    except LaunchpadError as e:
        raise HTTPException(status_code=400, detail=e.to_dict()) from e

    return SearchResponse(
        mode=req.mode,
        modality="text",
        hits=[Hit(id=h.id, score=h.score, fields=h.fields) for h in hits],
    )


class EmbedImageResponse(BaseModel):
    embedding: list[float]
    dim: int


def _require_image_collection() -> None:
    if not _is_image():
        raise HTTPException(
            status_code=400,
            detail={
                "code": "not_image_collection",
                "message": "This endpoint requires an image collection; active plan is text-only.",
            },
        )


def _read_upload(file: UploadFile) -> bytes:
    """Read an UploadFile into memory, enforcing the 10 MB cap.

    We buffer in 1 MB chunks and bail the moment the running total would
    exceed the limit — keeps a hostile 10 GB upload from pinning memory.
    """
    buf = bytearray()
    chunk_size = 1 << 20
    while True:
        chunk = file.file.read(chunk_size)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > MAX_IMAGE_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "upload_too_large",
                    "message": f"Upload exceeds {MAX_IMAGE_UPLOAD_BYTES} byte cap",
                    "max_bytes": MAX_IMAGE_UPLOAD_BYTES,
                },
            )
    return bytes(buf)


def _encode_upload(image_bytes: bytes) -> list[float]:
    """Encode via the already-loaded CLIP model (or Voyage for that plan)."""
    plan = _plan()
    embedding = plan.get("embedding") or {}
    provider = str(embedding.get("provider") or "").lower()
    if provider == "clip-local":
        import tempfile

        model_id = str(embedding.get("model") or "ViT-B-32")
        device_hint = embedding.get("device_hint")
        with tempfile.NamedTemporaryFile(suffix=".img", delete=True) as tmp:
            tmp.write(image_bytes)
            tmp.flush()
            try:
                vecs = embed_image_batch([tmp.name], model_id=model_id, device_hint=device_hint)
            except Exception as exc:
                raise ImageDecodeError(str(exc)) from exc
        if not vecs:
            raise ImageDecodeError("encoder returned no vector")
        return list(vecs[0])
    # Voyage + any other image-capable provider share the path via search.py;
    # delegate there so the provider-routing logic stays in one place.
    from .search import _encode_query_image

    return _encode_query_image(image_bytes, plan)


@app.post("/embed_image", response_model=EmbedImageResponse)
def embed_image(file: UploadFile = File(...)) -> EmbedImageResponse:  # noqa: B008
    _require_image_collection()
    raw = _read_upload(file)
    try:
        vec = _encode_upload(raw)
    except UnsupportedImageProviderError as e:
        raise HTTPException(status_code=400, detail=e.to_dict()) from e
    except ImageDecodeError as e:
        raise HTTPException(status_code=400, detail=e.to_dict()) from e
    except LaunchpadError as e:
        raise HTTPException(status_code=400, detail=e.to_dict()) from e
    return EmbedImageResponse(embedding=vec, dim=len(vec))


@app.post("/search_image", response_model=SearchResponse)
def search_image(
    file: UploadFile = File(...),  # noqa: B008
    top_k: int = Form(default=10),  # noqa: B008
) -> SearchResponse:
    _require_image_collection()
    k = max(1, min(_SEARCH_IMAGE_TOP_K_MAX, int(top_k)))
    raw = _read_upload(file)

    plan = _plan()
    client = _client()
    pk = plan["schema"]["primary_key"]
    vector_field = plan["schema"]["vector_field"]

    try:
        hits = search_image_to_image(
            client,
            plan["collection_name"],
            raw,
            plan,
            top_k=k,
            vector_field=vector_field,
            output_fields=[pk],
        )
    except UnsupportedImageProviderError as e:
        raise HTTPException(status_code=400, detail=e.to_dict()) from e
    except ImageDecodeError as e:
        raise HTTPException(status_code=400, detail=e.to_dict()) from e
    except LaunchpadError as e:
        raise HTTPException(status_code=400, detail=e.to_dict()) from e

    thumbs = _thumbnails()
    out: list[Hit] = []
    for h in hits:
        path = str(h.fields.get(pk, h.id))
        row = thumbs.get(path) or {}
        fields: dict[str, Any] = {pk: path}
        for key in ("thumbnail_b64", "width", "height", "bytes", "taken_at"):
            if key in row:
                fields[key] = row[key]
        out.append(Hit(id=path, score=h.score, fields=fields))
    return SearchResponse(mode="dense", modality="image", hits=out)


def _image_search(req: SearchRequest) -> SearchResponse:
    """Encode the query via CLIP text, dense-search, join thumbnails."""
    plan = _plan()
    client = _client()
    embedding = plan["embedding"]
    pk = plan["schema"]["primary_key"]
    vector_field = plan["schema"]["vector_field"]
    model_id = str(embedding.get("model") or "ViT-B-32")
    device_hint = embedding.get("device_hint")

    try:
        qvec = embed_text_with_clip([req.query], model_id=model_id, device_hint=device_hint)[0]
    except LaunchpadError as e:
        raise HTTPException(status_code=400, detail=e.to_dict()) from e

    res = client.search(
        collection_name=plan["collection_name"],
        data=[qvec],
        anns_field=vector_field,
        limit=req.top_k,
        output_fields=[pk],
    )
    thumbs = _thumbnails()
    hits: list[Hit] = []
    for batch in res or []:
        for raw in batch:
            entity = getattr(raw, "entity", None)
            pk_val = (
                (entity.get(pk) if entity is not None else raw.get(pk))
                if hasattr(raw, "get") or entity is not None
                else None
            )
            score = float(getattr(raw, "score", 0.0) or 0.0)
            if not score and hasattr(raw, "get"):
                score = float(raw.get("distance", 0.0) or 0.0)
            path = str(pk_val) if pk_val else ""
            row = thumbs.get(path) or {}
            fields: dict[str, Any] = {pk: path}
            for key in ("thumbnail_b64", "width", "height", "bytes", "taken_at"):
                if key in row:
                    fields[key] = row[key]
            hits.append(Hit(id=path, score=score, fields=fields))
    return SearchResponse(mode="dense", modality="image", hits=hits)
