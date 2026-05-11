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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pymilvus.exceptions import MilvusException

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
_VIDEO_STATIC_PREFIX = "/videos"

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
    """Map PK (image_path or frame_path) → row dict for visual runs.

    Returns an empty dict for text runs or when collect.json is missing.
    """
    collect_path = _run_dir() / "collect.json"
    if not collect_path.exists():
        return {}
    try:
        collect = json.loads(collect_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    data_shape = collect.get("data_shape")
    out: dict[str, dict[str, Any]] = {}
    if data_shape == "image_dir":
        for row in collect.get("rows") or []:
            path = row.get("image_path")
            if isinstance(path, str):
                out[path] = row
    elif data_shape == "video_dir":
        for row in collect.get("rows") or []:
            path = row.get("frame_path")
            if isinstance(path, str):
                out[path] = row
    return out


def _video_static_root() -> Path | None:
    """Directory to serve video files from. None if disabled/unresolvable."""
    override = os.environ.get("LAUNCHPAD_VIDEO_STATIC_ROOT")
    if override:
        p = Path(override).resolve()
        if p.exists() and p.is_dir():
            return p
    # Default: parent of the run directory (where user-supplied video dirs
    # typically live one level out).
    try:
        run = _run_dir().resolve()
    except RuntimeError:
        return None
    return run.parent if run.parent.exists() else None


def _video_url_for(video_path: str) -> tuple[str | None, str | None]:
    """Map a video path to a sidecar-served URL or return (None, warning)."""
    root = _video_static_root()
    if root is None:
        return None, "video static root not configured"
    try:
        resolved = Path(video_path).resolve()
        rel = resolved.relative_to(root)
    except (ValueError, OSError):
        return None, f"video path outside static mount root ({root})"
    return f"{_VIDEO_STATIC_PREFIX}/{rel.as_posix()}", None


_plan_cache: dict[str, Any] | None = None
_client_cache: Any = None
_embedder_cache: Any = None
_thumb_cache: dict[str, dict[str, Any]] | None = None
_last_query_vec: list[float] | None = None


def _stash_query_vector(vec: list[float]) -> None:
    global _last_query_vec
    _last_query_vec = vec


def _get_last_query_vector() -> list[float] | None:
    return _last_query_vec


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


def _is_video() -> bool:
    return bool(_plan().get("schema", {}).get("is_video"))


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
    filter: str | None = None
    rerank: str | None = None


def _scalar_output_fields(plan: dict[str, Any]) -> list[str]:
    """Schema scalar fields (pk + text + extra) excluding vector / sparse fields."""
    schema = plan.get("schema") or {}
    vector_field = schema.get("vector_field")
    sparse_field = schema.get("sparse_field")
    names: list[str] = []
    seen: set[str] = set()
    for candidate in (schema.get("primary_key"), schema.get("text_field")):
        if isinstance(candidate, str) and candidate and candidate not in seen:
            names.append(candidate)
            seen.add(candidate)
    for extra in schema.get("extra_fields") or []:
        name = extra.get("name") if isinstance(extra, dict) else None
        if not isinstance(name, str) or not name or name in seen:
            continue
        if name == vector_field or name == sparse_field:
            continue
        names.append(name)
        seen.add(name)
    return names


def _resolve_rerank(plan: dict[str, Any], requested: str | None) -> str | None:
    """Map the request's rerank value to a concrete reranker model name (or None)."""
    if requested is None:
        return None
    value = requested.strip().lower()
    if value in ("", "off"):
        return None
    if value == "default":
        default = plan.get("reranker")
        if not isinstance(default, str) or not default:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "no_default_reranker",
                    "message": "Active plan has no default reranker configured.",
                },
            )
        return default
    return requested


class Hit(BaseModel):
    id: str
    score: float
    fields: dict[str, Any]


class SearchResponse(BaseModel):
    mode: str
    modality: Literal["text", "image", "video"]
    hits: list[Hit]


class InfoResponse(BaseModel):
    collection_name: str
    modality: Literal["text", "image", "video"]
    primary_key: str
    vector_field: str
    sparse_enabled: bool
    embedding: dict[str, Any]
    has_thumbnails: bool
    video_static_prefix: str | None = None
    data_shape: str | None = None
    default_reranker: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "collection": _plan()["collection_name"]}


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    plan = _plan()
    if _is_video():
        modality: Literal["text", "image", "video"] = "video"
    elif _is_image():
        modality = "image"
    else:
        modality = "text"
    data_shape = None
    try:
        collect = json.loads((_run_dir() / "collect.json").read_text(encoding="utf-8"))
        data_shape = collect.get("data_shape")
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return InfoResponse(
        collection_name=plan["collection_name"],
        modality=modality,
        primary_key=plan["schema"]["primary_key"],
        vector_field=plan["schema"]["vector_field"],
        sparse_enabled=bool(plan.get("sparse_enabled")),
        embedding={
            "provider": plan["embedding"]["provider"],
            "model": plan["embedding"]["model"],
            "dim": plan["embedding"]["dim"],
        },
        has_thumbnails=bool(_thumbnails()),
        video_static_prefix=_VIDEO_STATIC_PREFIX if modality == "video" else None,
        data_shape=data_shape,
        default_reranker=(plan.get("reranker") if isinstance(plan.get("reranker"), str) else None),
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    if _is_video():
        return _video_search(req)
    if _is_image():
        return _image_search(req)

    plan = _plan()
    client = _client()
    embedder = _embedder()
    output_fields = _scalar_output_fields(plan)
    filter_expr = req.filter or None
    rerank = _resolve_rerank(plan, req.rerank)
    try:
        if req.mode == "dense":
            hits = search_dense(
                client,
                plan["collection_name"],
                req.query,
                embedder,
                top_k=req.top_k,
                vector_field=plan["schema"]["vector_field"],
                filter=filter_expr,
                output_fields=output_fields,
                rerank=rerank,
            )
        elif req.mode == "sparse":
            hits = search_sparse(
                client,
                plan["collection_name"],
                req.query,
                top_k=req.top_k,
                sparse_field=plan["schema"].get("sparse_field") or "sparse",
                filter=filter_expr,
                output_fields=output_fields,
                rerank=rerank,
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
                filter=filter_expr,
                output_fields=output_fields,
                rerank=rerank,
            )
    except SparseUnavailable as e:
        raise HTTPException(status_code=400, detail=e.to_dict()) from e
    except LaunchpadError as e:
        raise HTTPException(status_code=400, detail=e.to_dict()) from e
    except MilvusException as e:
        raise HTTPException(
            status_code=400,
            detail={"code": "bad_filter", "message": str(e)},
        ) from e

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
        if _is_video():
            _enrich_with_video_fields(fields, row)
        out.append(Hit(id=path, score=h.score, fields=fields))
    modality: Literal["text", "image", "video"] = "video" if _is_video() else "image"
    return SearchResponse(mode="dense", modality=modality, hits=out)


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


# --- Video branch ---------------------------------------------------------


def _hit_field(raw: Any, entity: Any, key: str) -> Any:
    if entity is not None:
        return entity.get(key)
    if hasattr(raw, "get"):
        return raw.get(key)
    return None


def _enrich_with_video_fields(fields: dict[str, Any], row: dict[str, Any]) -> None:
    """Add video_path, t_seconds, video_url to a hit's fields dict."""
    if "video_path" in row:
        fields["video_path"] = row["video_path"]
    if "t_seconds" in row:
        fields["t_seconds"] = row["t_seconds"]
    if "source_index" in row:
        fields["source_index"] = row["source_index"]
    video_path = row.get("video_path")
    if isinstance(video_path, str) and video_path:
        url, warning = _video_url_for(video_path)
        fields["video_url"] = url
        if warning:
            fields["video_url_warning"] = warning


def _video_search(req: SearchRequest) -> SearchResponse:
    """Text → CLIP-text → Milvus dense search, enrich each hit with video fields."""
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
    _stash_query_vector(qvec)

    res = client.search(
        collection_name=plan["collection_name"],
        data=[qvec],
        anns_field=vector_field,
        limit=req.top_k,
        output_fields=[pk, "video_path", "t_seconds"],
    )
    return SearchResponse(mode="dense", modality="video", hits=_extract_video_hits(res, pk))


def _extract_video_hits(res: Any, pk: str) -> list[Hit]:
    thumbs = _thumbnails()
    hits: list[Hit] = []
    for batch in res or []:
        for raw in batch:
            entity = getattr(raw, "entity", None)
            pk_val = _hit_field(raw, entity, pk)
            score = float(getattr(raw, "score", 0.0) or 0.0)
            if not score and hasattr(raw, "get"):
                score = float(raw.get("distance", 0.0) or 0.0)
            path = str(pk_val) if pk_val else ""
            thumb_row = thumbs.get(path) or {}
            fields: dict[str, Any] = {pk: path}
            if "thumbnail_b64" in thumb_row:
                fields["thumbnail_b64"] = thumb_row["thumbnail_b64"]
            # Prefer the live scalar fields the search returned (video_path, t_seconds);
            # fall back to collect.json row data.
            merged = dict(thumb_row)
            for key in ("video_path", "t_seconds"):
                live = _hit_field(raw, entity, key)
                if live is not None:
                    merged[key] = live
            _enrich_with_video_fields(fields, merged)
            hits.append(Hit(id=path, score=score, fields=fields))
    return hits


class VideoFramesRequest(BaseModel):
    video_path: str
    top_k: int = Field(default=4, ge=1, le=50)


@app.post("/video_frames", response_model=SearchResponse)
def video_frames(req: VideoFramesRequest) -> SearchResponse:
    if not _is_video():
        raise HTTPException(
            status_code=400,
            detail={
                "code": "not_video_collection",
                "message": "/video_frames requires a video collection",
            },
        )
    qvec = _get_last_query_vector()
    if qvec is None:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "no_prior_query",
                "message": (
                    "/video_frames reuses the most recent query vector; "
                    "call /search or /search_image first."
                ),
            },
        )
    plan = _plan()
    client = _client()
    pk = plan["schema"]["primary_key"]
    vector_field = plan["schema"]["vector_field"]
    filter_expr = f'video_path == "{req.video_path}"'
    res = client.search(
        collection_name=plan["collection_name"],
        data=[qvec],
        anns_field=vector_field,
        limit=req.top_k,
        output_fields=[pk, "video_path", "t_seconds"],
        filter=filter_expr,
    )
    return SearchResponse(mode="dense", modality="video", hits=_extract_video_hits(res, pk))


# --- Static video mount (mounted last so the other routes win path lookup) ---


_static_mounted = False


def _mount_video_static() -> None:
    """Mount the configured video static root once, lazily."""
    global _static_mounted
    if _static_mounted:
        return
    root = _video_static_root()
    if root is None:
        return
    app.mount(
        _VIDEO_STATIC_PREFIX,
        StaticFiles(directory=str(root), check_dir=False),
        name="videos",
    )
    _static_mounted = True


@app.on_event("startup")
def _on_startup() -> None:
    _mount_video_static()
