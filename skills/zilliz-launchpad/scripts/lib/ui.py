"""FastAPI sidecar for the demo UI.

Environment:
  LAUNCHPAD_RUN_DIR: path to the active run directory (contains plan.json)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .client import MilvusClient
from .embeddings import make_embedder
from .errors import LaunchpadError, SparseUnavailable
from .search import search_dense, search_hybrid, search_sparse

app = FastAPI(title="zilliz-launchpad sidecar")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _load_plan() -> dict[str, Any]:
    run_dir = os.environ.get("LAUNCHPAD_RUN_DIR")
    if not run_dir:
        raise RuntimeError("LAUNCHPAD_RUN_DIR is not set")
    plan_path = Path(run_dir) / "plan.json"
    data: dict[str, Any] = json.loads(plan_path.read_text(encoding="utf-8"))
    return data


_plan_cache: dict[str, Any] | None = None
_client_cache: Any = None
_embedder_cache: Any = None


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


def _embedder() -> Any:
    global _embedder_cache
    if _embedder_cache is None:
        p = _plan()["embedding"]
        _embedder_cache = make_embedder(p["provider"], p["model"], p["dim"])
    return _embedder_cache


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
    hits: list[Hit]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "collection": _plan()["collection_name"]}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
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
        hits=[Hit(id=h.id, score=h.score, fields=h.fields) for h in hits],
    )
