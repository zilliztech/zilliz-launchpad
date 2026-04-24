"""Dense / sparse / hybrid search with optional reranker."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pymilvus import MilvusClient

from .credentials import resolve_required
from .embeddings import EmbeddingProvider, embed_image_batch, embed_image_batch_voyage
from .errors import ImageDecodeError, SparseUnavailable, UnsupportedImageProviderError

FusionMode = Literal["rrf", "weighted"]
Mode = Literal["dense", "sparse", "hybrid"]


@dataclass
class Hit:
    id: str
    score: float
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "score": self.score, "fields": self.fields}


def _hits_from_milvus(raw: Sequence[Any]) -> list[Hit]:
    out: list[Hit] = []
    for h in raw:
        # pymilvus 2.4+ hit objects expose `id`, `distance`/`score`, and `entity`
        hid = str(getattr(h, "id", None) or h.get("id") if isinstance(h, dict) else h.id)
        raw_score = getattr(h, "score", None) or getattr(h, "distance", 0.0) or 0.0
        score = float(raw_score)
        entity = getattr(h, "entity", None) or (h.get("entity") if isinstance(h, dict) else {})
        fields = dict(entity) if entity else {}
        out.append(Hit(id=hid, score=score, fields=fields))
    return out


def search_dense(
    client: MilvusClient,
    collection: str,
    query_text: str,
    embedder: EmbeddingProvider,
    *,
    top_k: int = 10,
    vector_field: str = "embedding",
    filter: str | None = None,
    output_fields: Sequence[str] | None = None,
    rerank: str | None = None,
) -> list[Hit]:
    vec = embedder.embed([query_text])[0]
    k = top_k * 3 if rerank else top_k
    raw = client.search(
        collection_name=collection,
        data=[vec],
        anns_field=vector_field,
        limit=k,
        filter=filter or "",
        output_fields=list(output_fields) if output_fields else None,
    )
    hits = _hits_from_milvus(raw[0]) if raw else []
    if rerank:
        hits = apply_reranker(rerank, query_text, hits)[:top_k]
    return hits[:top_k]


_IMAGE_PROVIDERS = frozenset({"clip-local", "voyage"})


def _encode_query_image(image_bytes: bytes, plan: dict[str, Any]) -> list[float]:
    """Encode raw image bytes into a query vector matching the plan's provider.

    Writes bytes to a NamedTemporaryFile so the existing path-based encoders
    in `embeddings.py` can be reused without change. The file is unlinked on
    exit of the `with` block regardless of exception paths.
    """
    import tempfile

    embedding = plan.get("embedding") or {}
    provider = str(embedding.get("provider") or "").lower()
    if provider not in _IMAGE_PROVIDERS:
        raise UnsupportedImageProviderError(provider or "(none)")
    if provider == "voyage" and str(embedding.get("modality") or "") != "image":
        raise UnsupportedImageProviderError(str(embedding.get("provider")))

    with tempfile.NamedTemporaryFile(suffix=".img", delete=True) as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        try:
            if provider == "clip-local":
                model_id = str(embedding.get("model") or "ViT-B-32")
                device_hint = embedding.get("device_hint")
                vecs = embed_image_batch([tmp.name], model_id=model_id, device_hint=device_hint)
            else:
                model_id = str(embedding.get("model") or "voyage-multimodal-3")
                vecs = embed_image_batch_voyage([tmp.name], model_id=model_id)
        except UnsupportedImageProviderError:
            raise
        except Exception as exc:
            # Pillow's UnidentifiedImageError, Voyage API errors, torch failures —
            # they all mean we could not turn these bytes into a usable query vector.
            raise ImageDecodeError(str(exc)) from exc

    if not vecs:
        raise ImageDecodeError("encoder returned no vector")
    return list(vecs[0])


def search_image_to_image(
    client: MilvusClient,
    collection: str,
    image_bytes: bytes,
    plan: dict[str, Any],
    *,
    top_k: int = 10,
    vector_field: str = "embedding",
    filter: str | None = None,
    output_fields: Sequence[str] | None = None,
) -> list[Hit]:
    """Dense-search the image collection using another image as the query.

    The query image is encoded with whatever provider the plan recorded at
    ingest time — query vectors must live in the same embedding space as the
    stored vectors, so we read the provider/model/device_hint from `plan`
    rather than the environment.
    """
    vec = _encode_query_image(image_bytes, plan)
    raw = client.search(
        collection_name=collection,
        data=[vec],
        anns_field=vector_field,
        limit=top_k,
        filter=filter or "",
        output_fields=list(output_fields) if output_fields else None,
    )
    hits = _hits_from_milvus(raw[0]) if raw else []
    return hits[:top_k]


def _has_sparse_field(client: MilvusClient, collection: str, sparse_field: str) -> bool:
    info = client.describe_collection(collection)
    return any(f.get("name") == sparse_field for f in info.get("fields", []))


def search_sparse(
    client: MilvusClient,
    collection: str,
    query_text: str,
    *,
    top_k: int = 10,
    sparse_field: str = "sparse",
    filter: str | None = None,
    output_fields: Sequence[str] | None = None,
    rerank: str | None = None,
) -> list[Hit]:
    if not _has_sparse_field(client, collection, sparse_field):
        raise SparseUnavailable(collection=collection)

    # Milvus 2.4+ supports BM25-style sparse lookup with text input
    k = top_k * 3 if rerank else top_k
    raw = client.search(
        collection_name=collection,
        data=[query_text],
        anns_field=sparse_field,
        limit=k,
        filter=filter or "",
        output_fields=list(output_fields) if output_fields else None,
    )
    hits = _hits_from_milvus(raw[0]) if raw else []
    if rerank:
        hits = apply_reranker(rerank, query_text, hits)[:top_k]
    return hits[:top_k]


def _rrf(ranked_lists: list[list[Hit]], k: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    meta: dict[str, Hit] = {}
    for hits in ranked_lists:
        for rank, h in enumerate(hits, start=1):
            scores[h.id] = scores.get(h.id, 0.0) + 1.0 / (k + rank)
            meta.setdefault(h.id, h)
    ordered = sorted(meta.values(), key=lambda h: scores[h.id], reverse=True)
    for h in ordered:
        h.score = scores[h.id]
    return ordered


def _weighted(ranked_lists: list[list[Hit]], weights: Sequence[float]) -> list[Hit]:
    assert len(ranked_lists) == len(weights), "weights must match ranked_lists length"
    scores: dict[str, float] = {}
    meta: dict[str, Hit] = {}
    for w, hits in zip(weights, ranked_lists, strict=True):
        if not hits:
            continue
        max_score = max(h.score for h in hits) or 1.0
        for h in hits:
            scores[h.id] = scores.get(h.id, 0.0) + w * (h.score / max_score)
            meta.setdefault(h.id, h)
    ordered = sorted(meta.values(), key=lambda h: scores[h.id], reverse=True)
    for h in ordered:
        h.score = scores[h.id]
    return ordered


def search_hybrid(
    client: MilvusClient,
    collection: str,
    query_text: str,
    embedder: EmbeddingProvider,
    *,
    top_k: int = 10,
    fusion: FusionMode = "rrf",
    weights: tuple[float, float] = (0.5, 0.5),
    vector_field: str = "embedding",
    sparse_field: str = "sparse",
    filter: str | None = None,
    output_fields: Sequence[str] | None = None,
    rerank: str | None = None,
) -> list[Hit]:
    candidates = top_k * 3 if rerank else top_k * 2
    dense = search_dense(
        client,
        collection,
        query_text,
        embedder,
        top_k=candidates,
        vector_field=vector_field,
        filter=filter,
        output_fields=output_fields,
    )
    sparse = search_sparse(
        client,
        collection,
        query_text,
        top_k=candidates,
        sparse_field=sparse_field,
        filter=filter,
        output_fields=output_fields,
    )
    fused = _rrf([dense, sparse]) if fusion == "rrf" else _weighted([dense, sparse], weights)
    if rerank:
        fused = apply_reranker(rerank, query_text, fused)
    return fused[:top_k]


# --- Rerankers -------------------------------------------------------------


def apply_reranker(name: str, query: str, hits: list[Hit]) -> list[Hit]:
    if name.startswith("cohere-"):
        return _cohere_rerank(name, query, hits)
    if name.startswith("bge-"):
        return _bge_rerank(name, query, hits)
    raise ValueError(f"Unknown reranker: {name}")


def _cohere_rerank(model: str, query: str, hits: list[Hit]) -> list[Hit]:
    import cohere

    if not hits:
        return hits
    client = cohere.Client(api_key=resolve_required("COHERE_API_KEY"))
    texts = [str(h.fields.get("text", "")) for h in hits]
    model_id = model.replace("cohere-", "")
    resp = client.rerank(model=model_id, query=query, documents=texts)
    out: list[Hit] = []
    for r in resp.results:
        base = hits[r.index]
        base.score = float(r.relevance_score)
        out.append(base)
    return out


def _bge_rerank(model: str, query: str, hits: list[Hit]) -> list[Hit]:
    """BGE reranker via Zilliz BYOM endpoint."""
    import httpx

    if not hits:
        return hits
    url = resolve_required("ZILLIZ_BYOM_URL")
    key = resolve_required("ZILLIZ_BYOM_KEY")
    texts = [str(h.fields.get("text", "")) for h in hits]
    resp = httpx.post(
        url.rstrip("/") + "/rerank",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "query": query, "documents": texts},
        timeout=60.0,
    )
    resp.raise_for_status()
    scores = resp.json()["scores"]
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    out: list[Hit] = []
    for i in order:
        base = hits[i]
        base.score = float(scores[i])
        out.append(base)
    return out
