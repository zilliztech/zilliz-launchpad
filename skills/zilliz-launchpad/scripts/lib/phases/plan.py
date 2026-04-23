"""Phase 3: Plan — deterministic decision tree → plan.{json,md}."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..optional_deps import detect_device_hint
from ..profile import load_profile

DEFAULT_EMBEDDING = {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "dim": 1536,
    "modality": "text",
}
DEFAULT_IMAGE_EMBEDDING = {
    "provider": "clip-local",
    "model": "ViT-B-32",
    "dim": 512,
    "modality": "image",
}
VOYAGE_MULTIMODAL_EMBEDDING = {
    "provider": "voyage",
    "model": "voyage-multimodal-3",
    "dim": 1024,
    "modality": "image",
}
CLOUD_TARGETS = {"zilliz-serverless", "zilliz-dedicated", "zilliz-byoc"}
DEFAULT_BULK_IMPORT_THRESHOLD = 100_000
IMAGE_USE_CASES = {"image-search"}
IMAGE_DATA_SHAPE = "image_dir"


@dataclass
class IndexSpec:
    type: str
    metric: str = "COSINE"
    params: dict[str, Any] = field(default_factory=dict)
    quantization: str | None = None
    backend_compatibility: str = "Standalone-ok"  # or Cloud-only


@dataclass
class PlanSpec:
    collection_name: str
    target_uri: str
    schema: dict[str, Any]
    embedding: dict[str, Any]
    sparse_enabled: bool
    index: IndexSpec
    reranker: str | None
    chunking: dict[str, int]
    deployment_target: str
    bulk_import_threshold: int = DEFAULT_BULK_IMPORT_THRESHOLD
    cluster_id: str | None = None
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["index"] = asdict(self.index)
        return d


def _pick_index(dataset_size: int, deployment_target: str) -> IndexSpec:
    if dataset_size <= 100_000:
        return IndexSpec(
            type="HNSW",
            params={"M": 16, "efConstruction": 200},
            backend_compatibility="Standalone-ok",
        )
    if dataset_size <= 1_000_000:
        return IndexSpec(
            type="HNSW",
            params={"M": 24, "efConstruction": 256},
            backend_compatibility="Standalone-ok",
        )
    if deployment_target in CLOUD_TARGETS:
        return IndexSpec(
            type="DISKANN",
            params={"search_list_size": 100},
            backend_compatibility="Cloud-only",
        )
    # Large dataset, local — pick HNSW with a warning flag
    return IndexSpec(
        type="HNSW",
        params={"M": 32, "efConstruction": 256},
        backend_compatibility="Standalone-ok",
    )


def _pick_sparse(use_case: str, hybrid_preference: str) -> bool:
    if hybrid_preference == "dense":
        return False
    if hybrid_preference in ("sparse", "hybrid"):
        return True
    # auto
    return use_case in ("rag", "semantic-search")


def _pick_reranker(pref: str, use_case: str) -> str | None:
    if pref == "none":
        return None
    if pref == "cohere":
        return "cohere-rerank-3"
    if pref == "bge":
        return "bge-reranker-v2-m3"
    # auto: opt-in only for RAG with explicit quality focus → default off
    return None


def _target_uri(deployment_target: str) -> str:
    if deployment_target == "local-standalone":
        return "http://localhost:19530"
    return "https://<your-cluster>.api.gcp-us-west1.zillizcloud.com"


def _build_schema(
    collect: dict[str, Any], embedding: dict[str, Any], sparse: bool
) -> dict[str, Any]:
    pk = collect["suggested_primary_key"]
    text = collect["suggested_text_field"]
    skip = {pk, text} if text else {pk}
    extra = [f for f in collect["fields"] if f["name"] not in skip]
    return {
        "primary_key": pk,
        "text_field": text,
        "vector_field": "embedding",
        "dim": embedding["dim"],
        "sparse_field": "sparse" if sparse else None,
        "extra_fields": [
            {
                "name": f["name"],
                "type": f["type"],
                "max_length": 256 if f["type"] == "string" else None,
            }
            for f in extra
        ],
    }


def _pick_image_provider(configure: dict[str, Any]) -> dict[str, Any]:
    """Default to CLIP-local; honor a string or {provider} override."""
    pref = configure.get("embedding_preference")
    if pref is None or pref == "auto":
        embedding = dict(DEFAULT_IMAGE_EMBEDDING)
    elif isinstance(pref, str):
        if pref == "voyage-multimodal-3":
            embedding = dict(VOYAGE_MULTIMODAL_EMBEDDING)
        elif pref in ("clip-local", "ViT-B-32"):
            embedding = dict(DEFAULT_IMAGE_EMBEDDING)
        else:
            embedding = dict(DEFAULT_IMAGE_EMBEDDING)
            embedding["model"] = pref
    elif isinstance(pref, dict):
        provider = pref.get("provider", "clip-local")
        if provider == "voyage":
            embedding = dict(VOYAGE_MULTIMODAL_EMBEDDING)
        else:
            embedding = dict(DEFAULT_IMAGE_EMBEDDING)
        embedding.update({k: v for k, v in pref.items() if v is not None})
        embedding["modality"] = "image"
    else:
        embedding = dict(DEFAULT_IMAGE_EMBEDDING)

    embedding["device_hint"] = detect_device_hint()
    return embedding


def _pick_text_provider(configure: dict[str, Any]) -> dict[str, Any]:
    embedding = dict(DEFAULT_EMBEDDING)
    pref = configure.get("embedding_preference")
    if isinstance(pref, dict):
        embedding.update({k: v for k, v in pref.items() if v is not None})
    embedding.setdefault("modality", "text")
    return embedding


def _is_image_profile(collect: dict[str, Any], configure: dict[str, Any]) -> bool:
    return (
        collect.get("data_shape") == IMAGE_DATA_SHAPE
        or configure.get("use_case") in IMAGE_USE_CASES
    )


def plan_from_profile(profile: dict[str, Any]) -> PlanSpec:
    collect = profile["collect"]
    configure = profile["configure"]
    is_image = _is_image_profile(collect, configure)

    if is_image:
        embedding = _pick_image_provider(configure)
        sparse = False
        reranker = None
    else:
        embedding = _pick_text_provider(configure)
        sparse = _pick_sparse(configure["use_case"], configure.get("hybrid_preference", "auto"))
        reranker = _pick_reranker(
            configure.get("reranker_preference", "auto"), configure["use_case"]
        )

    index = _pick_index(configure["dataset_size"], configure["deployment_target"])
    schema = _build_schema(collect, embedding, sparse)
    target_uri = configure.get("target_uri") or _target_uri(configure["deployment_target"])

    rationale = [
        f"Dataset size {configure['dataset_size']} → index {index.type} {index.params}",
        f"Deployment '{configure['deployment_target']}' → URI {target_uri}",
        (
            f"Embedding provider '{embedding['provider']}' "
            f"model '{embedding['model']}' (dim {embedding['dim']})"
        ),
    ]
    if is_image:
        rationale.append(
            f"Image collection ({collect.get('record_count_estimate', '?')} images) "
            f"→ sparse field disabled, reranker none, device_hint={embedding['device_hint']}"
        )
    else:
        rationale.insert(
            1,
            f"Use case '{configure['use_case']}' + hybrid preference "
            f"'{configure.get('hybrid_preference', 'auto')}' → sparse={sparse}",
        )
    if reranker:
        rationale.append(f"Reranker '{reranker}' enabled per preference")

    return PlanSpec(
        collection_name="launchpad_collection",
        target_uri=target_uri,
        schema=schema,
        embedding=embedding,
        sparse_enabled=sparse,
        index=index,
        reranker=reranker,
        chunking={"size": 512, "overlap": 64},
        deployment_target=configure["deployment_target"],
        bulk_import_threshold=DEFAULT_BULK_IMPORT_THRESHOLD,
        cluster_id=configure.get("cluster_id"),
        rationale=rationale,
    )


def _plan_to_markdown(plan: PlanSpec) -> str:
    is_image = plan.embedding.get("modality") == "image"
    text_field_line = (
        f"- Text field: `{plan.schema['text_field']}`"
        if plan.schema["text_field"]
        else "- Text field: (none — image collection)"
    )
    sparse_line = (
        f"- Sparse field: `{plan.schema['sparse_field']}`"
        if plan.sparse_enabled
        else ("- Sparse field: disabled (image collection)" if is_image else "- Sparse: disabled")
    )
    embedding_lines = [
        "## Embedding",
        f"- Provider: `{plan.embedding['provider']}`",
        f"- Model: `{plan.embedding['model']}`",
        f"- Dim: {plan.embedding['dim']}",
        f"- Modality: `{plan.embedding.get('modality', 'text')}`",
    ]
    if is_image:
        embedding_lines.append(f"- Device hint: `{plan.embedding.get('device_hint', 'cpu')}`")

    lines = [
        "# Launchpad Plan",
        "",
        f"- **Collection**: `{plan.collection_name}`",
        f"- **Target URI**: `{plan.target_uri}`",
        f"- **Deployment**: `{plan.deployment_target}`",
        "",
        "## Schema",
        f"- Primary key: `{plan.schema['primary_key']}`",
        text_field_line,
        f"- Vector field: `{plan.schema['vector_field']}` (dim {plan.schema['dim']})",
        sparse_line,
        f"- Extra fields: {', '.join(f['name'] for f in plan.schema['extra_fields']) or '(none)'}",
        "",
        *embedding_lines,
        "",
        "## Index",
        f"- Type: `{plan.index.type}`",
        f"- Metric: `{plan.index.metric}`",
        f"- Params: `{json.dumps(plan.index.params)}`",
        f"- Backend compatibility: {plan.index.backend_compatibility}",
        "",
        "## Reranker",
        f"- {plan.reranker or 'disabled'}",
        "",
        "## Chunking",
        f"- Size: {plan.chunking['size']} tokens (approx)",
        f"- Overlap: {plan.chunking['overlap']} tokens (approx)",
        "",
        "## Rationale",
    ]
    lines += [f"- {r}" for r in plan.rationale]
    return "\n".join(lines) + "\n"


def run_plan(*, out_dir: Path) -> dict[str, Any]:
    profile = load_profile(out_dir)
    plan = plan_from_profile(profile)
    (out_dir / "plan.json").write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "plan.md").write_text(_plan_to_markdown(plan), encoding="utf-8")
    return plan.to_dict()
