"""Phase 3: Plan — deterministic decision tree → plan.{json,md}."""

from __future__ import annotations

import difflib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import typer

from ..cli import fail as _cli_fail
from ..errors import CliErrorEnvelope, InvalidProfileError, LaunchpadError
from ..optional_deps import detect_device_hint
from ..profile import load_profile
from ..run_dir import previous_run_dir, resolve_run_dir

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
VIDEO_USE_CASES = {"video-search"}
VIDEO_DATA_SHAPE = "video_dir"

VOYAGE_MULTIMODAL_PRICE_PER_IMAGE_USD = 0.00012  # indicative; see Voyage pricing


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
    chunking: dict[str, Any]
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
    is_video = collect.get("data_shape") == VIDEO_DATA_SHAPE
    extra_field_defs: list[dict[str, Any]] = []
    for f in extra:
        max_length: int | None
        if f["name"] == "video_path":
            max_length = 1024
        elif f["type"] == "string":
            max_length = 256
        else:
            max_length = None
        extra_field_defs.append({"name": f["name"], "type": f["type"], "max_length": max_length})
    return {
        "primary_key": pk,
        "text_field": text,
        "vector_field": "embedding",
        "dim": embedding["dim"],
        "sparse_field": "sparse" if sparse else None,
        "extra_fields": extra_field_defs,
        "is_video": is_video,
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


def _is_video_profile(collect: dict[str, Any], configure: dict[str, Any]) -> bool:
    return (
        collect.get("data_shape") == VIDEO_DATA_SHAPE
        or configure.get("use_case") in VIDEO_USE_CASES
    )


def _pick_video_provider(configure: dict[str, Any]) -> dict[str, Any]:
    """Same CLIP-or-Voyage selection as image, but label as video modality."""
    embedding = _pick_image_provider(configure)
    # Device hint already set by _pick_image_provider
    return embedding


def plan_from_profile(profile: dict[str, Any]) -> PlanSpec:
    collect = profile["collect"]
    configure = profile["configure"]
    is_video = _is_video_profile(collect, configure)
    is_image = not is_video and _is_image_profile(collect, configure)

    if is_video:
        embedding = _pick_video_provider(configure)
        sparse = False
        reranker = None
    elif is_image:
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

    chunking: dict[str, Any] = {"size": 512, "overlap": 64}
    if is_video:
        chunking["video"] = {
            "frame_interval_seconds": float(configure.get("frame_interval_seconds") or 2.0),
            "max_frames_per_video": int(configure.get("max_frames_per_video") or 600),
            "sampling_strategy": str(configure.get("sampling_strategy") or "every_n_seconds"),
            "scene_threshold": float(configure.get("scene_threshold") or 0.3),
        }

    rationale = [
        f"Dataset size {configure['dataset_size']} → index {index.type} {index.params}",
        f"Deployment '{configure['deployment_target']}' → URI {target_uri}",
        (
            f"Embedding provider '{embedding['provider']}' "
            f"model '{embedding['model']}' (dim {embedding['dim']})"
        ),
    ]
    if is_video:
        vrec = collect.get("record_count_estimate", "?")
        vcount = collect.get("video_count", "?")
        rationale.append(
            f"Video collection ({vcount} videos, {vrec} sampled frames) → "
            f"sparse field disabled, reranker none, device_hint={embedding['device_hint']}"
        )
        video_chunk = chunking["video"]
        rationale.append(
            f"Sampling strategy '{video_chunk['sampling_strategy']}' at "
            f"interval={video_chunk['frame_interval_seconds']}s, "
            f"cap={video_chunk['max_frames_per_video']} frames/video"
        )
        rationale.append(
            "Schema adds `video_path` (VARCHAR 1024) and `t_seconds` (FLOAT) "
            "scalars for deep-link playback"
        )
        if embedding["provider"] == "voyage" and isinstance(vrec, int):
            est_cost = vrec * VOYAGE_MULTIMODAL_PRICE_PER_IMAGE_USD
            rationale.append(
                f"Voyage multimodal ingest cost ≈ ${est_cost:.4f} "
                f"({vrec} frames × ${VOYAGE_MULTIMODAL_PRICE_PER_IMAGE_USD})"
            )
    elif is_image:
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
        chunking=chunking,
        deployment_target=configure["deployment_target"],
        bulk_import_threshold=DEFAULT_BULK_IMPORT_THRESHOLD,
        cluster_id=configure.get("cluster_id"),
        rationale=rationale,
    )


def _plan_to_markdown(plan: PlanSpec) -> str:
    is_image = plan.embedding.get("modality") == "image"
    is_video = bool(plan.schema.get("is_video"))
    if is_video:
        text_field_line = "- Text field: (none — video collection)"
        sparse_line = "- Sparse field: disabled (video collection)"
    elif plan.schema["text_field"]:
        text_field_line = f"- Text field: `{plan.schema['text_field']}`"
        sparse_line = (
            f"- Sparse field: `{plan.schema['sparse_field']}`"
            if plan.sparse_enabled
            else "- Sparse: disabled"
        )
    else:
        text_field_line = "- Text field: (none — image collection)"
        sparse_line = "- Sparse field: disabled (image collection)"

    embedding_lines = [
        "## Embedding",
        f"- Provider: `{plan.embedding['provider']}`",
        f"- Model: `{plan.embedding['model']}`",
        f"- Dim: {plan.embedding['dim']}",
        f"- Modality: `{plan.embedding.get('modality', 'text')}`",
    ]
    if is_image or is_video:
        embedding_lines.append(f"- Device hint: `{plan.embedding.get('device_hint', 'cpu')}`")

    chunking_lines = [
        "## Chunking",
        f"- Size: {plan.chunking.get('size', 512)} tokens (approx)",
        f"- Overlap: {plan.chunking.get('overlap', 64)} tokens (approx)",
    ]
    if is_video:
        vchunk = plan.chunking.get("video") or {}
        chunking_lines += [
            "",
            "## Video",
            f"- Sampling strategy: `{vchunk.get('sampling_strategy', 'every_n_seconds')}`",
            f"- Frame interval: {vchunk.get('frame_interval_seconds', 2.0)} s",
            f"- Max frames per video: {vchunk.get('max_frames_per_video', 600)}",
            (
                f"- Scene threshold: {vchunk.get('scene_threshold', 0.3)}"
                if vchunk.get("sampling_strategy") == "scene_change"
                else "- Scene threshold: n/a (every_n_seconds)"
            ),
            "- Scalar fields: `video_path` (deep-link), `t_seconds` (timestamp)",
        ]

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
        *chunking_lines,
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


def _diff_fail(err: LaunchpadError) -> None:
    """Emit the standard error envelope but exit with code 2.

    `plan diff` reserves exit code 1 for the clean "the two plans differ"
    signal (git-diff style), so failures must use a distinct non-1 code
    while still emitting the same JSON envelope the skill's remediation
    map consumes.
    """
    print(CliErrorEnvelope.from_error(err).to_json(), file=sys.stderr)
    raise typer.Exit(code=2)


def _resolve_for_diff(arg: str | None) -> Path:
    """Resolve a `plan diff` run-dir argument as a structured error path."""
    try:
        return resolve_run_dir(arg)
    except FileNotFoundError as exc:
        raise InvalidProfileError(
            pointer=str(arg) if arg is not None else "<latest>",
            reason=str(exc),
        ) from exc


def _plan_md_lines(run: Path) -> list[str]:
    plan_md = run / "plan.md"
    if not plan_md.exists():
        raise InvalidProfileError(
            pointer=str(plan_md),
            reason="missing plan.md — run `plan` in this run dir first",
        )
    return plan_md.read_text(encoding="utf-8").splitlines(keepends=True)


def register(app: typer.Typer) -> None:
    """Attach the Phase 3 ``plan`` command group to the shared app."""

    plan_app = typer.Typer(help="Phase 3 — produce plan.{json,md}.")

    @plan_app.callback(invoke_without_command=True)
    def plan(
        ctx: typer.Context,
        run_dir: str | None = typer.Option(None, "--run-dir"),
    ) -> None:
        """Phase 3 — produce plan.{json,md}."""
        if ctx.invoked_subcommand is not None:
            return
        out = resolve_run_dir(run_dir)
        try:
            result = run_plan(out_dir=out)
        except LaunchpadError as e:
            _cli_fail(e)
        typer.echo(f"run-dir: {out}")
        typer.echo(f"index: {result['index']['type']} {result['index']['params']}")
        typer.echo(f"sparse: {result['sparse_enabled']}")

    @plan_app.command("diff")
    def plan_diff(
        run_a: str | None = typer.Argument(None, help="New side (default: latest run)."),
        run_b: str | None = typer.Argument(None, help="Old side (default: the run before run-a)."),
    ) -> None:
        """Unified diff of two run dirs' plan.md (run-b → run-a).

        Exit 0 = identical, 1 = differ, 2 = error.
        """
        try:
            new_run = _resolve_for_diff(run_a)
            if run_b is not None:
                old_run = _resolve_for_diff(run_b)
            else:
                prev = previous_run_dir(new_run)
                if prev is None:
                    raise InvalidProfileError(
                        pointer=str(new_run),
                        reason="no previous run to compare against — "
                        "run `plan` again after editing configure.json",
                    )
                old_run = prev
            old_lines = _plan_md_lines(old_run)
            new_lines = _plan_md_lines(new_run)
        except LaunchpadError as e:
            _diff_fail(e)

        if old_lines == new_lines:
            raise typer.Exit(code=0)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"{old_run.name}/plan.md",
            tofile=f"{new_run.name}/plan.md",
        )
        sys.stdout.writelines(diff)
        raise typer.Exit(code=1)

    app.add_typer(plan_app, name="plan")
