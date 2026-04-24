"""Phase 5: Evaluate — run query set against live collection, score variants.

Consumes a completed Execute run (`plan.json` + `execute.json`) and writes
`eval_report.{json,md}` in the same run directory. Supports three query-set
tiers (qrels → queries → derived-from-corpus) and a comparison mode that
re-runs the same query set against alternative plan variants.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..client import Backend, MilvusClient, detect_target
from ..embeddings import embed_text_with_clip, make_embedder
from ..errors import (
    BackendUnsupportedError,
    InvalidProfileError,
    JudgeUnavailableError,
    QrelsMissingError,
)
from ..evaluator import (
    JudgeConfig,
    QueryWithExpectedIds,
    compute_latency,
    compute_rag_quality,
    compute_retrieval_metrics,
    derive_queries_from_corpus,
)
from ..run_dir import load_plan, preflight_execute_artifact
from ..search import search_dense, search_hybrid, search_image_to_image
from ..vision_judge import caption_images, is_vision_capable
from .execute import _iter_documents

DERIVED_IMAGE_QUERIES_FILE = "derived_image_queries.jsonl"
_DERIVED_IMAGE_SAMPLE_SIZE = 12  # vision API calls are expensive, keep modest

logger = logging.getLogger(__name__)

_TOP_K = 10
_DERIVED_SAMPLE_SIZE = 25
_VARIANT_CAP = 6
_ALLOWED_OVERRIDE_AXES = frozenset({"embedding", "index", "hybrid", "reranker"})
# Nested leaf axes that are allowed under their parent:
_ALLOWED_NESTED = {
    "embedding": frozenset({"model", "provider", "dim"}),
    "index": frozenset({"params", "type", "metric"}),
}

# --- Public driver ---------------------------------------------------------


def run_evaluate(
    *,
    out_dir: Path,
    qrels_path: str | None,
    queries_path: str | None,
    concurrency: int,
    judge_llm: str | None,
    compare_path: str | None,
    allow_large: bool,
) -> dict[str, Any]:
    """Run Phase 5 against `out_dir` and write eval_report.{json,md}.

    Returns the parsed report dict. All failures are raised as
    `LaunchpadError` subclasses so the CLI can route them through
    the standard envelope.
    """
    preflight_execute_artifact(out_dir)
    plan = load_plan(out_dir)
    target = detect_target(plan["target_uri"])
    _refuse_milvus_lite(target.uri, target.backend)

    judge = JudgeConfig.parse(judge_llm) if judge_llm else None
    queries, derived = _resolve_query_set(
        out_dir=out_dir,
        plan=plan,
        qrels_path=qrels_path,
        queries_path=queries_path,
        judge=judge,
    )

    variants = _resolve_variants(compare_path=compare_path, allow_large=allow_large)
    if variants and (derived or all(not q.relevant_ids for q in queries)):
        # Comparison mode ranks variants by retrieval quality — that requires
        # labelled qrels. Derived queries only carry self-ids, which would
        # produce a ranking of variants by how well each re-finds the same
        # docs the query was sampled from. Not meaningful.
        raise QrelsMissingError()

    base_row = _evaluate_single(
        label="base",
        plan=plan,
        queries=queries,
        concurrency=concurrency,
        judge=judge,
    )
    variant_rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_plan = _apply_variant(plan, variant)
        # We evaluate the variant against the *same* collection for now —
        # sub-plan Execute is the Phase-6 responsibility per the design doc's
        # "Decision 4". Record which axes were overridden so the user can see
        # the hypothesis even when execution isn't wired.
        row = _evaluate_single(
            label=str(variant["name"]),
            plan=variant_plan,
            queries=queries,
            concurrency=concurrency,
            judge=judge,
        )
        row["overrides"] = dict(variant.get("overrides") or {})
        variant_rows.append(row)

    report = _build_report(
        out_dir=out_dir,
        queries=queries,
        base_row=base_row,
        variant_rows=variant_rows,
        derived=derived,
    )

    _write_report(out_dir, report)
    _append_observability_sample(out_dir, report)
    return report


# --- Query-image smoke -----------------------------------------------------


def run_query_image_smoke(
    *,
    out_dir: Path,
    query_image_path: str,
    top_k: int = _TOP_K,
) -> list[dict[str, Any]]:
    """Smoke path: encode a query image, return top-k hits, skip metrics.

    Writes nothing to disk. The CLI formats the returned rows for stdout.
    """
    preflight_execute_artifact(out_dir)
    plan = load_plan(out_dir)
    if not _is_image_plan(plan):
        raise InvalidProfileError(
            pointer="cli",
            reason="--query-image requires an image collection; this run is text-only",
        )
    path = Path(query_image_path)
    if not path.exists():
        raise InvalidProfileError(pointer=str(path), reason="query image file not found")
    image_bytes = path.read_bytes()

    client = MilvusClient(uri=plan["target_uri"])
    schema = plan["schema"]
    hits = search_image_to_image(
        client,
        plan["collection_name"],
        image_bytes,
        plan,
        top_k=top_k,
        vector_field=schema["vector_field"],
        output_fields=[schema["primary_key"]],
    )
    return [{"id": h.id, "score": h.score} for h in hits]


# --- Query-set resolution --------------------------------------------------


def _is_image_plan(plan: dict[str, Any]) -> bool:
    return bool(plan.get("embedding", {}).get("modality") == "image")


def _resolve_query_set(
    *,
    out_dir: Path,
    plan: dict[str, Any],
    qrels_path: str | None,
    queries_path: str | None,
    judge: JudgeConfig | None = None,
) -> tuple[list[QueryWithExpectedIds], bool]:
    """Returns (queries, derived_flag)."""
    if qrels_path and queries_path:
        raise InvalidProfileError(
            pointer="cli",
            reason="pass exactly one of --qrels or --queries (or neither for derived mode)",
        )
    image = _is_image_plan(plan)
    if qrels_path:
        if image:
            return _load_image_qrels(Path(qrels_path)), False
        return _load_qrels(Path(qrels_path)), False
    if queries_path:
        return _load_queries(Path(queries_path)), False
    if image:
        return _derive_image_queries(out_dir=out_dir, plan=plan, judge=judge), True
    return _derive_queries(out_dir=out_dir, plan=plan), True


def _load_image_qrels(path: Path) -> list[QueryWithExpectedIds]:
    """Image qrels loader. Accepts two row shapes, possibly mixed in one file:

    - Text→image: `{query_text, image_paths}` (existing)
    - Image→image: `{query_image_path, expected_image_ids}` (new in #15)

    `image_paths` / `expected_image_ids` become the collection's primary keys,
    so both rows land in `relevant_ids` and share the downstream metric path.
    """
    if not path.exists():
        raise InvalidProfileError(pointer=str(path), reason="qrels file not found")
    out: list[QueryWithExpectedIds] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise InvalidProfileError(
                    pointer=f"{path}:{lineno}", reason=f"invalid JSON: {exc}"
                ) from exc
            grade = int(record.get("grade") or 1)
            query_image_path = record.get("query_image_path")
            query_text = record.get("query_text") or record.get("query")

            if query_image_path:
                relevant = (
                    record.get("expected_image_ids")
                    or record.get("expected_image_paths")
                    or record.get("image_paths")
                )
                if isinstance(relevant, str):
                    relevant = [relevant]
                if not relevant or not isinstance(relevant, list):
                    raise InvalidProfileError(
                        pointer=f"{path}:{lineno}",
                        reason=("image→image row missing 'expected_image_ids' list"),
                    )
                out.append(
                    QueryWithExpectedIds(
                        query=str(query_image_path),
                        relevant_ids=tuple(str(p) for p in relevant),
                        grade=grade,
                        query_image_path=str(query_image_path),
                    )
                )
                continue

            if query_text:
                paths = record.get("image_paths") or record.get("image_path")
                if isinstance(paths, str):
                    paths = [paths]
                if not paths or not isinstance(paths, list):
                    raise InvalidProfileError(
                        pointer=f"{path}:{lineno}",
                        reason="missing 'image_path' / 'image_paths' value",
                    )
                out.append(
                    QueryWithExpectedIds(
                        query=str(query_text).strip(),
                        relevant_ids=tuple(str(p) for p in paths),
                        grade=grade,
                    )
                )
                continue

            raise InvalidProfileError(
                pointer=f"{path}:{lineno}",
                reason=(
                    "qrels row must carry 'query_text' (text→image) or "
                    "'query_image_path' (image→image)"
                ),
            )
    return out


def _derive_image_queries(
    *, out_dir: Path, plan: dict[str, Any], judge: JudgeConfig | None
) -> list[QueryWithExpectedIds]:
    """Caption sampled images via a vision-capable judge.

    Caches the captions to `runs/<id>/derived_image_queries.jsonl` so reruns
    skip the LLM entirely. Each cache row carries `(image_path, caption)`.
    Re-uses cached rows whose paths still exist; new paths are appended.
    """
    if judge is None or not is_vision_capable(judge.provider, judge.model):
        env_var = (
            judge.env_var
            if judge is not None
            else "OPENAI_API_KEY"  # default suggestion when no --judge-llm at all
        )
        provider = f"{judge.provider}:{judge.model}" if judge else "(none)"
        raise JudgeUnavailableError(provider=provider, env_var=env_var)

    collect_path = out_dir / "collect.json"
    if not collect_path.exists():
        raise InvalidProfileError(
            pointer=str(out_dir), reason="image derived eval needs collect.json with rows[]"
        )
    collect = json.loads(collect_path.read_text(encoding="utf-8"))
    rows = collect.get("rows") or []
    if not rows:
        raise InvalidProfileError(
            pointer=str(out_dir), reason="collect.json has no image rows to sample"
        )

    cache_path = out_dir / DERIVED_IMAGE_QUERIES_FILE
    cache: dict[str, str] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = rec.get("image_path")
            cap = rec.get("caption")
            if isinstance(p, str) and isinstance(cap, str):
                cache[p] = cap

    sample = rows[:_DERIVED_IMAGE_SAMPLE_SIZE]
    sample_paths = [str(r["image_path"]) for r in sample if r.get("image_path")]
    missing = [p for p in sample_paths if p not in cache]
    if missing:
        captions = caption_images(judge.provider, judge.model, missing)
        for p, cap in zip(missing, captions, strict=True):
            cache[p] = cap
        with cache_path.open("a", encoding="utf-8") as f:
            for p in missing:
                f.write(
                    json.dumps({"image_path": p, "caption": cache[p]}, ensure_ascii=False) + "\n"
                )

    return [
        QueryWithExpectedIds(query=cache[p], relevant_ids=(p,))
        for p in sample_paths
        if cache.get(p)
    ]


def _load_qrels(path: Path) -> list[QueryWithExpectedIds]:
    if not path.exists():
        raise InvalidProfileError(pointer=str(path), reason="qrels file not found")
    out: list[QueryWithExpectedIds] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise InvalidProfileError(
                    pointer=f"{path}:{lineno}", reason=f"invalid JSON: {exc}"
                ) from exc
            query = record.get("query")
            relevant = record.get("relevant_ids") or []
            grade = int(record.get("grade") or 1)
            if not isinstance(query, str) or not query.strip():
                raise InvalidProfileError(
                    pointer=f"{path}:{lineno}", reason="missing 'query' string"
                )
            if not isinstance(relevant, list) or not all(
                isinstance(x, (str, int)) for x in relevant
            ):
                raise InvalidProfileError(
                    pointer=f"{path}:{lineno}",
                    reason="'relevant_ids' must be a list of strings or ints",
                )
            out.append(
                QueryWithExpectedIds(
                    query=query.strip(),
                    relevant_ids=tuple(str(x) for x in relevant),
                    grade=grade,
                )
            )
    return out


def _load_queries(path: Path) -> list[QueryWithExpectedIds]:
    if not path.exists():
        raise InvalidProfileError(pointer=str(path), reason="queries file not found")
    out: list[QueryWithExpectedIds] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            q = raw.strip()
            if q:
                out.append(QueryWithExpectedIds(query=q))
    return out


def _derive_queries(*, out_dir: Path, plan: dict[str, Any]) -> list[QueryWithExpectedIds]:
    schema = plan["schema"]
    try:
        docs = list(_iter_documents(plan=plan, run_dir=out_dir, sample=None, input_path=None))
    except (FileNotFoundError, RuntimeError) as exc:
        raise InvalidProfileError(
            pointer=str(out_dir),
            reason=(
                "cannot derive queries — corpus unavailable; "
                "pass --qrels or --queries, or re-run collect with --sample/--input"
            ),
        ) from exc
    return derive_queries_from_corpus(
        docs,
        text_field=schema["text_field"],
        id_field=schema["primary_key"],
        sample_size=_DERIVED_SAMPLE_SIZE,
    )


# --- Per-variant evaluation ------------------------------------------------


@dataclass
class _RowResult:
    label: str
    retrieval: dict[str, float] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    rag: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "retrieval": self.retrieval,
            "latency": self.latency,
            "rag": self.rag,
        }


def _evaluate_single(
    *,
    label: str,
    plan: dict[str, Any],
    queries: Sequence[QueryWithExpectedIds],
    concurrency: int,
    judge: JudgeConfig | None,
) -> dict[str, Any]:
    if _is_image_plan(plan):
        return _evaluate_single_image(
            label=label, plan=plan, queries=queries, concurrency=concurrency
        )

    client = MilvusClient(uri=plan["target_uri"])
    collection = plan["collection_name"]
    schema = plan["schema"]
    embedder = make_embedder(
        plan["embedding"]["provider"],
        plan["embedding"]["model"],
        plan["embedding"]["dim"],
    )
    use_hybrid = bool(plan.get("sparse_enabled"))

    def _search(q: str) -> list[Any]:
        if use_hybrid:
            return search_hybrid(
                client,
                collection,
                q,
                embedder,
                top_k=_TOP_K,
                vector_field=schema["vector_field"],
                sparse_field=schema.get("sparse_field") or "sparse",
                output_fields=[schema["text_field"], schema["primary_key"]],
                rerank=plan.get("reranker"),
            )
        return search_dense(
            client,
            collection,
            q,
            embedder,
            top_k=_TOP_K,
            vector_field=schema["vector_field"],
            output_fields=[schema["text_field"], schema["primary_key"]],
            rerank=plan.get("reranker"),
        )

    ranked_ids: list[list[str]] = []
    retrieved_texts: list[list[str]] = []
    pk = schema["primary_key"]
    text_field = schema["text_field"]
    for q in queries:
        hits = _search(q.query)
        ids = [str(h.fields.get(pk, h.id)) for h in hits]
        texts = [str(h.fields.get(text_field, "")) for h in hits]
        ranked_ids.append(ids)
        retrieved_texts.append(texts)

    retrieval = compute_retrieval_metrics(queries, ranked_ids, k=_TOP_K)
    latency = compute_latency(
        _search_only_timing(_search),
        [q.query for q in queries],
        concurrency=concurrency,
    )
    rag = _maybe_rag(judge, queries, retrieved_texts) if judge else None

    row = _RowResult(label=label, retrieval=retrieval, latency=latency, rag=rag)
    return row.to_dict()


def _evaluate_single_image(
    *,
    label: str,
    plan: dict[str, Any],
    queries: Sequence[QueryWithExpectedIds],
    concurrency: int,
) -> dict[str, Any]:
    """Image-flow eval: route text queries via CLIP-text, image queries via
    the image-to-image path.

    No RAG metrics (no text corpus to ground answers in). No hybrid (image
    collections always have sparse disabled). Queries whose `query_image_path`
    points at a missing file are skipped with a stderr warning and drop out
    of retrieval math; latency timing still runs over the surviving rows.
    """
    client = MilvusClient(uri=plan["target_uri"])
    collection = plan["collection_name"]
    schema = plan["schema"]
    embedding = plan["embedding"]
    pk = schema["primary_key"]
    vector_field = schema["vector_field"]
    model_id = str(embedding.get("model") or "ViT-B-32")
    device_hint = embedding.get("device_hint")
    provider = str(embedding["provider"]).lower()

    has_text_queries = any(q.query_image_path is None for q in queries)
    if provider != "clip-local" and has_text_queries:
        # Voyage multimodal query encoding for text lives behind its own API
        # call; the MVP scope (#14) gates evaluate on clip-local for text rows.
        raise BackendUnsupportedError(
            uri=provider,
            feature="image-evaluate text queries currently require clip-local (Voyage TODO)",
        )

    # Cache image-query embeddings so the same query file isn't re-encoded
    # when it appears in multiple qrels rows within one run.
    image_vec_cache: dict[str, list[float]] = {}
    skipped_missing: list[str] = []

    def _encode_text(text: str) -> list[float]:
        return embed_text_with_clip([text], model_id=model_id, device_hint=device_hint)[0]

    def _encode_image(query_image_path: str) -> list[float] | None:
        cached = image_vec_cache.get(query_image_path)
        if cached is not None:
            return cached
        p = Path(query_image_path)
        if not p.exists():
            logger.warning(
                "eval: skipping image-to-image row — query image not found: %s",
                query_image_path,
            )
            skipped_missing.append(query_image_path)
            return None
        from ..search import _encode_query_image

        vec = _encode_query_image(p.read_bytes(), plan)
        image_vec_cache[query_image_path] = vec
        return vec

    def _run_search(qvec: list[float]) -> list[str]:
        res = client.search(
            collection_name=collection,
            data=[qvec],
            anns_field=vector_field,
            limit=_TOP_K,
            output_fields=[pk],
        )
        ids: list[str] = []
        for batch in res or []:
            for hit in batch:
                entity = getattr(hit, "entity", None)
                pk_val = (
                    (entity.get(pk) if entity is not None else hit.get(pk))
                    if hasattr(hit, "get") or entity is not None
                    else None
                )
                ids.append(str(pk_val) if pk_val else "")
        return ids

    def _search_one(q: QueryWithExpectedIds) -> list[str]:
        if q.query_image_path:
            vec = _encode_image(q.query_image_path)
            if vec is None:
                return []
            return _run_search(vec)
        return _run_search(_encode_text(q.query))

    survivors: list[QueryWithExpectedIds] = []
    ranked_ids: list[list[str]] = []
    for q in queries:
        hits = _search_one(q)
        if q.query_image_path and not hits and q.query_image_path in skipped_missing:
            continue  # dropped entirely; no contribution to metrics or latency
        survivors.append(q)
        ranked_ids.append(hits)

    retrieval = compute_retrieval_metrics(survivors, ranked_ids, k=_TOP_K)
    # Latency uses a bytes-and-text agnostic lambda so concurrency timing
    # exercises both routing branches that a caller actually hits.
    latency = compute_latency(
        lambda label_str: _search_one(_query_by_label(survivors, label_str)),
        [_label_for(q) for q in survivors],
        concurrency=concurrency,
    )
    row = _RowResult(label=label, retrieval=retrieval, latency=latency, rag=None)
    out = row.to_dict()
    if skipped_missing:
        out["skipped_queries"] = list(skipped_missing)
    return out


def _label_for(q: QueryWithExpectedIds) -> str:
    return q.query_image_path or q.query


def _query_by_label(queries: Sequence[QueryWithExpectedIds], label: str) -> QueryWithExpectedIds:
    for q in queries:
        if _label_for(q) == label:
            return q
    # Fallback — shouldn't happen because labels come from the same list
    return queries[0]


def _search_only_timing(search_fn: Callable[[str], Any]) -> Callable[[str], Any]:
    """Pass-through; named separately for readability at the call site."""
    return search_fn


def _maybe_rag(
    judge: JudgeConfig,
    queries: Sequence[QueryWithExpectedIds],
    contexts: Sequence[Sequence[str]],
) -> dict[str, float] | None:
    # Answers aren't generated here — we use the top retrieved chunk as a
    # stand-in answer. Real RAG pipelines would swap in the generator's
    # output. This keeps Phase 5 honest about retrieval-supporting-generation
    # without pulling an LLM response path into scope.
    answers = [ctx[0] if ctx else "" for ctx in contexts]
    return compute_rag_quality(
        [q.query for q in queries],
        contexts,
        answers,
        judge,
    )


# --- Variant grammar -------------------------------------------------------


def _resolve_variants(*, compare_path: str | None, allow_large: bool) -> list[dict[str, Any]]:
    if not compare_path:
        return []
    path = Path(compare_path)
    if not path.exists():
        raise InvalidProfileError(pointer=str(path), reason="variants file not found")
    variants = _parse_variants_file(path)
    if len(variants) > _VARIANT_CAP and not allow_large:
        raise InvalidProfileError(
            pointer=str(path),
            reason=(
                f"variant count {len(variants)} exceeds cap {_VARIANT_CAP}; "
                "re-run with --allow-large if intentional"
            ),
        )
    return variants


def _parse_variants_file(path: Path) -> list[dict[str, Any]]:
    """Parse variants.yaml. Accepts YAML or JSON (YAML preferred).

    Shape:
        variants:
          - name: small-m
            overrides:
              index:
                params: {M: 8}
          - name: no-hybrid
            overrides:
              hybrid: false
    """
    text = path.read_text(encoding="utf-8")
    data = _load_yaml_or_json(text, path)
    variants_raw = data.get("variants") if isinstance(data, dict) else None
    if not isinstance(variants_raw, list) or not variants_raw:
        raise InvalidProfileError(
            pointer=str(path),
            reason="expected top-level `variants:` list with at least one entry",
        )

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, raw in enumerate(variants_raw):
        if not isinstance(raw, dict):
            raise InvalidProfileError(
                pointer=f"{path}#variants[{i}]", reason="variant must be a mapping"
            )
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise InvalidProfileError(
                pointer=f"{path}#variants[{i}]",
                reason="missing 'name' (non-empty string)",
            )
        if name in seen:
            raise InvalidProfileError(
                pointer=f"{path}#variants[{i}]",
                reason=f"duplicate variant name {name!r}",
            )
        seen.add(name)

        overrides = raw.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise InvalidProfileError(
                pointer=f"{path}#variants[{i}].overrides",
                reason="'overrides' must be a mapping",
            )
        _validate_override_axes(overrides, pointer=f"{path}#variants[{i}].overrides")
        out.append({"name": name, "overrides": overrides})
    return out


def _validate_override_axes(overrides: dict[str, Any], *, pointer: str) -> None:
    extra = set(overrides) - _ALLOWED_OVERRIDE_AXES
    if extra:
        raise InvalidProfileError(
            pointer=pointer,
            reason=(
                f"unsupported override axes: {sorted(extra)}; "
                f"allowed: {sorted(_ALLOWED_OVERRIDE_AXES)}"
            ),
        )
    for axis, allowed in _ALLOWED_NESTED.items():
        nested = overrides.get(axis)
        if isinstance(nested, dict):
            nested_extra = set(nested) - allowed
            if nested_extra:
                raise InvalidProfileError(
                    pointer=f"{pointer}.{axis}",
                    reason=(
                        f"unsupported nested keys: {sorted(nested_extra)}; "
                        f"allowed under '{axis}': {sorted(allowed)}"
                    ),
                )


def _load_yaml_or_json(text: str, path: Path) -> Any:
    try:
        import yaml  # type: ignore[import-untyped]

        return yaml.safe_load(text)
    except ImportError:
        # Fallback: JSON-only
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise InvalidProfileError(
                pointer=str(path),
                reason=(
                    "PyYAML not installed and file is not valid JSON; install pyyaml or supply JSON"
                ),
            ) from exc


def _apply_variant(plan: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    """Return a new plan with the variant's overrides merged in.

    Kept minimal: overrides are merged shallow-ly at the top level and one
    level deep under `embedding`/`index` (matching the grammar). Anything
    fancier belongs in Phase 6 sub-plan materialisation.
    """
    overrides = variant.get("overrides") or {}
    merged = dict(plan)
    for axis, value in overrides.items():
        if axis == "embedding" and isinstance(value, dict):
            merged["embedding"] = {**plan.get("embedding", {}), **value}
        elif axis == "index" and isinstance(value, dict):
            merged["index"] = {**plan.get("index", {}), **value}
            if "params" in value and isinstance(value["params"], dict):
                merged["index"]["params"] = {
                    **plan.get("index", {}).get("params", {}),
                    **value["params"],
                }
        elif axis == "hybrid":
            merged["sparse_enabled"] = bool(value)
        elif axis == "reranker":
            merged["reranker"] = value if value else None
    return merged


# --- Report assembly -------------------------------------------------------


def _build_report(
    *,
    out_dir: Path,
    queries: Sequence[QueryWithExpectedIds],
    base_row: dict[str, Any],
    variant_rows: list[dict[str, Any]],
    derived: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "derived": derived,
        "latency_metrics": base_row["latency"],
        "query_count": len(queries),
        "rag_metrics": base_row["rag"],
        "retrieval_metrics": base_row["retrieval"],
        "run_id": out_dir.name,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "variants": variant_rows,
    }
    if base_row.get("skipped_queries"):
        report["skipped_queries"] = list(base_row["skipped_queries"])
        report["skipped_count"] = len(base_row["skipped_queries"])
    return report


def _write_report(out_dir: Path, report: dict[str, Any]) -> None:
    (out_dir / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "eval_report.md").write_text(_render_markdown(report), encoding="utf-8")


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        f"# Evaluation Report — {report['run_id']}",
        "",
        f"- Timestamp: `{report['timestamp']}`",
        f"- Query count: {report['query_count']}",
        f"- Derived query set: **{str(report['derived']).lower()}**",
        "",
        "## Decision table",
        "",
        "| variant | recall@10 | p95 (ms) | faithfulness | cost/query |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.append(
        _decision_row(
            "base",
            report["retrieval_metrics"],
            report["latency_metrics"],
            report["rag_metrics"],
        )
    )
    for v in report["variants"]:
        lines.append(_decision_row(v["label"], v["retrieval"], v["latency"], v["rag"]))

    if report["derived"]:
        lines += [
            "",
            "> **Note:** queries were derived from the corpus (first sentence of each "
            "sampled doc); retrieval metrics measure self-recall, not real retrieval "
            "quality. Pass `--qrels <path>` for a labelled eval.",
        ]

    if report["rag_metrics"] is None:
        lines += [
            "",
            "RAG-quality metrics omitted — pass `--judge-llm <provider>:<model>` to enable.",
        ]
    return "\n".join(lines) + "\n"


def _decision_row(
    label: str,
    retrieval: dict[str, float],
    latency: dict[str, float],
    rag: dict[str, float] | None,
) -> str:
    recall = retrieval.get("recall@10")
    p95 = latency.get("p95_ms")
    faith = (rag or {}).get("faithfulness")
    recall_cell = f"{recall:.3f}" if recall is not None else "—"
    p95_cell = f"{p95:.1f}" if p95 is not None else "—"
    faith_cell = f"{faith:.3f}" if faith is not None else "—"
    # cost/query is a placeholder until Phase 5 wires a real cost estimator
    cost_cell = "—"
    return f"| {label} | {recall_cell} | {p95_cell} | {faith_cell} | {cost_cell} |"


def _append_observability_sample(out_dir: Path, report: dict[str, Any]) -> None:
    path = out_dir / "observability.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Don't clobber an unreadable obs file — the deployer owns its shape
        logger.warning("observability.json is not valid JSON; skipping append")
        return
    samples = data.setdefault("latency_samples", [])
    samples.append(
        {
            "source": "evaluate",
            "run_id": report["run_id"],
            "timestamp": report["timestamp"],
            "p50_ms": report["latency_metrics"].get("p50_ms"),
            "p95_ms": report["latency_metrics"].get("p95_ms"),
            "p99_ms": report["latency_metrics"].get("p99_ms"),
            "count": report["latency_metrics"].get("count"),
        }
    )
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


# --- Backend guard ---------------------------------------------------------


def _refuse_milvus_lite(uri: str, backend: Backend) -> None:
    # Milvus Lite URIs are file paths (e.g. `./milvus.db`); Standalone and
    # Cloud use http/https. Phase 5's spec says latency metrics against
    # Lite are meaningless, so refuse up front.
    if not uri.startswith(("http://", "https://")):
        raise BackendUnsupportedError(
            uri=uri,
            feature="Milvus Lite target — Phase 5 needs Standalone or Cloud",
        )
