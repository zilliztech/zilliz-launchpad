"""Phase 2: Configure — structured requirement capture.

Writes `configure.json` into the provided run dir. Two modes:
  - `--from-json <file>`  : non-interactive (agent-provided answers)
  - default (no tty)      : reads a minimal set of env-var-style inputs
                            or writes a template for the agent to fill
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import typer

from .. import zilliz_cli
from ..cli import fail as _cli_fail
from ..errors import InvalidProfileError, LaunchpadError
from ..run_dir import latest_run_dir, new_run_dir, resolve_run_dir

DEFAULTS: dict[str, Any] = {
    "use_case": "rag",
    "query_patterns": ["long-natural-language"],
    "dataset_size": 20,
    "deployment_target": "local-standalone",
    "latency_target_ms": None,
    "embedding_preference": None,
    "hybrid_preference": "auto",
    "reranker_preference": "auto",
}

KNOWN_USE_CASES = {
    "rag",
    "semantic-search",
    "hybrid-search",
    "recommendation",
    "image-search",
    "video-search",
}

IMAGE_USE_CASES = {"image-search"}
VIDEO_USE_CASES = {"video-search"}
TEXT_DATA_SHAPES = {"jsonl", "csv", "text"}

VIDEO_DEFAULTS: dict[str, Any] = {
    "frame_interval_seconds": 2.0,
    "max_frames_per_video": 600,
    "sampling_strategy": "every_n_seconds",
    "scene_threshold": 0.3,
}

CLOUD_TARGETS = {"zilliz-serverless", "zilliz-dedicated", "zilliz-byoc"}
TARGET_TIER_MAP = {
    "zilliz-serverless": "SERVERLESS",
    "zilliz-dedicated": "DEDICATED",
    "zilliz-byoc": "BYOC",
}

logger = logging.getLogger(__name__)


def _pick_cluster(
    clusters: list[dict[str, Any]],
    target_tier: str | None,
    preferred_id: str | None,
) -> dict[str, Any] | None:
    if not clusters:
        return None
    if preferred_id:
        for c in clusters:
            if str(c.get("clusterId") or c.get("id") or c.get("cluster_id")) == preferred_id:
                return c
    if target_tier:
        matches = [
            c
            for c in clusters
            if str(c.get("tier") or c.get("clusterType") or "").upper() == target_tier
        ]
        if matches:
            return matches[0]
    return clusters[0]


def _discover_cluster(
    deployment_target: str,
    preferred_id: str | None,
) -> dict[str, Any] | None:
    """Attempt CLI-backed cluster discovery. Returns enrichment data or None.

    Falls back silently to the prompt flow on any CLI error.
    """
    if deployment_target not in CLOUD_TARGETS:
        return None
    try:
        if not zilliz_cli.is_available():
            return None
        clusters = zilliz_cli.cluster_list()
    except LaunchpadError as exc:
        logger.info("zilliz CLI discovery skipped: %s", exc)
        return None
    if not clusters:
        logger.info("zilliz CLI returned no clusters; falling back to prompt flow")
        return None
    target_tier = TARGET_TIER_MAP.get(deployment_target)
    picked = _pick_cluster(clusters, target_tier, preferred_id)
    if picked is None:
        return None
    cluster_id = str(picked.get("clusterId") or picked.get("id") or picked.get("cluster_id") or "")
    uri = picked.get("connectAddress") or picked.get("uri") or picked.get("endpoint")
    if not cluster_id:
        return None
    return {
        "cluster_id": cluster_id,
        "resolved_from_cli": True,
        **({"target_uri": str(uri)} if uri else {}),
    }


def _read_collect_data_shape(out_dir: Path) -> str | None:
    collect_path = out_dir / "collect.json"
    if not collect_path.exists():
        return None
    try:
        with collect_path.open("r", encoding="utf-8") as f:
            shape = json.load(f).get("data_shape")
    except (json.JSONDecodeError, OSError):
        return None
    return str(shape) if shape is not None else None


def _apply_image_search_defaults(data: dict[str, Any]) -> None:
    """Force hybrid off and reranker none for image collections.

    Warns to stderr if the user passed conflicting preferences so the
    override is visible, not silent.
    """
    silent_hybrid = (None, False, "off", "none", "auto")
    silent_reranker = (None, "none", "off", "auto")
    conflicting: list[tuple[str, Any]] = []
    if data.get("hybrid_preference") not in silent_hybrid:
        conflicting.append(("hybrid_preference", data["hybrid_preference"]))
    if data.get("reranker_preference") not in silent_reranker:
        conflicting.append(("reranker_preference", data["reranker_preference"]))
    data["hybrid_preference"] = False
    data["reranker_preference"] = "none"
    for key, value in conflicting:
        print(
            f"warn: --use-case image-search overrides {key}={value!r} → forced off",
            file=sys.stderr,
        )


def _apply_video_search_defaults(data: dict[str, Any]) -> None:
    """Force hybrid off / reranker none for video collections and backfill knobs.

    Mirrors ``_apply_image_search_defaults`` and additionally populates the
    video-specific sampling knobs with their defaults when the user didn't
    supply overrides. Emits a stderr warning for each conflicting preference
    so the override is visible instead of silent.
    """
    silent_hybrid = (None, False, "off", "none", "auto")
    silent_reranker = (None, "none", "off", "auto")
    conflicting: list[tuple[str, Any]] = []
    if data.get("hybrid_preference") not in silent_hybrid:
        conflicting.append(("hybrid_preference", data["hybrid_preference"]))
    if data.get("reranker_preference") not in silent_reranker:
        conflicting.append(("reranker_preference", data["reranker_preference"]))
    data["hybrid_preference"] = False
    data["reranker_preference"] = "none"
    for key, default in VIDEO_DEFAULTS.items():
        if data.get(key) in (None, ""):
            data[key] = default
    for key, value in conflicting:
        print(
            f"warn: --use-case video-search overrides {key}={value!r} → forced off",
            file=sys.stderr,
        )

    if data.get("sampling_strategy") == "scene_change":
        import shutil  # noqa: PLC0415

        if shutil.which("ffmpeg") is None:
            print(
                "warn: sampling_strategy=scene_change requested but ffmpeg is not on PATH; "
                "collect will fall back to every_n_seconds",
                file=sys.stderr,
            )


def _validate_modality(data: dict[str, Any], data_shape: str | None) -> None:
    use_case = data.get("use_case")
    if data_shape is None:
        return
    if data_shape == "image_dir" and use_case not in IMAGE_USE_CASES:
        raise InvalidProfileError(
            "use_case",
            f"collect detected image_dir but use_case={use_case!r} expects text — "
            f"set --use-case image-search or pick a text input",
        )
    if data_shape == "video_dir" and use_case not in VIDEO_USE_CASES:
        raise InvalidProfileError(
            "use_case",
            f"collect detected video_dir but use_case={use_case!r} expects text — "
            f"set --use-case video-search or pick a text input",
        )
    if data_shape in TEXT_DATA_SHAPES and use_case in IMAGE_USE_CASES:
        raise InvalidProfileError(
            "use_case",
            f"collect detected {data_shape} but use_case=image-search expects an image directory",
        )
    if data_shape in TEXT_DATA_SHAPES and use_case in VIDEO_USE_CASES:
        raise InvalidProfileError(
            "use_case",
            f"collect detected {data_shape} but use_case=video-search expects a video directory",
        )


def run_configure(
    *,
    from_json: str | None,
    out_dir: Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any]
    if from_json:
        with Path(from_json).open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = dict(DEFAULTS)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                data[k] = v

    use_case = data.get("use_case")
    if use_case not in KNOWN_USE_CASES:
        raise InvalidProfileError(
            "use_case",
            f"unknown use_case={use_case!r}. Pick one of {sorted(KNOWN_USE_CASES)}",
        )

    _validate_modality(data, _read_collect_data_shape(out_dir))

    if use_case in IMAGE_USE_CASES:
        _apply_image_search_defaults(data)
    elif use_case in VIDEO_USE_CASES:
        _apply_video_search_defaults(data)

    deployment_target = data.get("deployment_target", "local-standalone")
    preferred_id = data.get("cluster_id")
    discovery = _discover_cluster(deployment_target, preferred_id)
    if discovery:
        data["cluster_id"] = discovery["cluster_id"]
        data["resolved_from_cli"] = True
        if "target_uri" in discovery and not data.get("target_uri"):
            data["target_uri"] = discovery["target_uri"]

    out = out_dir / "configure.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    return data


def register(app: typer.Typer) -> None:
    """Attach the Phase 2 ``configure`` subcommand to the shared app."""

    @app.command()
    def configure(
        from_json: Path | None = typer.Option(None, "--from-json", help="Pre-filled answers"),  # noqa: B008
        run_dir: str | None = typer.Option(None, "--run-dir"),
        use_case: str | None = typer.Option(None, "--use-case"),
        dataset_size: int | None = typer.Option(None, "--dataset-size"),
        deployment_target: str | None = typer.Option(None, "--deployment"),
        hybrid_preference: str | None = typer.Option(None, "--hybrid"),
        reranker_preference: str | None = typer.Option(None, "--reranker"),
        frame_interval_seconds: float | None = typer.Option(None, "--frame-interval-seconds"),
        max_frames_per_video: int | None = typer.Option(None, "--max-frames-per-video"),
        sampling_strategy: str | None = typer.Option(None, "--sampling-strategy"),
        scene_threshold: float | None = typer.Option(None, "--scene-threshold"),
    ) -> None:
        """Phase 2 — capture requirements."""
        out = (
            resolve_run_dir(run_dir)
            if run_dir
            else (latest_run_dir() or new_run_dir(label="configure"))
        )
        overrides = {
            "use_case": use_case,
            "dataset_size": dataset_size,
            "deployment_target": deployment_target,
            "hybrid_preference": hybrid_preference,
            "reranker_preference": reranker_preference,
            "frame_interval_seconds": frame_interval_seconds,
            "max_frames_per_video": max_frames_per_video,
            "sampling_strategy": sampling_strategy,
            "scene_threshold": scene_threshold,
        }
        try:
            data = run_configure(
                from_json=str(from_json) if from_json else None, out_dir=out, overrides=overrides
            )
        except LaunchpadError as e:
            _cli_fail(e)
        typer.echo(f"run-dir: {out}")
        typer.echo(f"deployment_target: {data['deployment_target']}")
