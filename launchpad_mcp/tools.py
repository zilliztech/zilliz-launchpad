"""Pure-Python wrappers around each launchpad phase.

Each ``run_*`` here calls the matching ``run_<phase>`` helper that already
exists in ``skills/zilliz-launchpad/scripts/lib/phases/<phase>.py`` — those
helpers write the canonical ``<phase>.json`` artifact and return it as a
``dict``. We do not touch ``lib/`` from this package.

Phase → underlying callable (no Typer in the call path):
  collect    → lib.phases.collect.run_collect
  configure  → lib.phases.configure.run_configure
  plan       → lib.phases.plan.run_plan
  execute    → lib.phases.execute.run_execute  (+ run_execute_append when append=True)
  evaluate   → lib.phases.evaluate.run_evaluate
  deploy     → lib.phases.deploy.run_deploy

Each wrapper returns ``{"run_dir": <abs path str>, "artifact": <dict>}``.

All ``LaunchpadError`` subclasses propagate untouched so the server layer
can attach ``err.to_dict()`` to the MCP tool error. Any other exception is
wrapped in :class:`InternalError` so the host always sees a structured
``{"code", "message"}`` envelope.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Make the lib package importable regardless of CWD. pyproject already sets
# this on the pytest pythonpath; we replicate it here so `python -m
# launchpad_mcp.server` works from any directory.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "zilliz-launchpad" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.errors import LaunchpadError  # noqa: E402
from lib.phases import collect as _collect  # noqa: E402
from lib.phases import configure as _configure  # noqa: E402
from lib.phases import deploy as _deploy  # noqa: E402
from lib.phases import evaluate as _evaluate  # noqa: E402
from lib.phases import execute as _execute  # noqa: E402
from lib.phases import plan as _plan  # noqa: E402
from lib.run_dir import new_run_dir, resolve_run_dir  # noqa: E402

__all__ = [
    "InternalError",
    "LaunchpadError",
    "run_collect",
    "run_configure",
    "run_deploy",
    "run_evaluate",
    "run_execute",
    "run_plan",
]


class InternalError(Exception):
    """Catch-all wrapper for non-LaunchpadError exceptions."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {"code": "internal_error", "message": self.message}


class _MissingInputError(LaunchpadError):
    code = "missing_input"


def _result(run_dir: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    return {"run_dir": str(run_dir), "artifact": artifact}


# --- Phase 1 ---------------------------------------------------------------


def run_collect(
    *,
    sample: str | None = None,
    input_path: str | None = None,
    run_dir: str | None = None,
    with_thumbnails: bool | None = None,
    thumbnail_cap_rows: int = 5000,
    split_markdown_headings: bool = False,
) -> dict[str, Any]:
    if sample is None and input_path is None:
        raise _MissingInputError("Pass `sample` or `input_path`")
    out = resolve_run_dir(run_dir) if run_dir else new_run_dir(label="collect")
    artifact = _collect.run_collect(
        input_path=input_path,
        sample=sample,
        out_dir=out,
        with_thumbnails=with_thumbnails,
        thumbnail_cap_rows=thumbnail_cap_rows,
        split_markdown_headings=split_markdown_headings,
    )
    return _result(out, artifact)


# --- Phase 2 ---------------------------------------------------------------


def run_configure(
    *,
    run_dir: str,
    from_json: str | None = None,
    use_case: str | None = None,
    dataset_size: int | None = None,
    deployment_target: str | None = None,
    hybrid_preference: str | None = None,
    reranker_preference: str | None = None,
    frame_interval_seconds: float | None = None,
    max_frames_per_video: int | None = None,
    sampling_strategy: str | None = None,
    scene_threshold: float | None = None,
) -> dict[str, Any]:
    out = resolve_run_dir(run_dir)
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
    artifact = _configure.run_configure(from_json=from_json, out_dir=out, overrides=overrides)
    return _result(out, artifact)


# --- Phase 3 ---------------------------------------------------------------


def run_plan(*, run_dir: str) -> dict[str, Any]:
    out = resolve_run_dir(run_dir)
    artifact = _plan.run_plan(out_dir=out)
    return _result(out, artifact)


# --- Phase 4 ---------------------------------------------------------------


def run_execute(
    *,
    run_dir: str,
    sample: str | None = None,
    input_path: str | None = None,
    ui_port: int = 8000,
    no_ui: bool = False,
    frame_progress: bool = False,
    append: bool = False,
) -> dict[str, Any]:
    out = resolve_run_dir(run_dir)
    if append:
        if input_path is None:
            raise _MissingInputError("`append=True` requires `input_path`")
        artifact = _execute.run_execute_append(out_dir=out, input_path=input_path)
        return _result(out, artifact)
    artifact = _execute.run_execute(
        out_dir=out,
        sample=sample,
        input_path=input_path,
        ui_port=ui_port,
        start_ui=not no_ui,
        frame_progress=frame_progress,
    )
    return _result(out, artifact)


# --- Phase 5 ---------------------------------------------------------------


def run_evaluate(
    *,
    run_dir: str,
    qrels_path: str | None = None,
    queries_path: str | None = None,
    concurrency: int = 4,
    judge_llm: str | None = None,
    compare_path: str | None = None,
    allow_large: bool = False,
) -> dict[str, Any]:
    out = resolve_run_dir(run_dir)
    artifact = _evaluate.run_evaluate(
        out_dir=out,
        qrels_path=qrels_path,
        queries_path=queries_path,
        concurrency=concurrency,
        judge_llm=judge_llm,
        compare_path=compare_path,
        allow_large=allow_large,
    )
    return _result(out, artifact)


# --- Phase 6 ---------------------------------------------------------------


def run_deploy(
    *,
    run_dir: str,
    cluster_id: str | None = None,
    create: bool = False,
    confirm: bool = False,
    stop_local: bool = False,
) -> dict[str, Any]:
    out = resolve_run_dir(run_dir)
    artifact = _deploy.run_deploy(
        out_dir=out,
        cluster_id=cluster_id,
        create=create,
        confirm=confirm,
        stop_local=stop_local,
    )
    return _result(out, artifact)
