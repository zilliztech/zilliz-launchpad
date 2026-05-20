"""FastMCP stdio server exposing the six zilliz-launchpad phases as tools.

One tool per phase. Each tool's input matches the corresponding CLI flag
surface (kebab-case CLI → snake_case JSON). Each tool returns
``{"run_dir": str, "artifact": dict}`` where ``artifact`` is the JSON
parsed from ``<run_dir>/<phase>.json``.

Error envelope
--------------

On failure a tool raises a ``ToolError`` whose message is a JSON-encoded
envelope. The envelope is exactly the dict produced by
``LaunchpadError.to_dict()`` — i.e. ``{"code", "message", ...}`` — and is
identical to the JSON the CLI prints to stderr today. Non-LaunchpadError
exceptions are wrapped as ``{"code": "internal_error", "message": ...}``
so the host never sees an opaque Python traceback.

Hosts that want the structured envelope should ``json.loads`` the error
text content. The unit/smoke tests below call the wrapper functions
directly, so they observe the original ``LaunchpadError`` (or
``InternalError``) before FastMCP serializes it.

Run
---

::

    python -m launchpad_mcp.server      # stdio transport
    zilliz-launchpad-mcp                  # console script (if installed)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from . import tools
from .tools import InternalError, LaunchpadError  # re-exported below

mcp = FastMCP("zilliz-launchpad")


def _envelope_error(exc: BaseException) -> ToolError:
    if isinstance(exc, tools.LaunchpadError):
        payload = exc.to_dict()
    else:
        payload = {"code": "internal_error", "message": str(exc)}
    return ToolError(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _invoke(fn: Callable[..., dict[str, Any]], /, **kwargs: Any) -> dict[str, Any]:
    """Call a `tools.run_*` helper and convert any failure into a ToolError
    whose message is the structured `{code, message, ...}` envelope.
    """
    try:
        return fn(**kwargs)
    except tools.LaunchpadError as exc:
        raise _envelope_error(exc) from exc
    except Exception as exc:  # noqa: BLE001  (we re-raise as ToolError)
        raise _envelope_error(exc) from exc


@mcp.tool(
    name="collect",
    description=(
        "Phase 1 — analyze a sample file or directory and produce collect.json. "
        "Starts a new run_dir (or appends to an existing one) and returns the "
        "absolute run_dir path plus the collect.json contents. Call this first; "
        "every later phase needs the run_dir returned here."
    ),
)
def collect(
    sample: str | None = None,
    input_path: str | None = None,
    run_dir: str | None = None,
    with_thumbnails: bool | None = None,
    thumbnail_cap_rows: int = 5000,
    split_markdown_headings: bool = False,
) -> dict[str, Any]:
    """Phase 1: analyze sample data; mirrors `zilliz_ops.py collect`.

    Pass exactly one of `sample` (bundled sample name) or `input_path`
    (path to a JSONL/CSV/MD/PDF/TXT file or a directory of images / videos).
    """
    return _invoke(
        tools.run_collect,
        sample=sample,
        input_path=input_path,
        run_dir=run_dir,
        with_thumbnails=with_thumbnails,
        thumbnail_cap_rows=thumbnail_cap_rows,
        split_markdown_headings=split_markdown_headings,
    )


@mcp.tool(
    name="configure",
    description=(
        "Phase 2 — capture requirements and write configure.json. "
        "Requires a run_dir created by `collect`. Overrides default knobs "
        "(use_case, dataset_size, deployment_target, hybrid/reranker preferences, "
        "and video sampling)."
    ),
)
def configure(
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
    """Phase 2: capture requirements; mirrors `zilliz_ops.py configure`."""
    return _invoke(
        tools.run_configure,
        run_dir=run_dir,
        from_json=from_json,
        use_case=use_case,
        dataset_size=dataset_size,
        deployment_target=deployment_target,
        hybrid_preference=hybrid_preference,
        reranker_preference=reranker_preference,
        frame_interval_seconds=frame_interval_seconds,
        max_frames_per_video=max_frames_per_video,
        sampling_strategy=sampling_strategy,
        scene_threshold=scene_threshold,
    )


@mcp.tool(
    name="plan",
    description=(
        "Phase 3 — turn configure.json into plan.json + plan.md. "
        "Requires a run_dir whose configure phase has already run. "
        "Returns the parsed plan dict."
    ),
)
def plan(run_dir: str) -> dict[str, Any]:
    """Phase 3: produce plan artifacts; mirrors `zilliz_ops.py plan`."""
    return _invoke(tools.run_plan, run_dir=run_dir)


@mcp.tool(
    name="execute",
    description=(
        "Phase 4 — apply the plan: create collection, build index, ingest, "
        "and optionally start a local search-UI sidecar. Requires a run_dir "
        "whose plan phase has already run. Set `append=True` together with "
        "`input_path` to ingest additional rows into the existing collection."
    ),
)
def execute(
    run_dir: str,
    sample: str | None = None,
    input_path: str | None = None,
    ui_port: int = 8000,
    no_ui: bool = False,
    frame_progress: bool = False,
    append: bool = False,
) -> dict[str, Any]:
    """Phase 4: apply plan; mirrors `zilliz_ops.py execute`."""
    return _invoke(
        tools.run_execute,
        run_dir=run_dir,
        sample=sample,
        input_path=input_path,
        ui_port=ui_port,
        no_ui=no_ui,
        frame_progress=frame_progress,
        append=append,
    )


@mcp.tool(
    name="evaluate",
    description=(
        "Phase 5 — measure retrieval quality and write eval_report.{json,md}. "
        "Requires a run_dir whose execute phase has produced an ingested "
        "collection. Optionally compare against alternative plan variants "
        "via `compare_path`."
    ),
)
def evaluate(
    run_dir: str,
    qrels_path: str | None = None,
    queries_path: str | None = None,
    concurrency: int = 4,
    judge_llm: str | None = None,
    compare_path: str | None = None,
    allow_large: bool = False,
) -> dict[str, Any]:
    """Phase 5: evaluate retrieval; mirrors `zilliz_ops.py evaluate`."""
    return _invoke(
        tools.run_evaluate,
        run_dir=run_dir,
        qrels_path=qrels_path,
        queries_path=queries_path,
        concurrency=concurrency,
        judge_llm=judge_llm,
        compare_path=compare_path,
        allow_large=allow_large,
    )


@mcp.tool(
    name="deploy",
    description=(
        "Phase 6 — provision or reuse a Zilliz Cloud cluster and migrate the "
        "local collection into it. Requires a run_dir whose execute phase has "
        "produced a local collection. Pass `cluster_id` to target an existing "
        "cluster, or `create=True` (with `confirm=True`) to provision a new one."
    ),
)
def deploy(
    run_dir: str,
    cluster_id: str | None = None,
    create: bool = False,
    confirm: bool = False,
    stop_local: bool = False,
) -> dict[str, Any]:
    """Phase 6: deploy to Zilliz Cloud; mirrors `zilliz_ops.py deploy`."""
    return _invoke(
        tools.run_deploy,
        run_dir=run_dir,
        cluster_id=cluster_id,
        create=create,
        confirm=confirm,
        stop_local=stop_local,
    )


def main() -> None:
    """Console-script entrypoint: run the stdio MCP server."""
    mcp.run("stdio")


__all__ = ["mcp", "main", "InternalError", "LaunchpadError"]


if __name__ == "__main__":
    main()
