"""zilliz-launchpad CLI.

Phase-to-subcommand mapping:
  collect    → Phase 1
  configure  → Phase 2
  plan       → Phase 3
  execute    → Phase 4
  evaluate   → Phase 5
  deploy     → Phase 6

Usage (typical):
  python scripts/zilliz_ops.py collect --sample movies
  python scripts/zilliz_ops.py configure --from-json configure.json
  python scripts/zilliz_ops.py plan
  python scripts/zilliz_ops.py execute --sample movies
  python scripts/zilliz_ops.py evaluate
  python scripts/zilliz_ops.py deploy --cluster-id <id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from lib.errors import CliErrorEnvelope, InvalidProfileError, LaunchpadError
from lib.phases import collect as phase_collect
from lib.phases import configure as phase_configure
from lib.phases import deploy as phase_deploy
from lib.phases import evaluate as phase_evaluate
from lib.phases import execute as phase_execute
from lib.phases import plan as phase_plan
from lib.run_dir import latest_run_dir, new_run_dir, resolve_run_dir

app = typer.Typer(help="zilliz-launchpad — Phases 1–6")


def _fail(err: LaunchpadError) -> None:
    env = CliErrorEnvelope.from_error(err)
    print(env.to_json(), file=sys.stderr)
    raise typer.Exit(code=1)


@app.command()
def collect(
    sample: str | None = typer.Option(None, "--sample", "-s", help="Bundled sample name"),
    input: Path | None = typer.Option(None, "--input", "-i", help="Path to user data"),  # noqa: B008
    run_dir: str | None = typer.Option(None, "--run-dir", help="Existing run dir; default = new"),
    with_thumbnails: bool | None = typer.Option(
        None,
        "--with-thumbnails/--no-thumbnails",
        help="Image dir only. Default: on for ≤5000 images, off above.",
    ),
    thumbnail_cap_rows: int = typer.Option(
        5000,
        "--thumbnail-cap-rows",
        help="Image dir only. Auto-disable thumbnails above this many images.",
    ),
) -> None:
    """Phase 1 — analyze sample data."""
    if sample is None and input is None:
        print(
            json.dumps({"code": "missing_input", "message": "Pass --sample or --input"}),
            file=sys.stderr,
        )
        raise typer.Exit(code=2)
    out = resolve_run_dir(run_dir) if run_dir else new_run_dir(label="collect")
    try:
        result = phase_collect.run_collect(
            input_path=str(input) if input else None,
            sample=sample,
            out_dir=out,
            with_thumbnails=with_thumbnails,
            thumbnail_cap_rows=thumbnail_cap_rows,
        )
    except LaunchpadError as e:
        _fail(e)
    typer.echo(f"run-dir: {out}")
    if result.get("data_shape") == "image_dir":
        typer.echo("data_shape: image_dir")
        typer.echo(f"images: {result['record_count_estimate']}")
        typer.echo(f"thumbnails_included: {result['thumbnails_included']}")
    else:
        typer.echo(f"suggested_primary_key: {result['suggested_primary_key']}")
        typer.echo(f"suggested_text_field: {result['suggested_text_field']}")


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
        data = phase_configure.run_configure(
            from_json=str(from_json) if from_json else None, out_dir=out, overrides=overrides
        )
    except LaunchpadError as e:
        _fail(e)
    typer.echo(f"run-dir: {out}")
    typer.echo(f"deployment_target: {data['deployment_target']}")


@app.command()
def plan(
    run_dir: str | None = typer.Option(None, "--run-dir"),
) -> None:
    """Phase 3 — produce plan.{json,md}."""
    out = resolve_run_dir(run_dir)
    try:
        result = phase_plan.run_plan(out_dir=out)
    except LaunchpadError as e:
        _fail(e)
    typer.echo(f"run-dir: {out}")
    typer.echo(f"index: {result['index']['type']} {result['index']['params']}")
    typer.echo(f"sparse: {result['sparse_enabled']}")


@app.command()
def execute(
    run_dir: str | None = typer.Option(None, "--run-dir"),
    sample: str | None = typer.Option(None, "--sample", "-s"),
    input: Path | None = typer.Option(None, "--input", "-i"),  # noqa: B008
    ui_port: int = typer.Option(8000, "--ui-port"),
    no_ui: bool = typer.Option(False, "--no-ui", help="Skip starting the sidecar"),
    prefetch_models: bool = typer.Option(
        False,
        "--prefetch-models",
        help="Download CLIP / image-embedding weights without ingesting and exit.",
    ),
    frame_progress: bool = typer.Option(
        False,
        "--frame-progress",
        help="Emit one log line per frame batch during video ingest (noisy).",
    ),
) -> None:
    """Phase 4 — apply plan and start UI sidecar."""
    if prefetch_models:
        from lib.embeddings import prefetch_clip

        try:
            prefetch_clip()
        except LaunchpadError as e:
            _fail(e)
        typer.echo("✓ CLIP weights cached")
        return

    out = resolve_run_dir(run_dir)
    try:
        report = phase_execute.run_execute(
            out_dir=out,
            sample=sample,
            input_path=str(input) if input else None,
            ui_port=ui_port,
            start_ui=not no_ui,
            frame_progress=frame_progress,
        )
    except LaunchpadError as e:
        _fail(e)
    typer.echo(f"run-dir: {out}")
    typer.echo(f"collection: {report['collection_status']}")
    typer.echo(f"index: {report['index_status']}")
    typer.echo(f"ingest: {report['ingest']}")
    if report.get("smoke_hits"):
        top = report["smoke_hits"][0]
        typer.echo(f"✓ Top-1: {top['id']} score={top['score']:.4f}")
    elif report.get("ingest_path") in ("image-batch", "video-batch"):
        typer.echo(f"({report['ingest_path']} — smoke query is best-effort)")
    else:
        typer.echo("✗ Smoke query returned zero hits")
        raise typer.Exit(code=3)
    if report.get("sidecar_pid"):
        typer.echo(f"UI sidecar pid {report['sidecar_pid']} on port {report['ui_port']}")
        typer.echo("Start the Next.js UI: (cd scripts/ui && pnpm install && pnpm dev)")
    target_uri = report.get("target_uri", "")
    if target_uri and "cloud.zilliz.com" not in target_uri:
        typer.echo(
            "Tip: run ./start_milvus.sh attu up to inspect the collection in Attu "
            "(http://localhost:8000)"
        )


@app.command()
def evaluate(
    run_dir: str | None = typer.Option(None, "--run-dir"),
    qrels: Path | None = typer.Option(None, "--qrels", help="JSONL with {query, relevant_ids[]}"),  # noqa: B008
    queries: Path | None = typer.Option(None, "--queries", help="Plain query list, one per line"),  # noqa: B008
    query_image: Path | None = typer.Option(  # noqa: B008
        None,
        "--query-image",
        help="Image collections only: run a one-shot similarity smoke and print top-k",
    ),
    concurrency: int = typer.Option(1, "--concurrency", min=1, max=64),
    judge_llm: str | None = typer.Option(
        None, "--judge-llm", help="<provider>:<model> — enables ragas metrics"
    ),
    compare: Path | None = typer.Option(  # noqa: B008
        None, "--compare", help="variants.yaml for comparison mode"
    ),
    allow_large: bool = typer.Option(False, "--allow-large", help="Override the 6-variant cap"),
) -> None:
    """Phase 5 — score retrieval/latency/RAG quality against the live collection."""
    out = resolve_run_dir(run_dir)
    if query_image is not None and qrels is not None:
        _fail(
            InvalidProfileError(
                pointer="cli",
                reason="--query-image is a smoke tool and cannot be combined with --qrels",
            )
        )
    if query_image is not None:
        try:
            rows = phase_evaluate.run_query_image_smoke(
                out_dir=out, query_image_path=str(query_image)
            )
        except LaunchpadError as e:
            _fail(e)
        typer.echo(f"run-dir: {out}")
        typer.echo(f"query-image: {query_image}")
        if not rows:
            typer.echo("(no hits)")
            return
        for rank, row in enumerate(rows, start=1):
            typer.echo(f"  {rank:>2}. score={row['score']:.4f}  {row['id']}")
        return
    try:
        report = phase_evaluate.run_evaluate(
            out_dir=out,
            qrels_path=str(qrels) if qrels else None,
            queries_path=str(queries) if queries else None,
            concurrency=concurrency,
            judge_llm=judge_llm,
            compare_path=str(compare) if compare else None,
            allow_large=allow_large,
        )
    except LaunchpadError as e:
        _fail(e)
    typer.echo(f"run-dir: {out}")
    typer.echo(f"queries: {report['query_count']} (derived={report['derived']})")
    latency = report["latency_metrics"]
    if latency.get("count"):
        typer.echo(
            f"latency: p50={latency['p50_ms']:.1f}ms "
            f"p95={latency['p95_ms']:.1f}ms "
            f"p99={latency['p99_ms']:.1f}ms"
        )
    retrieval = report["retrieval_metrics"]
    if retrieval:
        typer.echo(
            f"retrieval: recall@10={retrieval['recall@10']:.3f} "
            f"MRR@10={retrieval['MRR@10']:.3f} "
            f"NDCG@10={retrieval['NDCG@10']:.3f}"
        )
    if report["rag_metrics"]:
        typer.echo(f"rag: {json.dumps(report['rag_metrics'], sort_keys=True)}")
    if report["variants"]:
        typer.echo(f"variants scored: {len(report['variants'])}")
    typer.echo(f"report: {out / 'eval_report.md'}")


@app.command()
def deploy(
    run_dir: str | None = typer.Option(None, "--run-dir"),
    cluster_id: str | None = typer.Option(
        None, "--cluster-id", help="Target an existing RUNNING cluster"
    ),
    create: bool = typer.Option(
        False, "--create", help="Provision a new cluster via `zilliz cluster create`"
    ),
    confirm: bool = typer.Option(
        False, "--confirm", help="Required with --create to acknowledge billing impact"
    ),
    stop_local: bool = typer.Option(
        False, "--stop-local", help="Stop the Phase 4 local UI sidecar after deploy"
    ),
) -> None:
    """Phase 6 — promote the Execute run to Zilliz Cloud."""
    out = resolve_run_dir(run_dir)
    try:
        report = phase_deploy.run_deploy(
            out_dir=out,
            cluster_id=cluster_id,
            create=create,
            confirm=confirm,
            stop_local=stop_local,
        )
    except LaunchpadError as e:
        _fail(e)
    typer.echo(f"run-dir: {out}")
    typer.echo(f"cluster: {report['cluster_id']} ({report['cluster_uri']})")
    typer.echo(f"collection: {report['collection_name']}")
    typer.echo(
        f"ingest: {report['ingest_mode']} "
        f"({report['ingest_row_count']} rows, status={report['ingest_status']})"
    )
    obs = report.get("observability") or {}
    if obs.get("grafana_dashboard"):
        typer.echo(f"grafana: {obs['grafana_dashboard']}")
    if obs.get("prometheus_url"):
        typer.echo(f"prometheus: {obs['prometheus_url']}")
    typer.echo(f"deploy.json: {out / 'deploy.json'}")


if __name__ == "__main__":
    app()
