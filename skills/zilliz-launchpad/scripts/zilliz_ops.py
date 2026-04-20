"""zilliz-launchpad CLI.

Phase-to-subcommand mapping (1:1):
  collect    → Phase 1
  configure  → Phase 2
  plan       → Phase 3
  execute    → Phase 4

Usage (typical):
  python scripts/zilliz_ops.py collect --sample movies
  python scripts/zilliz_ops.py configure --from-json configure.json
  python scripts/zilliz_ops.py plan
  python scripts/zilliz_ops.py execute --sample movies
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from lib.errors import CliErrorEnvelope, LaunchpadError
from lib.phases import collect as phase_collect
from lib.phases import configure as phase_configure
from lib.phases import execute as phase_execute
from lib.phases import plan as phase_plan
from lib.run_dir import latest_run_dir, new_run_dir, resolve_run_dir

app = typer.Typer(help="zilliz-launchpad — MVP Phases 1–4")


def _fail(err: LaunchpadError) -> None:
    env = CliErrorEnvelope.from_error(err)
    print(env.to_json(), file=sys.stderr)
    raise typer.Exit(code=1)


@app.command()
def collect(
    sample: Optional[str] = typer.Option(None, "--sample", "-s", help="Bundled sample name"),
    input: Optional[Path] = typer.Option(None, "--input", "-i", help="Path to user data"),
    run_dir: Optional[str] = typer.Option(None, "--run-dir", help="Existing run dir; default = new"),
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
            input_path=str(input) if input else None, sample=sample, out_dir=out
        )
    except LaunchpadError as e:
        _fail(e)
    typer.echo(f"run-dir: {out}")
    typer.echo(f"suggested_primary_key: {result['suggested_primary_key']}")
    typer.echo(f"suggested_text_field: {result['suggested_text_field']}")


@app.command()
def configure(
    from_json: Optional[Path] = typer.Option(None, "--from-json", help="Pre-filled answers"),
    run_dir: Optional[str] = typer.Option(None, "--run-dir"),
    use_case: Optional[str] = typer.Option(None, "--use-case"),
    dataset_size: Optional[int] = typer.Option(None, "--dataset-size"),
    deployment_target: Optional[str] = typer.Option(None, "--deployment"),
) -> None:
    """Phase 2 — capture requirements."""
    out = resolve_run_dir(run_dir) if run_dir else (latest_run_dir() or new_run_dir(label="configure"))
    overrides = {
        "use_case": use_case,
        "dataset_size": dataset_size,
        "deployment_target": deployment_target,
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
    run_dir: Optional[str] = typer.Option(None, "--run-dir"),
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
    run_dir: Optional[str] = typer.Option(None, "--run-dir"),
    sample: Optional[str] = typer.Option(None, "--sample", "-s"),
    input: Optional[Path] = typer.Option(None, "--input", "-i"),
    ui_port: int = typer.Option(8000, "--ui-port"),
    no_ui: bool = typer.Option(False, "--no-ui", help="Skip starting the sidecar"),
) -> None:
    """Phase 4 — apply plan and start UI sidecar."""
    out = resolve_run_dir(run_dir)
    try:
        report = phase_execute.run_execute(
            out_dir=out,
            sample=sample,
            input_path=str(input) if input else None,
            ui_port=ui_port,
            start_ui=not no_ui,
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
    else:
        typer.echo("✗ Smoke query returned zero hits")
        raise typer.Exit(code=3)
    if report.get("sidecar_pid"):
        typer.echo(f"UI sidecar pid {report['sidecar_pid']} on port {report['ui_port']}")
        typer.echo("Start the Next.js UI: (cd scripts/ui && npm run dev)")


if __name__ == "__main__":
    app()
