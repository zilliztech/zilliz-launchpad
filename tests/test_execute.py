"""End-to-end test against Milvus Standalone.

Marked `e2e` — excluded from the default CI lane. Run locally with:

    ./skills/zilliz-launchpad/scripts/start_milvus.sh up
    uv run pytest -m e2e tests/test_execute.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None,
    reason="OPENAI_API_KEY is required for real embedding calls",
)
def test_end_to_end_movies(tmp_path: Path):
    from lib.phases.collect import run_collect
    from lib.phases.configure import run_configure
    from lib.phases.execute import run_execute
    from lib.phases.plan import run_plan

    run_dir = tmp_path
    run_collect(input_path=None, sample="movies", out_dir=run_dir)
    run_configure(
        from_json=None,
        out_dir=run_dir,
        overrides={"dataset_size": 20, "deployment_target": "local-standalone"},
    )
    run_plan(out_dir=run_dir)
    report = run_execute(
        out_dir=run_dir,
        sample="movies",
        input_path=None,
        ui_port=8001,
        start_ui=False,
    )
    assert report["ingest"]["documents"] == 20
    assert report["smoke_hits"], "smoke query returned no hits"
    plan = json.loads((run_dir / "plan.json").read_text())
    assert plan["index"]["type"] == "HNSW"


@pytest.mark.cloud
def test_cloud_placeholder():
    """Cloud tests run only with ZILLIZ_TOKEN set; keep this out of default CI."""
    pytest.skip("Cloud tests opt-in only")


@pytest.mark.e2e
@pytest.mark.skipif(
    os.environ.get("LAUNCHPAD_BULK_IMPORT_E2E") != "1",
    reason="Set LAUNCHPAD_BULK_IMPORT_E2E=1 to exercise the CLI-backed bulk-import path",
)
def test_bulk_import_path_cli_present(tmp_path: Path):
    """CLI-present bulk-import branch — requires a logged-in `zilliz` CLI
    and a reachable cluster. Opt-in only."""
    from lib import zilliz_cli
    from lib.client import Backend
    from lib.phases.execute import _should_bulk_import

    assert zilliz_cli.is_available(), "zilliz CLI must be installed and logged in for this test"
    assert (
        _should_bulk_import(
            row_count=200_000,
            threshold=100_000,
            target_backend=Backend.ZILLIZ_CLOUD,
            cluster_id="c-test",
        )
        is True
    )
