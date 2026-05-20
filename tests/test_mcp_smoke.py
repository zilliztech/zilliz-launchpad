"""Smoke test for the launchpad_mcp wrapper.

Exercises the three deterministic phases (collect → configure → plan) over
the in-process FastMCP client/server harness. Asserts that the per-tool
response carries `run_dir` + `artifact`, that the artifact matches the
JSON written to disk, and that the LaunchpadError → structured envelope
mapping works.

Skipped (not failed) when the optional `mcp` extra is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# Target a deep SDK module rather than the bare `mcp` package: the repo's
# `mcp/` docs directory is picked up as a namespace package, so the bare
# import succeeds even without the SDK installed and the skip never fires.
pytest.importorskip(
    "mcp.server.fastmcp",
    reason="MCP SDK not installed — `uv sync --extra mcp`",
)

import lib.run_dir as _run_dir_mod  # noqa: E402  (after importorskip)

from launchpad_mcp import server as launchpad_server  # noqa: E402
from launchpad_mcp import tools as launchpad_tools  # noqa: E402


@pytest.fixture
def isolated_runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect new_run_dir/resolve_run_dir to tmp_path so phases don't touch
    the developer's real runs/ directory.
    """
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(_run_dir_mod, "_RUNS_ROOT", root)
    return root


def _parse_tool_result(result: Any) -> dict[str, Any]:
    """Extract the structured payload from an MCP CallToolResult."""
    if getattr(result, "structuredContent", None):
        payload: dict[str, Any] = result.structuredContent
        # FastMCP wraps non-model returns under a "result" key.
        return payload.get("result", payload)  # type: ignore[no-any-return]
    text: str = result.content[0].text
    return json.loads(text)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Direct wrapper tests (no MCP transport) — covers task 4.2 and 5.x.


def test_collect_returns_envelope_on_missing_input(isolated_runs_root: Path) -> None:
    """LaunchpadError surfaces with structured code/message."""
    with pytest.raises(launchpad_tools.LaunchpadError) as exc_info:
        launchpad_tools.run_collect()
    envelope = exc_info.value.to_dict()
    assert envelope["code"] == "missing_input"
    assert "sample" in envelope["message"] or "input_path" in envelope["message"]


def test_three_phase_pipeline_direct(isolated_runs_root: Path) -> None:
    """collect → configure → plan via the wrapper functions directly."""
    collect_result = launchpad_tools.run_collect(sample="movies")
    run_dir = Path(collect_result["run_dir"])
    assert run_dir.is_dir()
    assert (run_dir / "collect.json").exists()
    assert collect_result["artifact"] == json.loads(
        (run_dir / "collect.json").read_text(encoding="utf-8")
    )

    configure_result = launchpad_tools.run_configure(
        run_dir=str(run_dir),
        dataset_size=20,
        deployment_target="local-standalone",
    )
    assert configure_result["run_dir"] == str(run_dir)
    assert (run_dir / "configure.json").exists()
    assert configure_result["artifact"] == json.loads(
        (run_dir / "configure.json").read_text(encoding="utf-8")
    )

    plan_result = launchpad_tools.run_plan(run_dir=str(run_dir))
    assert plan_result["run_dir"] == str(run_dir)
    assert (run_dir / "plan.json").exists()
    assert plan_result["artifact"] == json.loads(
        (run_dir / "plan.json").read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# End-to-end MCP transport test.


async def test_three_phase_pipeline_over_mcp(isolated_runs_root: Path) -> None:
    """Drive collect → configure → plan through the FastMCP harness."""
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(
        launchpad_server.mcp._mcp_server
    ) as client:
        tools_list = await client.list_tools()
        names = [t.name for t in tools_list.tools]
        assert names == ["collect", "configure", "plan", "execute", "evaluate", "deploy"]

        collect_resp = await client.call_tool("collect", {"sample": "movies"})
        assert not collect_resp.isError, collect_resp
        collect_payload = _parse_tool_result(collect_resp)
        run_dir = Path(collect_payload["run_dir"])
        assert (run_dir / "collect.json").exists()
        assert collect_payload["artifact"] == json.loads(
            (run_dir / "collect.json").read_text(encoding="utf-8")
        )

        configure_resp = await client.call_tool(
            "configure",
            {
                "run_dir": str(run_dir),
                "dataset_size": 20,
                "deployment_target": "local-standalone",
            },
        )
        assert not configure_resp.isError, configure_resp
        configure_payload = _parse_tool_result(configure_resp)
        assert configure_payload["run_dir"] == str(run_dir)
        assert configure_payload["artifact"] == json.loads(
            (run_dir / "configure.json").read_text(encoding="utf-8")
        )

        plan_resp = await client.call_tool("plan", {"run_dir": str(run_dir)})
        assert not plan_resp.isError, plan_resp
        plan_payload = _parse_tool_result(plan_resp)
        assert plan_payload["run_dir"] == str(run_dir)
        assert plan_payload["artifact"] == json.loads(
            (run_dir / "plan.json").read_text(encoding="utf-8")
        )


async def test_error_envelope_over_mcp(isolated_runs_root: Path) -> None:
    """LaunchpadError → structured JSON envelope in the tool error content."""
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(
        launchpad_server.mcp._mcp_server
    ) as client:
        # collect with neither sample nor input_path → MissingInput envelope.
        resp = await client.call_tool("collect", {})
        assert resp.isError
        text = resp.content[0].text  # type: ignore[union-attr]
        # FastMCP prefixes the error string; the JSON envelope is appended.
        # Extract the JSON suffix and assert its shape.
        start = text.find("{")
        envelope = json.loads(text[start:])
        assert envelope["code"] == "missing_input"
        assert "message" in envelope
