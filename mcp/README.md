# MCP Path

zilliz-launchpad ships an MCP (Model Context Protocol) server that wraps the
six CLI phases (`collect`, `configure`, `plan`, `execute`, `evaluate`, `deploy`)
as MCP tools. Use this path when your host isn't agent-skill-aware (Cursor,
Claude Desktop, generic MCP clients).

The server lives in the [`launchpad_mcp`](../launchpad_mcp/) Python package.
The directory name (`mcp/`) is reserved for this documentation entry point —
the package itself is named `launchpad_mcp` so it doesn't shadow the official
`mcp` SDK from PyPI.

## Install

```bash
uv sync --extra mcp
```

This adds the [`mcp`](https://pypi.org/project/mcp/) SDK. The Skill install
path does not need this extra.

## Launch

```bash
uv run python -m launchpad_mcp.server     # stdio transport
```

A console script `zilliz-launchpad-mcp` is also declared in `pyproject.toml`,
available once the project is installed via a packaging tool that honors
`[project.scripts]` (e.g. `pip install -e .[mcp]`).

## Tools

| Tool        | Phase | Returns                                                    |
|-------------|-------|------------------------------------------------------------|
| `collect`   | 1     | `{run_dir, artifact}` — `artifact` = parsed `collect.json` |
| `configure` | 2     | `{run_dir, artifact}` — `artifact` = parsed `configure.json`|
| `plan`      | 3     | `{run_dir, artifact}` — `artifact` = parsed `plan.json`    |
| `execute`   | 4     | `{run_dir, artifact}` — `artifact` = execute report dict   |
| `evaluate`  | 5     | `{run_dir, artifact}` — `artifact` = parsed `eval_report.json` |
| `deploy`    | 6     | `{run_dir, artifact}` — `artifact` = deploy state dict     |

Each tool's input schema mirrors the corresponding `python scripts/zilliz_ops.py
<phase>` flag set (kebab-case CLI flag → snake_case JSON field). `collect`
creates a new `run_dir` (or appends to an existing one if you pass `run_dir`);
phases 2–6 require the `run_dir` returned by an earlier `collect` call.

Run `zilliz-launchpad-mcp` and call `list_tools` from any MCP client to see
the full parameter schemas.

## Error envelope

On failure a tool returns an MCP `CallToolResult` with `isError: true`. The
text content is the standard CLI error envelope JSON, identical to what the
CLI prints to stderr:

```json
{"code": "missing_credential", "message": "Required credential not set: OPENAI_API_KEY", "env_var": "OPENAI_API_KEY", "export_hint": "export OPENAI_API_KEY=<value>"}
```

Unexpected exceptions (anything that isn't a `LaunchpadError` subclass)
surface with `code: "internal_error"` so hosts never see an opaque Python
traceback.

The FastMCP error content prefixes the envelope with `Error executing tool
<name>: ` — host integrations should locate the first `{` and `json.loads`
from there.

## Worked example

End-to-end `collect → configure → plan` over MCP (pseudocode for any MCP
client):

```python
collect = await client.call_tool("collect", {"sample": "movies"})
run_dir = collect.structuredContent["run_dir"]

await client.call_tool("configure", {
    "run_dir": run_dir,
    "dataset_size": 20,
    "deployment_target": "local-standalone",
})

plan = await client.call_tool("plan", {"run_dir": run_dir})
print(plan.structuredContent["artifact"]["index"]["type"])  # → HNSW
```

The corresponding smoke test lives at
[`tests/test_mcp_smoke.py`](../tests/test_mcp_smoke.py) and runs the same
sequence through the in-process FastMCP client harness.

## Host registration

### Cursor / Claude Desktop

Add an entry to your host's MCP config (`~/.cursor/mcp.json` for Cursor,
`~/Library/Application Support/Claude/claude_desktop_config.json` for
Claude Desktop on macOS):

```json
{
  "mcpServers": {
    "zilliz-launchpad": {
      "command": "uv",
      "args": ["run", "--project", "/abs/path/to/zilliz-launchpad",
               "python", "-m", "launchpad_mcp.server"]
    }
  }
}
```

Replace `/abs/path/to/zilliz-launchpad` with the absolute path of your
checkout. Restart the host to pick up the server.

## Boundary

- The server does not modify anything under
  `skills/zilliz-launchpad/scripts/lib/`; it imports the same `run_<phase>`
  helpers the CLI calls.
- Tool input schemas are maintained by hand and must stay in sync with the
  CLI flags in `lib/phases/<phase>.py`. The smoke test catches drift for the
  three deterministic phases; the others are exercised by their existing
  unit/e2e tests.
- Transport is stdio only. SSE / streamable-HTTP are out of scope for v1.
