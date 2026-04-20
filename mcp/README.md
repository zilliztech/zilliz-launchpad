# MCP Path (Stub)

This directory is a placeholder for a future MCP (Model Context Protocol) server
that will expose the same phase commands exposed by `zilliz_ops.py` as MCP tools.

The MVP ships only the Agent Skill path (see `skills/zilliz-launchpad/`).
The CLI is designed so a thin MCP wrapper can be added here without touching
`scripts/lib/` — each phase subcommand maps 1:1 to an MCP tool.

Not implemented in this change.
