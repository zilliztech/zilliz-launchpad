"""Shared CLI helpers for phase subcommands.

Leaf module: imports only ``typer`` and ``lib.errors``. It must never import
a phase module or ``zilliz_ops`` so phases can depend on it without creating
an import cycle (zilliz_ops -> phase -> lib.cli, never back to zilliz_ops).
"""

from __future__ import annotations

import sys

import typer

from .errors import CliErrorEnvelope, LaunchpadError


def fail(err: LaunchpadError) -> None:
    """Print the error envelope to stderr and exit with code 1."""
    env = CliErrorEnvelope.from_error(err)
    print(env.to_json(), file=sys.stderr)
    raise typer.Exit(code=1)
