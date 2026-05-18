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

This module is a thin shell: each phase under ``lib/phases/`` owns its own
subcommand and exposes ``register(app)``. Adding a phase is one new file plus
one entry in ``_PHASES`` below — no other edits here.
"""

from __future__ import annotations

import typer
from lib.phases import collect as phase_collect
from lib.phases import configure as phase_configure
from lib.phases import deploy as phase_deploy
from lib.phases import evaluate as phase_evaluate
from lib.phases import execute as phase_execute
from lib.phases import plan as phase_plan

app = typer.Typer(help="zilliz-launchpad — Phases 1–6")

# Order defines the `--help` listing order (Phase 1 → 6).
_PHASES = (
    phase_collect,
    phase_configure,
    phase_plan,
    phase_execute,
    phase_evaluate,
    phase_deploy,
)

for _phase in _PHASES:
    _phase.register(app)


if __name__ == "__main__":
    app()
