"""Phase 6: Deploy — placeholder.

The Deploy phase is out of scope for the `integrate-zilliz-cli` change.
This module exists so that follow-up changes can wire up a real
implementation without re-plumbing call sites. It MUST raise until
the dedicated Deploy change lands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_deploy(*, out_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError(
        "Phase 6 Deploy lands in a later change. Scaffold present so call "
        "sites compile. See openspec/changes/integrate-zilliz-cli/proposal.md."
    )
