"""Phase 2: Configure — structured requirement capture.

Writes `configure.json` into the provided run dir. Two modes:
  - `--from-json <file>`  : non-interactive (agent-provided answers)
  - default (no tty)      : reads a minimal set of env-var-style inputs
                            or writes a template for the agent to fill
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "use_case": "rag",
    "query_patterns": ["long-natural-language"],
    "dataset_size": 20,
    "deployment_target": "local-standalone",
    "latency_target_ms": None,
    "embedding_preference": None,
    "hybrid_preference": "auto",
    "reranker_preference": "auto",
}


def run_configure(
    *,
    from_json: str | None,
    out_dir: Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if from_json:
        with Path(from_json).open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = dict(DEFAULTS)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                data[k] = v

    out = out_dir / "configure.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    return data
