"""Shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure `lib.*` imports resolve when running pytest from the repo root
_SCRIPTS = Path(__file__).resolve().parent.parent / "skills" / "zilliz-launchpad" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
