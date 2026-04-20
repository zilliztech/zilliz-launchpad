"""Phase implementations (Collect, Configure, Plan, Execute, Deploy scaffold)."""

from . import deploy  # re-exported so `from lib.phases import deploy` works cleanly

__all__ = ["deploy"]
