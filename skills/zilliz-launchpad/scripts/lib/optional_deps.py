"""Lazy loaders for optional extras.

Image / multimodal code paths import these helpers instead of importing
torch / open_clip directly, so a user without the [multimodal] extra gets a
structured `missing_dependency` envelope on stderr rather than a raw
ModuleNotFoundError traceback.
"""

from __future__ import annotations

from types import ModuleType

from .errors import MissingDependencyError

MULTIMODAL_INSTALL_HINT = "uv pip install -e '.[multimodal]'"


def require_multimodal() -> tuple[ModuleType, ModuleType]:
    """Return (torch, open_clip) modules or raise MissingDependencyError."""
    try:
        import open_clip  # noqa: PLC0415
        import torch  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            feature="image-search",
            install_hint=MULTIMODAL_INSTALL_HINT,
        ) from exc
    return torch, open_clip


def detect_device_hint() -> str:
    """Best-available device for CLIP inference. Falls back to cpu without torch."""
    try:
        import torch  # noqa: PLC0415
    except ModuleNotFoundError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
