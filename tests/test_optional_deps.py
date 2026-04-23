"""Test that image code paths fail with a structured envelope when the
[multimodal] extra is not installed."""

from __future__ import annotations

import builtins
import sys

import pytest
from lib.errors import MissingDependencyError
from lib.optional_deps import MULTIMODAL_INSTALL_HINT, detect_device_hint, require_multimodal


def test_missing_dependency_payload_shape():
    err = MissingDependencyError(feature="image-search", install_hint="uv pip install foo")
    payload = err.to_dict()
    assert payload["code"] == "missing_dependency"
    assert payload["feature"] == "image-search"
    assert payload["install_hint"] == "uv pip install foo"


def _is_multimodal_module(name: str) -> bool:
    return (
        name == "torch"
        or name.startswith("torch.")
        or name == "open_clip"
        or name.startswith("open_clip.")
    )


def _hide_modules(prefixes: tuple[str, ...]) -> dict[str, object]:
    """Pop matching modules from sys.modules and return them so we can restore.

    Removing torch from sys.modules would corrupt torch's internal state on
    re-import (it re-runs module init, which fails because docstrings are
    already set). The caller MUST restore the dict after the test.
    """

    def matches(name: str) -> bool:
        return any(name == p or name.startswith(p + ".") for p in prefixes)

    saved = {k: sys.modules[k] for k in list(sys.modules) if matches(k)}
    for k in saved:
        del sys.modules[k]
    return saved


def test_require_multimodal_raises_envelope_when_missing(monkeypatch: pytest.MonkeyPatch):
    """Simulate torch / open_clip not being installed."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if _is_multimodal_module(name):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    saved = _hide_modules(("torch", "open_clip"))
    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        with pytest.raises(MissingDependencyError) as exc:
            require_multimodal()
        assert exc.value.payload["feature"] == "image-search"
        assert exc.value.payload["install_hint"] == MULTIMODAL_INSTALL_HINT
    finally:
        sys.modules.update(saved)


def test_detect_device_hint_returns_cpu_without_torch(monkeypatch: pytest.MonkeyPatch):
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    saved = _hide_modules(("torch",))
    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        assert detect_device_hint() == "cpu"
    finally:
        sys.modules.update(saved)
