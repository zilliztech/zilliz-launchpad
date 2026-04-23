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


def test_require_multimodal_raises_envelope_when_missing(monkeypatch: pytest.MonkeyPatch):
    """Simulate torch / open_clip not being installed."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if _is_multimodal_module(name):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    for mod in list(sys.modules):
        if _is_multimodal_module(mod):
            del sys.modules[mod]
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(MissingDependencyError) as exc:
        require_multimodal()
    assert exc.value.payload["feature"] == "image-search"
    assert exc.value.payload["install_hint"] == MULTIMODAL_INSTALL_HINT


def test_detect_device_hint_returns_cpu_without_torch(monkeypatch: pytest.MonkeyPatch):
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    for mod in list(sys.modules):
        if mod == "torch" or mod.startswith("torch."):
            del sys.modules[mod]
    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert detect_device_hint() == "cpu"
