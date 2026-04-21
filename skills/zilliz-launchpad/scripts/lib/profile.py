"""Requirement profile loader + validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]

from .errors import InvalidProfileError

_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent.parent / "references" / "requirement-profile.schema.json"
)


def load_schema() -> dict[str, Any]:
    with _SCHEMA_PATH.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def load_profile(run_dir: Path) -> dict[str, Any]:
    collect_path = run_dir / "collect.json"
    configure_path = run_dir / "configure.json"
    if not collect_path.exists():
        raise InvalidProfileError(pointer="/collect", reason="collect.json missing")
    if not configure_path.exists():
        raise InvalidProfileError(pointer="/configure", reason="configure.json missing")

    with collect_path.open("r", encoding="utf-8") as f:
        collect = json.load(f)
    with configure_path.open("r", encoding="utf-8") as f:
        configure = json.load(f)

    profile = {"collect": collect, "configure": configure}
    validate_profile(profile)
    return profile


def validate_profile(profile: dict[str, Any]) -> None:
    schema = load_schema()
    try:
        jsonschema.validate(instance=profile, schema=schema)
    except jsonschema.ValidationError as e:
        pointer = "/" + "/".join(str(p) for p in e.absolute_path)
        raise InvalidProfileError(pointer=pointer or "/", reason=e.message) from e
