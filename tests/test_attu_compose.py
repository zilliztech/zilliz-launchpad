"""Static assertions on the Attu service entry in docker-compose.yml.

These tests only parse YAML — they do not invoke Docker, pull images, or
start containers, so they are safe to run in any CI environment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

COMPOSE_PATH = (
    Path(__file__).resolve().parent.parent
    / "skills"
    / "zilliz-launchpad"
    / "scripts"
    / "docker-compose.yml"
)


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def test_attu_service_exists(compose: dict) -> None:
    assert "attu" in compose["services"], "attu service missing from docker-compose.yml"


def test_attu_image_tag_matches_milvus_minor(compose: dict) -> None:
    milvus_image = compose["services"]["standalone"]["image"]
    attu_image = compose["services"]["attu"]["image"]
    assert milvus_image.startswith("milvusdb/milvus:v2.6"), (
        "Milvus tag changed; update this test and the Attu tag deliberately"
    )
    assert attu_image.startswith("zilliz/attu:v2.6"), (
        f"Attu tag {attu_image!r} must track Milvus minor (v2.6.x)"
    )


def test_attu_gated_by_ops_profile(compose: dict) -> None:
    profiles = compose["services"]["attu"].get("profiles")
    assert profiles == ["ops"], (
        f"attu must be gated by profile 'ops' so default compose up skips it; got {profiles!r}"
    )


def test_attu_port_bound_to_loopback(compose: dict) -> None:
    ports = compose["services"]["attu"]["ports"]
    assert ports == ["127.0.0.1:8000:3000"], (
        f"attu port must be bound to 127.0.0.1:8000:3000 (loopback only); got {ports!r}"
    )


def test_attu_depends_on_standalone(compose: dict) -> None:
    deps = compose["services"]["attu"].get("depends_on") or []
    assert "standalone" in deps, f"attu must depend_on standalone; got {deps!r}"


def test_attu_env_vars_support_remote_override(compose: dict) -> None:
    env = compose["services"]["attu"]["environment"]
    assert "MILVUS_URL" in env
    assert "${ATTU_MILVUS_URL:-standalone:19530}" in env["MILVUS_URL"], (
        "ATTU_MILVUS_URL must override MILVUS_URL with local default"
    )
    assert "ATTU_MILVUS_TOKEN" in env
