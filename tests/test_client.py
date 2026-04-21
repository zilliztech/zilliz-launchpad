"""URI detection and credential behavior for `lib.client`."""

from __future__ import annotations

import pytest
from lib.client import Backend, detect_target
from lib.errors import MissingCredentialError


def test_local_uri_detected_without_token():
    t = detect_target("http://localhost:19530")
    assert t.backend is Backend.LOCAL
    assert t.requires_token is False


def test_loopback_ip_detected_as_local():
    t = detect_target("http://127.0.0.1:19530")
    assert t.backend is Backend.LOCAL


def test_zilliz_cloud_uri_requires_token():
    t = detect_target("https://in03-xxx.api.gcp-us-west1.cloud.zilliz.com")
    assert t.backend is Backend.ZILLIZ_CLOUD
    assert t.requires_token is True


def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("ZILLIZ_TOKEN", raising=False)
    from lib.client import MilvusClient

    with pytest.raises(MissingCredentialError) as exc:
        MilvusClient(uri="https://any.api.gcp.cloud.zilliz.com")
    assert exc.value.payload["env_var"] == "ZILLIZ_TOKEN"


def test_explicit_token_overrides_env(monkeypatch, mocker):
    monkeypatch.delenv("ZILLIZ_TOKEN", raising=False)
    fake = mocker.patch("lib.client._PyMilvusClient", autospec=False)
    from lib.client import MilvusClient

    MilvusClient(uri="https://any.api.gcp.cloud.zilliz.com", token="explicit")
    fake.assert_called_once()
    _, kwargs = fake.call_args
    assert kwargs["token"] == "explicit"
