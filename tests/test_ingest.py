from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from lib.chunking import ChunkConfig
from lib.ingest import _deterministic_id, _is_retryable, ingest_documents


@dataclass
class FakeEmbedder:
    dim: int = 4
    name: str = "fake"
    calls: int = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(i % 10)] * self.dim for i in range(len(texts))]


def test_deterministic_id_stable():
    assert _deterministic_id("doc-1", 0) == _deterministic_id("doc-1", 0)
    assert _deterministic_id("doc-1", 0) != _deterministic_id("doc-1", 1)


def test_is_retryable_heuristics():
    assert _is_retryable(Exception("timeout"))
    assert _is_retryable(Exception("503 service unavailable"))
    assert not _is_retryable(Exception("invalid schema"))


def test_ingest_idempotent_primary_keys(mocker):
    client = mocker.Mock()
    client.upsert.return_value = None
    embedder = FakeEmbedder()

    docs = [{"id": "d1", "text": "alpha beta gamma delta"}]
    stats1 = ingest_documents(
        client, "c", docs, embedder, chunk_config=ChunkConfig(size=4096, overlap=0)
    )
    stats2 = ingest_documents(
        client, "c", docs, embedder, chunk_config=ChunkConfig(size=4096, overlap=0)
    )
    assert stats1.chunks == stats2.chunks

    # Same PKs on both invocations
    all_rows: list[list[dict[str, Any]]] = [call.kwargs["data"] for call in client.upsert.call_args_list]
    ids_first = [r["id"] for r in all_rows[0]]
    ids_second = [r["id"] for r in all_rows[1]]
    assert ids_first == ids_second


def test_ingest_retries_transient_errors(mocker):
    client = mocker.Mock()

    call_count = {"n": 0}

    def flaky_upsert(collection_name: str, data: list[dict[str, Any]]) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("503 service unavailable")

    client.upsert.side_effect = flaky_upsert
    embedder = FakeEmbedder()

    stats = ingest_documents(
        client,
        "c",
        [{"id": "d1", "text": "hello"}],
        embedder,
        chunk_config=ChunkConfig(size=4096, overlap=0),
    )
    assert stats.retries == 1


def test_ingest_fails_fast_on_schema_error(mocker):
    client = mocker.Mock()
    client.upsert.side_effect = Exception("invalid schema")
    embedder = FakeEmbedder()

    with pytest.raises(Exception, match="invalid schema"):
        ingest_documents(
            client,
            "c",
            [{"id": "d1", "text": "hello"}],
            embedder,
            chunk_config=ChunkConfig(size=4096, overlap=0),
        )
