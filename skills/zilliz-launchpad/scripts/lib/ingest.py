"""chunk → embed → batched insert, with deterministic PKs and retry."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from pymilvus import MilvusClient

from .chunking import ChunkConfig, chunk_text
from .embeddings import EmbeddingProvider


def _deterministic_id(source_id: str, chunk_idx: int) -> str:
    h = hashlib.sha256(f"{source_id}::{chunk_idx}".encode()).hexdigest()
    return h[:32]


@dataclass(frozen=True)
class IngestStats:
    documents: int = 0
    chunks: int = 0
    batches: int = 0
    retries: int = 0


def _batched(iterable: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _is_retryable(err: Exception) -> bool:
    msg = str(err).lower()
    return any(s in msg for s in ("timeout", "503", "502", "unavailable", "reset"))


def _insert_with_retry(
    client: MilvusClient,
    collection: str,
    rows: list[dict[str, Any]],
    *,
    max_attempts: int = 5,
    base_delay: float = 0.5,
) -> int:
    """Returns the number of retries actually performed."""
    retries = 0
    for attempt in range(1, max_attempts + 1):
        try:
            client.upsert(collection_name=collection, data=rows)
            return retries
        except Exception as err:
            if attempt == max_attempts or not _is_retryable(err):
                raise
            retries += 1
            time.sleep(base_delay * (2 ** (attempt - 1)))
    return retries  # unreachable, for type-checker


def ingest_documents(
    client: MilvusClient,
    collection: str,
    documents: Iterable[dict[str, Any]],
    embedder: EmbeddingProvider,
    *,
    text_field: str = "text",
    id_field: str = "id",
    vector_field: str = "embedding",
    chunk_config: ChunkConfig | None = None,
    batch_size: int = 64,
    extra_field_keys: tuple[str, ...] = (),
) -> IngestStats:
    """Chunk → embed → upsert documents. Upsert makes reruns idempotent
    because PKs are deterministic (SHA256 of source id + chunk index).
    """
    chunk_config = chunk_config or ChunkConfig()

    def gen_rows() -> Iterator[dict[str, Any]]:
        nonlocal stats_chunks, stats_docs
        for doc in documents:
            stats_docs += 1
            source_id = str(doc.get(id_field) or f"_synth_{stats_docs}")
            text = str(doc.get(text_field) or "").strip()
            if not text:
                continue
            for idx, chunk in enumerate(chunk_text(text, chunk_config)):
                row: dict[str, Any] = {
                    id_field: _deterministic_id(source_id, idx),
                    text_field: chunk,
                }
                for k in extra_field_keys:
                    if k in doc:
                        row[k] = doc[k]
                pending_texts.append(chunk)
                pending_rows.append(row)
                stats_chunks += 1
                if len(pending_rows) >= batch_size:
                    yield from _flush()

        yield from _flush()

    def _flush() -> Iterator[dict[str, Any]]:
        if not pending_rows:
            return
        vectors = embedder.embed(pending_texts)
        for row, vec in zip(pending_rows, vectors, strict=True):
            row[vector_field] = vec
            yield row
        pending_rows.clear()
        pending_texts.clear()

    pending_rows: list[dict[str, Any]] = []
    pending_texts: list[str] = []
    stats_docs = 0
    stats_chunks = 0
    stats_batches = 0
    stats_retries = 0

    for batch in _batched(gen_rows(), batch_size):
        stats_retries += _insert_with_retry(client, collection, batch)
        stats_batches += 1

    return IngestStats(
        documents=stats_docs,
        chunks=stats_chunks,
        batches=stats_batches,
        retries=stats_retries,
    )
