"""Bundled sample datasets.

`list_datasets()` — returns known dataset names
`load(name)` — yields dict records
`path_to(name)` — file path for tools that want raw access
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent / "sample_data"

_DATASETS: dict[str, dict[str, str]] = {
    "movies": {
        "file": "movies.jsonl",
        "description": "20 fictional movies with title/body/year/genre",
        "text_field": "body",
        "id_field": "id",
    },
    "beir-scifact-mini": {
        "file": "beir_scifact_mini.jsonl",
        "description": "15 scientific-claim docs with paired queries + qrels",
        "text_field": "body",
        "id_field": "id",
    },
}


def list_datasets() -> list[str]:
    return list(_DATASETS.keys())


def describe(name: str) -> dict[str, str]:
    if name not in _DATASETS:
        raise KeyError(f"Unknown dataset: {name}. Available: {list_datasets()}")
    return dict(_DATASETS[name])


def path_to(name: str) -> Path:
    meta = describe(name)
    return _ROOT / meta["file"]


def load(name: str) -> Iterator[dict[str, object]]:
    path = path_to(name)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_qrels(name: str = "beir-scifact-mini") -> list[tuple[str, str, int]]:
    """Return list of (query_id, doc_id, relevance)."""
    base = _ROOT / "beir_scifact_qrels.tsv"
    if name != "beir-scifact-mini" or not base.exists():
        return []
    out: list[tuple[str, str, int]] = []
    with base.open("r", encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            out.append((parts[0], parts[1], int(parts[2])))
    return out


def load_queries(name: str = "beir-scifact-mini") -> list[dict[str, str]]:
    if name != "beir-scifact-mini":
        return []
    p = _ROOT / "beir_scifact_queries.jsonl"
    if not p.exists():
        return []
    out: list[dict[str, str]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
