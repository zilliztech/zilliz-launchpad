"""Phase 1: Collect — analyze a sample file and produce collect.json."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .. import samples

SUPPORTED_SUFFIXES = {".jsonl", ".ndjson", ".csv", ".txt"}


def _infer_field_type(values: list[Any]) -> str:
    types = Counter()
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            types["bool"] += 1
        elif isinstance(v, int):
            types["int"] += 1
        elif isinstance(v, float):
            types["float"] += 1
        else:
            types["string"] += 1
    if not types:
        return "string"
    return types.most_common(1)[0][0]


def _analyze_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "data_shape": "jsonl",
            "fields": [],
            "suggested_primary_key": "id",
            "suggested_text_field": "text",
            "record_count_estimate": 0,
        }
    keys = list(records[0].keys())
    fields = []
    for k in keys:
        values = [r.get(k) for r in records[:50]]
        t = _infer_field_type(values)
        avg_len = None
        if t == "string":
            lens = [len(str(v)) for v in values if v is not None]
            avg_len = int(sum(lens) / len(lens)) if lens else 0
        fields.append({"name": k, "type": t, "avg_length": avg_len, "sample_value": values[0]})

    pk_candidates = [
        f["name"] for f in fields if f["name"].lower() in ("id", "_id", "doc_id", "uid")
    ]
    pk = pk_candidates[0] if pk_candidates else fields[0]["name"]

    text_candidates = sorted(
        (f for f in fields if f["type"] == "string" and f["avg_length"]),
        key=lambda f: f["avg_length"] or 0,
        reverse=True,
    )
    text_field = text_candidates[0]["name"] if text_candidates else fields[0]["name"]

    return {
        "data_shape": "jsonl",
        "fields": fields,
        "suggested_primary_key": pk,
        "suggested_text_field": text_field,
        "record_count_estimate": len(records),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
            if len(out) >= 500:
                break
    return out


def _read_csv(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(dict(row))
            if len(out) >= 500:
                break
    return out


def run_collect(
    *,
    input_path: str | None,
    sample: str | None,
    out_dir: Path,
) -> dict[str, Any]:
    if sample is not None:
        records = list(samples.load(sample))
        result = _analyze_records(records)
        result["source_path"] = None
        result["source_sample"] = sample
    else:
        assert input_path, "Either --sample or --input is required"
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")
        suffix = path.suffix.lower()
        if suffix in (".jsonl", ".ndjson"):
            records = _read_jsonl(path)
            result = _analyze_records(records)
        elif suffix == ".csv":
            records = _read_csv(path)
            result = _analyze_records(records)
            result["data_shape"] = "csv"
        elif suffix == ".txt":
            text = path.read_text(encoding="utf-8")
            result = {
                "data_shape": "text",
                "fields": [
                    {
                        "name": "text",
                        "type": "string",
                        "avg_length": len(text),
                        "sample_value": text[:200],
                    }
                ],
                "suggested_primary_key": "id",
                "suggested_text_field": "text",
                "record_count_estimate": 1,
            }
        else:
            raise ValueError(f"Unsupported input suffix: {suffix}. Use {SUPPORTED_SUFFIXES}")
        result["source_path"] = str(path)

    out = out_dir / "collect.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)
    return result
