"""Phase 1: Collect — analyze a sample file and produce collect.json."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import UnidentifiedImageError

from .. import samples
from ..errors import InvalidProfileError, MissingDependencyError
from ..image_io import (
    IMAGE_SUFFIXES,
    THUMBNAIL_DEFAULT_CAP_ROWS,
    list_images,
    read_image_metadata,
)

SUPPORTED_SUFFIXES = {".jsonl", ".ndjson", ".csv", ".txt", ".md", ".pdf"}
SUPPORTED_DIR_SUFFIXES = IMAGE_SUFFIXES
VIDEO_SUFFIXES: frozenset[str] = frozenset({".mp4", ".mov", ".mkv", ".webm"})

DEFAULT_VIDEO_FRAME_INTERVAL_S = 2.0
DEFAULT_VIDEO_MAX_FRAMES = 600
DEFAULT_VIDEO_SCENE_THRESHOLD = 0.3
DEFAULT_VIDEO_SAMPLING_STRATEGY = "every_n_seconds"

PDF_SCANNED_TEXT_THRESHOLD = 50


def _infer_field_type(values: list[Any]) -> str:
    types: Counter[str] = Counter()
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
    return str(types.most_common(1)[0][0])


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
        f["name"] for f in fields if str(f["name"]).lower() in ("id", "_id", "doc_id", "uid")
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


def _read_image_dir(
    path: Path,
    *,
    with_thumbnails: bool | None,
    thumbnail_cap_rows: int,
) -> dict[str, Any]:
    images = list_images(path)
    if not images:
        suffixes = ", ".join(sorted(IMAGE_SUFFIXES))
        raise InvalidProfileError(
            "input",
            f"directory '{path}' contains no supported image files (suffixes: {suffixes})",
        )

    if with_thumbnails is None:
        with_thumbnails = len(images) <= thumbnail_cap_rows

    rows: list[dict[str, Any]] = []
    for img_path in images:
        try:
            rows.append(read_image_metadata(img_path, with_thumbnail=with_thumbnails))
        except (UnidentifiedImageError, OSError) as exc:
            print(
                f"warn: could not read {img_path.name}: {exc}",
                file=sys.stderr,
            )

    if not rows:
        raise InvalidProfileError(
            "input",
            f"directory '{path}' had {len(images)} candidate image(s) but none could be decoded",
        )

    fields = [
        {"name": "image_path", "type": "string", "sample_value": rows[0]["image_path"]},
        {"name": "width", "type": "int", "sample_value": rows[0]["width"]},
        {"name": "height", "type": "int", "sample_value": rows[0]["height"]},
        {"name": "bytes", "type": "int", "sample_value": rows[0]["bytes"]},
    ]
    if any("taken_at" in r for r in rows):
        fields.append({"name": "taken_at", "type": "string", "sample_value": None})

    return {
        "data_shape": "image_dir",
        "fields": fields,
        "suggested_primary_key": "image_path",
        "suggested_text_field": None,
        "record_count_estimate": len(rows),
        "rows": rows,
        "thumbnails_included": with_thumbnails,
    }


def _read_video_dir(
    path: Path,
    *,
    out_dir: Path,
    interval_s: float,
    max_frames_per_video: int,
    sampling_strategy: str,
    scene_threshold: float,
) -> dict[str, Any]:
    from ..video import (  # noqa: PLC0415 — optional extra, gated by video_dir detection
        VideoProbeError,
        list_videos,
        sample_frames,
    )

    videos = list_videos(path)
    if not videos:
        suffixes = ", ".join(sorted(VIDEO_SUFFIXES))
        raise InvalidProfileError(
            "input",
            f"directory '{path}' contains no supported video files (suffixes: {suffixes})",
        )

    frames_dir = out_dir / "frames"
    rows: list[dict[str, Any]] = []
    for video_path in videos:
        try:
            video_rows = sample_frames(
                video_path,
                strategy=sampling_strategy,
                interval_s=interval_s,
                max_frames=max_frames_per_video,
                scene_threshold=scene_threshold,
                out_dir=frames_dir,
            )
        except VideoProbeError as exc:
            print(f"warn: could not probe {video_path.name}: {exc.message}", file=sys.stderr)
            continue
        except Exception as exc:  # noqa: BLE001 — unexpected decode/sampler failure
            print(f"warn: could not sample {video_path.name}: {exc}", file=sys.stderr)
            continue

        if len(video_rows) >= max_frames_per_video:
            print(
                f"warn: {video_path.name} hit max_frames_per_video={max_frames_per_video}; "
                "increase the cap to sample more densely",
                file=sys.stderr,
            )
        rows.extend(r.to_dict() for r in video_rows)

    if not rows:
        raise InvalidProfileError(
            "input",
            f"directory '{path}' had {len(videos)} candidate video(s) but none could be sampled",
        )

    fields = [
        {"name": "video_path", "type": "string", "sample_value": rows[0]["video_path"]},
        {"name": "t_seconds", "type": "float", "sample_value": rows[0]["t_seconds"]},
        {"name": "frame_path", "type": "string", "sample_value": rows[0]["frame_path"]},
        {"name": "source_index", "type": "int", "sample_value": rows[0]["source_index"]},
    ]
    return {
        "data_shape": "video_dir",
        "fields": fields,
        "suggested_primary_key": "frame_path",
        "suggested_text_field": None,
        "record_count_estimate": len(rows),
        "rows": rows,
        "thumbnails_included": True,
        "video_count": len(videos),
        "video_sampling": {
            "strategy": sampling_strategy,
            "frame_interval_seconds": interval_s,
            "max_frames_per_video": max_frames_per_video,
            "scene_threshold": scene_threshold if sampling_strategy == "scene_change" else None,
        },
    }


def _read_configure_video_knobs(out_dir: Path) -> dict[str, Any]:
    """Read video sampling knobs from configure.json if present, else defaults."""
    configure_path = out_dir / "configure.json"
    if not configure_path.exists():
        return {
            "frame_interval_seconds": DEFAULT_VIDEO_FRAME_INTERVAL_S,
            "max_frames_per_video": DEFAULT_VIDEO_MAX_FRAMES,
            "sampling_strategy": DEFAULT_VIDEO_SAMPLING_STRATEGY,
            "scene_threshold": DEFAULT_VIDEO_SCENE_THRESHOLD,
        }
    try:
        data = json.loads(configure_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        data = {}
    return {
        "frame_interval_seconds": float(
            data.get("frame_interval_seconds") or DEFAULT_VIDEO_FRAME_INTERVAL_S
        ),
        "max_frames_per_video": int(data.get("max_frames_per_video") or DEFAULT_VIDEO_MAX_FRAMES),
        "sampling_strategy": str(data.get("sampling_strategy") or DEFAULT_VIDEO_SAMPLING_STRATEGY),
        "scene_threshold": float(data.get("scene_threshold") or DEFAULT_VIDEO_SCENE_THRESHOLD),
    }


def _dir_is_video_collection(path: Path) -> bool:
    return any(p.suffix.lower() in VIDEO_SUFFIXES for p in path.iterdir() if p.is_file())


def _strip_yaml_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end == -1:
        return text
    return text[end + len("\n---\n") :]


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_heading is not None:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_heading, body))
            current_heading = line[3:].strip()
            current_lines = []
        elif current_heading is not None:
            current_lines.append(line)
    if current_heading is not None:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_heading, body))
    return sections


def _read_markdown(path: Path, *, split_headings: bool) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    text = _strip_yaml_frontmatter(raw)
    stem = path.stem

    sections = _split_markdown_sections(text) if split_headings else []

    if sections:
        records: list[dict[str, Any]] = [
            {"id": f"{stem}-s{i}", "text": body, "section_heading": heading}
            for i, (heading, body) in enumerate(sections, start=1)
        ]
        text_lens = [len(r["text"]) for r in records]
        head_lens = [len(r["section_heading"]) for r in records]
        fields: list[dict[str, Any]] = [
            {
                "name": "id",
                "type": "string",
                "avg_length": int(sum(len(r["id"]) for r in records) / len(records)),
                "sample_value": records[0]["id"],
            },
            {
                "name": "text",
                "type": "string",
                "avg_length": int(sum(text_lens) / len(text_lens)),
                "sample_value": records[0]["text"][:200],
            },
            {
                "name": "section_heading",
                "type": "string",
                "avg_length": int(sum(head_lens) / len(head_lens)),
                "sample_value": records[0]["section_heading"],
            },
        ]
    else:
        body = text.strip()
        records = [{"id": f"{stem}-1", "text": body}]
        fields = [
            {
                "name": "id",
                "type": "string",
                "avg_length": len(records[0]["id"]),
                "sample_value": records[0]["id"],
            },
            {
                "name": "text",
                "type": "string",
                "avg_length": len(body),
                "sample_value": body[:200],
            },
        ]

    return {
        "data_shape": "markdown",
        "fields": fields,
        "suggested_primary_key": "id",
        "suggested_text_field": "text",
        "record_count_estimate": len(records),
    }


def _read_pdf(path: Path) -> dict[str, Any]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise MissingDependencyError(
            feature="pdf-collect",
            install_hint="pip install zilliz-launchpad[documents]",
        ) from exc

    reader = PdfReader(str(path))
    stem = path.stem
    records: list[dict[str, Any]] = []
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        records.append(
            {
                "id": f"{stem}-p{i}",
                "text": page_text,
                "page_number": i,
                "source_path": str(path),
            }
        )

    text_lens = [len(r["text"]) for r in records]
    fields: list[dict[str, Any]] = [
        {
            "name": "id",
            "type": "string",
            "avg_length": int(sum(len(r["id"]) for r in records) / max(len(records), 1)),
            "sample_value": records[0]["id"] if records else None,
        },
        {
            "name": "text",
            "type": "string",
            "avg_length": int(sum(text_lens) / len(text_lens)) if text_lens else 0,
            "sample_value": records[0]["text"][:200] if records else None,
        },
        {
            "name": "page_number",
            "type": "int",
            "sample_value": records[0]["page_number"] if records else None,
        },
        {
            "name": "source_path",
            "type": "string",
            "avg_length": len(str(path)),
            "sample_value": str(path),
        },
    ]

    result: dict[str, Any] = {
        "data_shape": "pdf",
        "fields": fields,
        "suggested_primary_key": "id",
        "suggested_text_field": "text",
        "record_count_estimate": len(records),
    }
    if sum(text_lens) < PDF_SCANNED_TEXT_THRESHOLD:
        result["warnings"] = [
            f"PDF '{path.name}' appears to contain no extractable text — likely a "
            "scanned image; consider an OCR pre-processing step."
        ]
    return result


def run_collect(
    *,
    input_path: str | None,
    sample: str | None,
    out_dir: Path,
    with_thumbnails: bool | None = None,
    thumbnail_cap_rows: int = THUMBNAIL_DEFAULT_CAP_ROWS,
    split_markdown_headings: bool = False,
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
        if path.is_dir():
            if _dir_is_video_collection(path):
                knobs = _read_configure_video_knobs(out_dir)
                result = _read_video_dir(
                    path,
                    out_dir=out_dir,
                    interval_s=knobs["frame_interval_seconds"],
                    max_frames_per_video=knobs["max_frames_per_video"],
                    sampling_strategy=knobs["sampling_strategy"],
                    scene_threshold=knobs["scene_threshold"],
                )
            else:
                result = _read_image_dir(
                    path,
                    with_thumbnails=with_thumbnails,
                    thumbnail_cap_rows=thumbnail_cap_rows,
                )
            result["source_path"] = str(path)
        else:
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
            elif suffix == ".md":
                result = _read_markdown(path, split_headings=split_markdown_headings)
            elif suffix == ".pdf":
                result = _read_pdf(path)
            else:
                raise ValueError(
                    f"Unsupported input suffix: {suffix}. "
                    f"Use {sorted(SUPPORTED_SUFFIXES)} "
                    f"or pass a directory of images ({sorted(IMAGE_SUFFIXES)})"
                )
            result["source_path"] = str(path)

    out = out_dir / "collect.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)
    return result
