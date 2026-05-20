"""Phase 1: Collect — analyze a sample file and produce collect.json."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import typer
from PIL import UnidentifiedImageError

from .. import samples
from ..cli import fail as _cli_fail
from ..errors import (
    InputSchemaConflictError,
    InvalidProfileError,
    LaunchpadError,
    MissingDependencyError,
)
from ..image_io import (
    IMAGE_SUFFIXES,
    THUMBNAIL_DEFAULT_CAP_ROWS,
    list_images,
    read_image_metadata,
)
from ..inputs import has_glob_chars, resolve_inputs
from ..run_dir import new_run_dir, resolve_run_dir

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


def _analyze_one_file(
    path: Path,
    *,
    split_markdown_headings: bool,
) -> dict[str, Any]:
    """Per-file analysis for the document branches (jsonl/csv/txt/md/pdf).

    Returns the same dict shape the existing per-file branches produced before
    multi-file support was added.
    """
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        records = _read_jsonl(path)
        return _analyze_records(records)
    if suffix == ".csv":
        records = _read_csv(path)
        result = _analyze_records(records)
        result["data_shape"] = "csv"
        return result
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8")
        return {
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
    if suffix == ".md":
        return _read_markdown(path, split_headings=split_markdown_headings)
    if suffix == ".pdf":
        return _read_pdf(path)
    raise ValueError(
        f"Unsupported input suffix: {suffix}. "
        f"Use {sorted(SUPPORTED_SUFFIXES)} "
        f"or pass a directory of images ({sorted(IMAGE_SUFFIXES)}), "
        f"a directory of documents, or a glob like 'docs/*.pdf'"
    )


def _merge_field(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    *,
    field_name: str,
    existing_files: list[str],
    incoming_file: str,
) -> dict[str, Any]:
    """Merge two per-file field entries; raise InputSchemaConflictError on type clash."""
    if existing["type"] != incoming["type"]:
        raise InputSchemaConflictError(
            field_name=field_name,
            files_and_types=[
                # `existing_files` may carry several already-merged sources, but
                # the *type* is the same across all of them — report the first
                # disagreeing file alongside the incoming one.
                {"path": existing_files[0], "type": existing["type"]},
                {"path": incoming_file, "type": incoming["type"]},
            ],
        )
    merged = dict(existing)
    # Average length: simple mean of the two reported averages when both present.
    e_len = existing.get("avg_length")
    i_len = incoming.get("avg_length")
    if isinstance(e_len, int) and isinstance(i_len, int):
        merged["avg_length"] = (e_len + i_len) // 2
    elif isinstance(i_len, int):
        merged["avg_length"] = i_len
    # Sample value: keep the longer string sample if both are strings.
    e_sv = existing.get("sample_value")
    i_sv = incoming.get("sample_value")
    if isinstance(e_sv, str) and isinstance(i_sv, str):
        merged["sample_value"] = e_sv if len(e_sv) >= len(i_sv) else i_sv
    elif e_sv is None and i_sv is not None:
        merged["sample_value"] = i_sv
    return merged


def _merge_collect_results(
    per_file: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    """Merge per-file analyses into a single multi-file collect result.

    Computes the union schema, raising `InputSchemaConflictError` on a
    cross-file type disagreement for a shared field name.
    """
    merged_fields: dict[str, dict[str, Any]] = {}
    field_seen_in: dict[str, list[str]] = {}
    source_files: list[dict[str, Any]] = []
    total_records = 0
    shapes: set[str] = set()
    warnings: list[str] = []

    for path, result in per_file:
        path_str = str(path)
        shapes.add(str(result.get("data_shape") or "unknown"))
        total_records += int(result.get("record_count_estimate") or 0)
        source_files.append(
            {
                "path": path_str,
                "data_shape": result.get("data_shape"),
                "record_count_estimate": result.get("record_count_estimate"),
            }
        )
        for warning in result.get("warnings") or []:
            warnings.append(warning)
        for field in result.get("fields") or []:
            name = str(field["name"])
            if name in merged_fields:
                merged_fields[name] = _merge_field(
                    merged_fields[name],
                    field,
                    field_name=name,
                    existing_files=field_seen_in[name],
                    incoming_file=path_str,
                )
                field_seen_in[name].append(path_str)
            else:
                merged_fields[name] = dict(field)
                field_seen_in[name] = [path_str]

    file_count = len(per_file)
    for name in merged_fields:
        if len(field_seen_in[name]) < file_count:
            merged_fields[name]["nullable"] = True

    fields = list(merged_fields.values())
    pk_candidates = [
        f["name"] for f in fields if str(f["name"]).lower() in ("id", "_id", "doc_id", "uid")
    ]
    pk = pk_candidates[0] if pk_candidates else (fields[0]["name"] if fields else "id")
    text_candidates = sorted(
        (f for f in fields if f["type"] == "string" and f.get("avg_length")),
        key=lambda f: f.get("avg_length") or 0,
        reverse=True,
    )
    text_field = text_candidates[0]["name"] if text_candidates else "text"

    data_shape = next(iter(shapes)) if len(shapes) == 1 else "mixed"
    out: dict[str, Any] = {
        "data_shape": data_shape,
        "fields": fields,
        "suggested_primary_key": pk,
        "suggested_text_field": text_field,
        "record_count_estimate": total_records,
        "source_files": source_files,
    }
    if warnings:
        out["warnings"] = warnings
    return out


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
        # Image/video directories keep their bespoke single-shape semantics and
        # do NOT route through resolve_inputs.
        raw = input_path
        literal_path = Path(raw)
        if not has_glob_chars(raw) and literal_path.is_dir():
            if _dir_is_video_collection(literal_path):
                knobs = _read_configure_video_knobs(out_dir)
                result = _read_video_dir(
                    literal_path,
                    out_dir=out_dir,
                    interval_s=knobs["frame_interval_seconds"],
                    max_frames_per_video=knobs["max_frames_per_video"],
                    sampling_strategy=knobs["sampling_strategy"],
                    scene_threshold=knobs["scene_threshold"],
                )
                result["source_path"] = str(literal_path)
            elif _dir_is_image_collection(literal_path):
                result = _read_image_dir(
                    literal_path,
                    with_thumbnails=with_thumbnails,
                    thumbnail_cap_rows=thumbnail_cap_rows,
                )
                result["source_path"] = str(literal_path)
            else:
                # Document directory: walk + merge.
                files = resolve_inputs(raw, supported_suffixes=SUPPORTED_SUFFIXES)
                result = _run_multi_or_single(
                    files,
                    split_markdown_headings=split_markdown_headings,
                )
        elif has_glob_chars(raw):
            files = resolve_inputs(raw, supported_suffixes=SUPPORTED_SUFFIXES)
            result = _run_multi_or_single(
                files,
                split_markdown_headings=split_markdown_headings,
            )
        else:
            if not literal_path.exists():
                raise FileNotFoundError(f"Input not found: {literal_path}")
            # Single existing file — preserve existing single-file semantics
            # (including the ValueError on unsupported suffix).
            result = _analyze_one_file(
                literal_path,
                split_markdown_headings=split_markdown_headings,
            )
            result["source_path"] = str(literal_path)

    out = out_dir / "collect.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)
    return result


def _run_multi_or_single(
    files: list[Path],
    *,
    split_markdown_headings: bool,
) -> dict[str, Any]:
    """Dispatch a resolved file list to single-file or merged multi-file analysis."""
    if len(files) == 1:
        result = _analyze_one_file(files[0], split_markdown_headings=split_markdown_headings)
        result["source_path"] = str(files[0])
        return result
    per_file: list[tuple[Path, dict[str, Any]]] = [
        (
            f,
            _analyze_one_file(f, split_markdown_headings=split_markdown_headings),
        )
        for f in files
    ]
    return _merge_collect_results(per_file)


def _dir_is_image_collection(path: Path) -> bool:
    return any(p.suffix.lower() in IMAGE_SUFFIXES for p in path.iterdir() if p.is_file())


def register(app: typer.Typer) -> None:
    """Attach the Phase 1 ``collect`` subcommand to the shared app."""

    @app.command()
    def collect(
        sample: str | None = typer.Option(None, "--sample", "-s", help="Bundled sample name"),
        input: Path | None = typer.Option(  # noqa: B008
            None,
            "--input",
            "-i",
            help="Path to user data: a file, directory (recursive), or glob like 'docs/*.pdf'",
        ),
        run_dir: str | None = typer.Option(
            None, "--run-dir", help="Existing run dir; default = new"
        ),
        with_thumbnails: bool | None = typer.Option(
            None,
            "--with-thumbnails/--no-thumbnails",
            help="Image dir only. Default: on for ≤5000 images, off above.",
        ),
        thumbnail_cap_rows: int = typer.Option(
            5000,
            "--thumbnail-cap-rows",
            help="Image dir only. Auto-disable thumbnails above this many images.",
        ),
        split_markdown_headings: bool = typer.Option(
            False,
            "--split-markdown-headings/--no-split-markdown-headings",
            help="Markdown only. Emit one record per `## ` section instead of one per file.",
        ),
    ) -> None:
        """Phase 1 — analyze sample data."""
        if sample is None and input is None:
            print(
                json.dumps({"code": "missing_input", "message": "Pass --sample or --input"}),
                file=sys.stderr,
            )
            raise typer.Exit(code=2)
        out = resolve_run_dir(run_dir) if run_dir else new_run_dir(label="collect")
        try:
            result = run_collect(
                input_path=str(input) if input else None,
                sample=sample,
                out_dir=out,
                with_thumbnails=with_thumbnails,
                thumbnail_cap_rows=thumbnail_cap_rows,
                split_markdown_headings=split_markdown_headings,
            )
        except LaunchpadError as e:
            _cli_fail(e)
        typer.echo(f"run-dir: {out}")
        shape = result.get("data_shape")
        if shape == "image_dir":
            typer.echo("data_shape: image_dir")
            typer.echo(f"images: {result['record_count_estimate']}")
            typer.echo(f"thumbnails_included: {result['thumbnails_included']}")
        elif shape in ("markdown", "pdf"):
            typer.echo(f"data_shape: {shape}")
            typer.echo(f"records: {result['record_count_estimate']}")
            typer.echo(f"suggested_primary_key: {result['suggested_primary_key']}")
            typer.echo(f"suggested_text_field: {result['suggested_text_field']}")
            for warning in result.get("warnings", []):
                typer.echo(f"warning: {warning}", err=True)
        else:
            typer.echo(f"suggested_primary_key: {result['suggested_primary_key']}")
            typer.echo(f"suggested_text_field: {result['suggested_text_field']}")
