"""Image-shaped I/O helpers shared by Phase 1 (Collect) and Phase 4 (Execute).

Pillow lives in the base deps, so these helpers can be imported unconditionally.
The torch / open_clip path stays behind `optional_deps.require_multimodal()`.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

from PIL import ExifTags, Image

IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})

THUMBNAIL_DEFAULT_LONG_SIDE = 256
THUMBNAIL_JPEG_QUALITY = 80
THUMBNAIL_DEFAULT_CAP_ROWS = 5000

_EXIF_DATE_TAG_ID: int | None = next(
    (tag for tag, name in ExifTags.TAGS.items() if name == "DateTimeOriginal"), None
)


def list_images(directory: Path) -> list[Path]:
    """Return supported image files under ``directory``, sorted for determinism."""
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def read_exif_taken_at(img: Image.Image) -> str | None:
    """Best-effort EXIF DateTimeOriginal as ISO-8601, or None on any failure."""
    if _EXIF_DATE_TAG_ID is None:
        return None
    try:
        exif = img.getexif() if hasattr(img, "getexif") else None
        if not exif:
            return None
        raw = exif.get(_EXIF_DATE_TAG_ID)
        if not raw:
            return None
        # EXIF format: "YYYY:MM:DD HH:MM:SS"
        parsed = datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
        return parsed.isoformat()
    except Exception:
        return None


def make_thumbnail_b64(img: Image.Image, *, long_side: int = THUMBNAIL_DEFAULT_LONG_SIDE) -> str:
    """Fit-resize an image to ``long_side`` and return JPEG-encoded base64."""
    thumb = img.copy()
    thumb.thumbnail((long_side, long_side))
    if thumb.mode != "RGB":
        thumb = thumb.convert("RGB")
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=THUMBNAIL_JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def read_image_metadata(path: Path, *, with_thumbnail: bool) -> dict[str, object]:
    """Open an image and produce a Phase-1 row dict.

    Raises Pillow's ``UnidentifiedImageError`` (or OSError) if the file
    cannot be decoded — the caller is responsible for catching and
    counting these as skipped files.
    """
    with Image.open(path) as img:
        img.load()
        row: dict[str, object] = {
            "image_path": str(path),
            "width": img.width,
            "height": img.height,
            "bytes": path.stat().st_size,
        }
        taken_at = read_exif_taken_at(img)
        if taken_at is not None:
            row["taken_at"] = taken_at
        if with_thumbnail:
            row["thumbnail_b64"] = make_thumbnail_b64(img)
    return row
