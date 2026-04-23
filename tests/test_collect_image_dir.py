"""Phase 1 Collect — image directory branch."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from lib.errors import InvalidProfileError
from lib.phases.collect import run_collect
from PIL import Image

FIXTURE_PHOTOS = Path(__file__).parent / "fixtures" / "photos"


def test_image_dir_produces_correct_collect_json(tmp_path: Path):
    result = run_collect(input_path=str(FIXTURE_PHOTOS), sample=None, out_dir=tmp_path)
    assert result["data_shape"] == "image_dir"
    assert result["record_count_estimate"] == 20
    assert result["suggested_primary_key"] == "image_path"
    assert result["suggested_text_field"] is None
    assert result["thumbnails_included"] is True

    field_names = {f["name"] for f in result["fields"]}
    assert {"image_path", "width", "height", "bytes"} <= field_names

    first = result["rows"][0]
    assert first["image_path"].endswith(".jpg")
    assert first["width"] > 0 and first["height"] > 0
    assert first["bytes"] > 0
    assert "thumbnail_b64" in first  # auto-on for ≤5000 images
    assert isinstance(first["thumbnail_b64"], str) and len(first["thumbnail_b64"]) > 100


def test_no_supported_files_errors_with_invalid_profile(tmp_path: Path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    (empty_dir / "notes.txt").write_text("not an image")

    with pytest.raises(InvalidProfileError) as exc:
        run_collect(input_path=str(empty_dir), sample=None, out_dir=tmp_path)
    assert "no supported image files" in exc.value.message
    assert ".jpg" in exc.value.message


def test_undecodable_file_warns_and_omits_row(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    # Copy two real images and add one corrupt "image" file.
    real = sorted(FIXTURE_PHOTOS.glob("*.jpg"))[:2]
    for p in real:
        shutil.copy(p, photo_dir / p.name)
    (photo_dir / "broken.jpg").write_text("definitely not a JPEG")

    result = run_collect(input_path=str(photo_dir), sample=None, out_dir=tmp_path)
    assert result["record_count_estimate"] == 2
    paths = {r["image_path"] for r in result["rows"]}
    assert all("broken.jpg" not in p for p in paths)
    err = capsys.readouterr().err
    assert "broken.jpg" in err


def test_exif_less_image_omits_taken_at(tmp_path: Path):
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    img = Image.new("RGB", (32, 24), color=(255, 128, 0))
    img.save(photo_dir / "synthetic.jpg", format="JPEG")

    result = run_collect(input_path=str(photo_dir), sample=None, out_dir=tmp_path)
    assert result["record_count_estimate"] == 1
    row = result["rows"][0]
    assert "taken_at" not in row


def test_thumbnail_cap_disables_thumbs_above_threshold(tmp_path: Path):
    """With cap=1, the 20-image fixture should auto-disable thumbnails."""
    result = run_collect(
        input_path=str(FIXTURE_PHOTOS),
        sample=None,
        out_dir=tmp_path,
        thumbnail_cap_rows=1,
    )
    assert result["thumbnails_included"] is False
    assert "thumbnail_b64" not in result["rows"][0]


def test_explicit_no_thumbnails_overrides_default(tmp_path: Path):
    result = run_collect(
        input_path=str(FIXTURE_PHOTOS),
        sample=None,
        out_dir=tmp_path,
        with_thumbnails=False,
    )
    assert result["thumbnails_included"] is False
    assert "thumbnail_b64" not in result["rows"][0]
