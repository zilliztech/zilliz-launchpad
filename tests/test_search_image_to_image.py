"""Unit tests for `search_image_to_image`.

Mocks the Milvus client and both image encoders so the test stays fast and
doesn't require open-clip-torch weights or a Voyage API key.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from lib.errors import ImageDecodeError, UnsupportedImageProviderError
from lib.search import search_image_to_image

JPEG_SOI = b"\xff\xd8\xff\xe0"  # arbitrary bytes — encoder is mocked


def _image_plan(provider: str = "clip-local", dim: int = 512) -> dict:
    return {
        "embedding": {
            "provider": provider,
            "model": "ViT-B-32" if provider == "clip-local" else "voyage-multimodal-3",
            "dim": dim,
            "modality": "image",
            "device_hint": "cpu",
        },
    }


def _fake_milvus_hit(pk: str, score: float):
    hit = MagicMock()
    hit.id = pk
    hit.score = score
    hit.entity = {"image_path": pk}
    return hit


def test_clip_local_jpeg_routes_through_embed_image_batch():
    client = MagicMock()
    client.search.return_value = [[_fake_milvus_hit("/photos/a.jpg", 0.91)]]
    with patch("lib.search.embed_image_batch", return_value=[[0.1] * 512]) as mock_embed:
        hits = search_image_to_image(client, "img_demo", JPEG_SOI, _image_plan(), top_k=5)
    assert len(hits) == 1
    assert hits[0].id == "/photos/a.jpg"
    assert hits[0].score == pytest.approx(0.91)
    mock_embed.assert_called_once()
    # Confirm the encoder was handed a tempfile path, not raw bytes
    ((paths,), kwargs) = mock_embed.call_args
    assert kwargs["model_id"] == "ViT-B-32"
    assert kwargs["device_hint"] == "cpu"
    assert len(list(paths)) == 1


def test_voyage_multimodal_routes_through_voyage_path():
    client = MagicMock()
    client.search.return_value = [[_fake_milvus_hit("/photos/b.png", 0.5)]]
    plan = _image_plan(provider="voyage", dim=1024)
    with patch("lib.search.embed_image_batch_voyage", return_value=[[0.2] * 1024]) as mock_voyage:
        hits = search_image_to_image(client, "img_demo", JPEG_SOI, plan)
    assert len(hits) == 1
    assert hits[0].id == "/photos/b.png"
    mock_voyage.assert_called_once()


def test_undecodable_bytes_raise_image_decode_error():
    client = MagicMock()
    with (
        patch(
            "lib.search.embed_image_batch",
            side_effect=OSError("cannot identify image file"),
        ),
        pytest.raises(ImageDecodeError) as excinfo,
    ):
        search_image_to_image(client, "img_demo", b"not an image", _image_plan())
    payload = excinfo.value.to_dict()
    assert payload["code"] == "image_decode_failed"
    assert "cannot identify" in payload["reason"]
    # Search must not be called when encoding fails
    client.search.assert_not_called()


def test_text_only_provider_raises_typed_error():
    client = MagicMock()
    text_plan = {
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dim": 1536,
            "modality": "text",
        },
    }
    with pytest.raises(UnsupportedImageProviderError) as excinfo:
        search_image_to_image(client, "img_demo", JPEG_SOI, text_plan)
    assert excinfo.value.to_dict()["provider"] == "openai"
    client.search.assert_not_called()


def test_empty_encoder_output_raises_image_decode_error():
    client = MagicMock()
    with (
        patch("lib.search.embed_image_batch", return_value=[]),
        pytest.raises(ImageDecodeError),
    ):
        search_image_to_image(client, "img_demo", JPEG_SOI, _image_plan())
    client.search.assert_not_called()


def test_search_respects_top_k_and_output_fields():
    client = MagicMock()
    client.search.return_value = [
        [
            _fake_milvus_hit("/photos/a.jpg", 0.9),
            _fake_milvus_hit("/photos/b.jpg", 0.8),
            _fake_milvus_hit("/photos/c.jpg", 0.7),
        ]
    ]
    with patch("lib.search.embed_image_batch", return_value=[[0.1] * 512]):
        hits = search_image_to_image(
            client,
            "img_demo",
            JPEG_SOI,
            _image_plan(),
            top_k=2,
            output_fields=["image_path"],
        )
    assert len(hits) == 2
    args, kwargs = client.search.call_args
    assert kwargs["limit"] == 2
    assert kwargs["output_fields"] == ["image_path"]
    assert kwargs["anns_field"] == "embedding"
