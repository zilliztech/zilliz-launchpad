from __future__ import annotations

from lib.chunking import ChunkConfig, chunk_text


def test_empty_text_produces_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n   ") == []


def test_short_text_is_a_single_chunk():
    out = chunk_text("Hello world.", ChunkConfig(size=512, overlap=0))
    assert out == ["Hello world."]


def test_splits_long_text_at_separators():
    paragraph = "Sentence one. " * 300  # ~4500 chars
    out = chunk_text(paragraph, ChunkConfig(size=128, overlap=0))
    assert len(out) > 2
    # Every chunk must fit the target (+ some slack because we only cut on separators)
    for c in out:
        assert len(c) <= 128 * 4 * 2  # generous upper bound


def test_overlap_carries_previous_tail():
    text = ("A" * 400) + (" " * 1) + ("B" * 400)
    out = chunk_text(text, ChunkConfig(size=100, overlap=20))
    assert len(out) >= 2
    # Chunk i>0 must start with characters from the end of chunk i-1
    for i in range(1, len(out)):
        prev_tail = out[i - 1][-20 * 4 :]
        assert out[i].startswith(prev_tail[: min(len(prev_tail), 20)])
