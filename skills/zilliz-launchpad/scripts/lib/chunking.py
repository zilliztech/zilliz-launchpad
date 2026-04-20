"""Recursive character splitter with configurable size / overlap.

Token counts are approximated by character / 4 (fast, good enough for
splitting; exact tokenization happens provider-side during embedding).
"""

from __future__ import annotations

from dataclasses import dataclass

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class ChunkConfig:
    size: int = 512            # approx tokens
    overlap: int = 64          # approx tokens
    chars_per_token: int = 4   # approximation

    @property
    def size_chars(self) -> int:
        return self.size * self.chars_per_token

    @property
    def overlap_chars(self) -> int:
        return self.overlap * self.chars_per_token


def _recursive_split(text: str, max_chars: int, seps: list[str]) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text else []
    sep = seps[0]
    rest = seps[1:] if len(seps) > 1 else [""]
    if sep == "":
        # hard cut
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    parts = text.split(sep)
    if len(parts) == 1:
        return _recursive_split(text, max_chars, rest)

    chunks: list[str] = []
    buf = ""
    for part in parts:
        candidate = part if not buf else buf + sep + part
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
            buf = ""
        if len(part) > max_chars:
            chunks.extend(_recursive_split(part, max_chars, rest))
        else:
            buf = part
    if buf:
        chunks.append(buf)
    return chunks


def chunk_text(text: str, config: ChunkConfig | None = None) -> list[str]:
    """Split `text` into chunks roughly sized to `config.size` tokens.

    Adds overlap between adjacent chunks (except the last).
    """
    config = config or ChunkConfig()
    if not text.strip():
        return []

    base = _recursive_split(text, config.size_chars, SEPARATORS)
    if config.overlap_chars <= 0 or len(base) <= 1:
        return base

    out: list[str] = []
    for i, chunk in enumerate(base):
        if i == 0:
            out.append(chunk)
            continue
        prev_tail = base[i - 1][-config.overlap_chars :]
        out.append(prev_tail + chunk)
    return out
