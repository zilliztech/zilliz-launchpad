"""Resolve a `--input` argument to a sorted list of files.

Supports three forms:
- a single file path (returned as-is if its suffix is supported)
- a directory (walked recursively, filtered to supported suffixes)
- a shell glob containing `*`, `?`, or `[`

Used by both `collect` and `execute` so the two phases agree on what
`--input ./docs/` means.
"""

from __future__ import annotations

import glob as _glob
from pathlib import Path

from .errors import EmptyInputSetError

_GLOB_CHARS = ("*", "?", "[")


def has_glob_chars(raw: str) -> bool:
    return any(ch in raw for ch in _GLOB_CHARS)


def resolve_inputs(raw: str, *, supported_suffixes: frozenset[str] | set[str]) -> list[Path]:
    """Resolve `raw` to a sorted list of absolute file paths.

    Raises `EmptyInputSetError` if the resolved set is empty.
    Raises `FileNotFoundError` with a glob-quoting hint if `raw` is a literal
    non-existent path with no glob metacharacters.
    """
    supported = frozenset(s.lower() for s in supported_suffixes)

    if has_glob_chars(raw):
        candidates = [Path(p) for p in _glob.glob(raw, recursive=True)]
        files = sorted(
            p.resolve() for p in candidates if p.is_file() and p.suffix.lower() in supported
        )
        if not files:
            raise EmptyInputSetError(
                raw=raw,
                reason=(f"glob matched no files with supported suffixes ({sorted(supported)})"),
            )
        return files

    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(
            f"Input not found: {path}. If you meant a glob, quote it (e.g. --input 'docs/*.pdf')."
        )

    if path.is_dir():
        files = sorted(
            p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() in supported
        )
        if not files:
            raise EmptyInputSetError(
                raw=raw,
                reason=(
                    f"directory '{path}' contains no files with supported suffixes "
                    f"({sorted(supported)})"
                ),
            )
        return files

    # Single file: do NOT validate suffix here. The caller's dispatcher
    # produces a more informative ValueError for unsupported suffixes.
    return [path.resolve()]
