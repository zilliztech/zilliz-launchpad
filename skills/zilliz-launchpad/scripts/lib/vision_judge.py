"""Vision-capable LLM helpers for Phase 5 image derived eval.

Wraps a small allow-list of `<provider>:<model>` strings the launchpad
trusts for image captioning. Keeps `evaluate.py` thin and lets us reuse
the same allow-list at the CLI argparse layer.
"""

from __future__ import annotations

import base64
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .credentials import resolve_required
from .errors import JudgeUnavailableError

logger = logging.getLogger(__name__)

# Each entry is a glob-style match against the model id. We keep this list
# tight on purpose — adding a model is cheap, but accepting a non-vision
# model silently produces unusable derived queries.
_VISION_ALLOW_LIST: dict[str, list[re.Pattern[str]]] = {
    "openai": [re.compile(p) for p in (r"^gpt-4o.*$", r"^gpt-5.*$")],
    "anthropic": [re.compile(p) for p in (r"^claude-.+$",)],
}

_CAPTION_PROMPT = (
    "Write one short search query (5-10 words) that someone would type "
    "to find this image. No quotes, no preamble, just the query."
)


def is_vision_capable(provider: str, model: str) -> bool:
    patterns = _VISION_ALLOW_LIST.get(provider.lower())
    if not patterns:
        return False
    return any(p.match(model) for p in patterns)


def require_vision(provider: str, model: str) -> None:
    """Raise `JudgeUnavailableError` if (provider, model) isn't on the allow-list."""
    if not is_vision_capable(provider, model):
        env_var = "OPENAI_API_KEY" if provider.lower() == "openai" else "ANTHROPIC_API_KEY"
        raise JudgeUnavailableError(provider=f"{provider}:{model}", env_var=env_var)


def caption_image(provider: str, model: str, image_path: Path | str) -> str:
    """Return a short caption for `image_path` via the named vision LLM.

    Routes to the provider-specific HTTP client; both providers accept
    base64 image data inline. Caller is responsible for caching.
    """
    require_vision(provider, model)
    p = Path(image_path)
    suffix = p.suffix.lower().lstrip(".")
    media_type = "jpeg" if suffix in ("jpg", "jpeg") else suffix or "jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")

    if provider.lower() == "openai":
        return _caption_openai(model, b64, media_type)
    if provider.lower() == "anthropic":
        return _caption_anthropic(model, b64, media_type)
    # Allow-list already gated this; defensive fall-through.
    raise JudgeUnavailableError(provider=f"{provider}:{model}", env_var="UNKNOWN")


def caption_images(
    provider: str, model: str, paths: Iterable[Path | str]
) -> list[str]:
    """Sequential captioning. Vision APIs are slow + rate-limited; parallelism
    is left to the caller (Phase 5 typically only captions ~25 derived images).
    """
    return [caption_image(provider, model, p) for p in paths]


# --- Provider-specific clients --------------------------------------------


def _caption_openai(model: str, b64: str, media_type: str) -> str:
    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI(api_key=resolve_required("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        max_tokens=64,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _CAPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{media_type};base64,{b64}"},
                    },
                ],
            }
        ],
    )
    msg = resp.choices[0].message.content if resp.choices else None
    return _clean(msg or "")


def _caption_anthropic(model: str, b64: str, media_type: str) -> str:
    try:
        from anthropic import Anthropic  # noqa: PLC0415
    except ImportError as exc:
        raise JudgeUnavailableError(
            provider=f"anthropic:{model}", env_var="ANTHROPIC_API_KEY"
        ) from exc

    client = Anthropic(api_key=resolve_required("ANTHROPIC_API_KEY"))
    msg: Any = client.messages.create(
        model=model,
        max_tokens=64,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{media_type}",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": _CAPTION_PROMPT},
                ],
            }
        ],
    )
    text_parts: list[str] = []
    for block in msg.content or []:
        if getattr(block, "type", None) == "text":
            text_parts.append(getattr(block, "text", ""))
    return _clean(" ".join(text_parts))


def _clean(s: str) -> str:
    return s.strip().strip('"').strip("'").strip()
