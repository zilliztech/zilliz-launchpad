"""Embedding providers — text (API-only) and image (local CLIP + Voyage API).

Strategy pattern keyed on provider name. Text providers are thin HTTP
wrappers. Image providers add `clip-local` (loaded on demand via
`optional_deps.require_multimodal`) and Voyage's multimodal endpoint.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .credentials import resolve_required
from .errors import BackendUnsupportedError, MissingCredentialError
from .optional_deps import detect_device_hint, require_multimodal

_PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "voyage": "VOYAGE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "zilliz-byom": "ZILLIZ_BYOM_KEY",
    "clip-local": None,
}

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class OpenAIEmbedder:
    model: str = "text-embedding-3-small"
    dim: int = 1536
    name: str = "openai"

    def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI

        client = OpenAI(api_key=resolve_required("OPENAI_API_KEY"))
        resp = client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


@dataclass
class VoyageEmbedder:
    model: str = "voyage-3"
    dim: int = 1024
    name: str = "voyage"

    def embed(self, texts: list[str]) -> list[list[float]]:
        import voyageai

        client = voyageai.Client(api_key=resolve_required("VOYAGE_API_KEY"))  # type: ignore[attr-defined]
        result = client.embed(texts, model=self.model, input_type="document")
        return [list(e) for e in result.embeddings]


@dataclass
class CohereEmbedder:
    model: str = "embed-english-v3.0"
    dim: int = 1024
    name: str = "cohere"

    def embed(self, texts: list[str]) -> list[list[float]]:
        import cohere

        client = cohere.Client(api_key=resolve_required("COHERE_API_KEY"))
        resp = client.embed(texts=texts, model=self.model, input_type="search_document")
        return [[float(x) for x in e] for e in resp.embeddings]


@dataclass
class ZillizBYOMEmbedder:
    """Generic HTTP endpoint compatible with OpenAI embeddings protocol."""

    model: str
    dim: int
    endpoint: str = ""  # resolved from ZILLIZ_BYOM_URL if empty
    name: str = "zilliz-byom"

    def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        url = self.endpoint or resolve_required("ZILLIZ_BYOM_URL")
        key = resolve_required("ZILLIZ_BYOM_KEY")
        resp = httpx.post(
            url.rstrip("/") + "/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": self.model, "input": texts},
            timeout=60.0,
        )
        resp.raise_for_status()
        return [d["embedding"] for d in resp.json()["data"]]


def make_embedder(
    provider: str, model: str | None = None, dim: int | None = None
) -> EmbeddingProvider:
    key = provider.lower()
    if key == "openai":
        return OpenAIEmbedder(
            model=model or "text-embedding-3-small",
            dim=dim or 1536,
        )
    if key == "voyage":
        return VoyageEmbedder(model=model or "voyage-3", dim=dim or 1024)
    if key == "cohere":
        return CohereEmbedder(model=model or "embed-english-v3.0", dim=dim or 1024)
    if key == "zilliz-byom":
        if model is None or dim is None:
            raise BackendUnsupportedError(uri="zilliz-byom", feature="missing model+dim")
        return ZillizBYOMEmbedder(model=model, dim=dim)
    raise BackendUnsupportedError(uri="embedding-provider", feature=provider)


def env_var_for(provider: str) -> str:
    val = _PROVIDER_ENV[provider.lower()]
    if val is None:
        raise BackendUnsupportedError(uri=provider, feature="no-env-var (local provider)")
    return val


# ---------------------------------------------------------------------------
# Image embedders
# ---------------------------------------------------------------------------

# Singleton cache keyed by (model_id, device). open-clip downloads weights to
# ~/.cache/clip on first load; the cache hit costs ~250 ms.
_clip_cache: dict[tuple[str, str], Any] = {}
_DOWNLOAD_NOTICE_SHOWN: set[str] = set()

_CLIP_PRETRAINED = "openai"  # use the OpenAI checkpoint for ViT-B-32


def _maybe_announce_download(model_id: str) -> None:
    """Print a one-line stderr notice the first time we touch a model.

    open-clip caches under ~/.cache/clip; if the file is already there the
    download is a no-op but the notice is harmless context for the user.
    """
    if model_id in _DOWNLOAD_NOTICE_SHOWN:
        return
    cache = Path.home() / ".cache" / "clip"
    if cache.exists() and any(p.name.startswith(model_id) for p in cache.glob("*")):
        _DOWNLOAD_NOTICE_SHOWN.add(model_id)
        return
    print(
        f"info: loading CLIP model {model_id} (one-time, ~150 MB)…",
        file=sys.stderr,
    )
    _DOWNLOAD_NOTICE_SHOWN.add(model_id)


def _load_clip(model_id: str, device_hint: str | None) -> tuple[Any, Any, Any, str]:
    """Return (model, preprocess, tokenizer, device) — cached singleton."""
    _, open_clip = require_multimodal()
    device = device_hint or detect_device_hint()
    cache_key = (model_id, device)
    cached = _clip_cache.get(cache_key)
    if cached is not None:
        return (cached[0], cached[1], cached[2], cached[3])

    _maybe_announce_download(model_id)
    # The OpenAI ViT-B-32 checkpoint was trained with QuickGELU; passing
    # force_quick_gelu silences open-clip's mismatch warning and matches the
    # original training-time architecture for query↔image vector parity.
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_id, pretrained=_CLIP_PRETRAINED, force_quick_gelu=True
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_id)
    bundle = (model, preprocess, tokenizer, device)
    _clip_cache[cache_key] = bundle
    return bundle


def prefetch_clip(model_id: str = "ViT-B-32", device_hint: str | None = None) -> None:
    """Force the model+weights download without doing any embedding."""
    _load_clip(model_id, device_hint)


def embed_image_batch(
    image_paths: Iterable[str | Path],
    *,
    model_id: str = "ViT-B-32",
    device_hint: str | None = None,
    batch_size: int = 16,
) -> list[list[float]]:
    """Embed images via local CLIP. Returns one row of floats per input path.

    Falls back to CPU if the requested device errors at runtime.
    """
    torch, _ = require_multimodal()
    paths = [Path(p) for p in image_paths]
    if not paths:
        return []

    try:
        model, preprocess, _, device = _load_clip(model_id, device_hint)
    except Exception as exc:
        if device_hint and device_hint != "cpu":
            logger.warning(
                "CLIP load failed on device=%s, falling back to cpu: %s", device_hint, exc
            )
            model, preprocess, _, device = _load_clip(model_id, "cpu")
        else:
            raise

    from PIL import Image  # noqa: PLC0415  — Pillow is in base deps

    out: list[list[float]] = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            tensors = []
            for p in chunk:
                with Image.open(p) as img:
                    tensors.append(preprocess(img.convert("RGB")))
            stacked = torch.stack(tensors).to(device)
            try:
                vecs = model.encode_image(stacked)
            except Exception as exc:
                if device != "cpu":
                    logger.warning(
                        "encode_image failed on %s, falling back to cpu: %s", device, exc
                    )
                    model, preprocess, _, device = _load_clip(model_id, "cpu")
                    stacked = stacked.to(device)
                    vecs = model.encode_image(stacked)
                else:
                    raise
            vecs = vecs / vecs.norm(dim=-1, keepdim=True)
            out.extend(v.cpu().tolist() for v in vecs)
    return out


def embed_text_with_clip(
    texts: Iterable[str],
    *,
    model_id: str = "ViT-B-32",
    device_hint: str | None = None,
) -> list[list[float]]:
    """Used at query time to encode the user's text query in the same space."""
    torch, _ = require_multimodal()
    items = list(texts)
    if not items:
        return []
    model, _, tokenizer, device = _load_clip(model_id, device_hint)
    with torch.no_grad():
        tokens = tokenizer(items).to(device)
        vecs = model.encode_text(tokens)
        vecs = vecs / vecs.norm(dim=-1, keepdim=True)
    return [v.cpu().tolist() for v in vecs]


# Voyage multimodal --------------------------------------------------------


def embed_image_batch_voyage(
    image_paths: Iterable[str | Path],
    *,
    model_id: str = "voyage-multimodal-3",
    batch_size: int = 8,
    max_retries: int = 5,
) -> list[list[float]]:
    """Voyage multimodal embedding with 429 retry-on-Retry-After."""
    import voyageai  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    paths = [Path(p) for p in image_paths]
    if not paths:
        return []

    api_key = resolve_required("VOYAGE_API_KEY")
    client = voyageai.Client(api_key=api_key)  # type: ignore[attr-defined]

    out: list[list[float]] = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        # voyageai's multimodal_embed accepts lists of mixed text + Image
        # entries; here one image per input.
        inputs: Any = [[Image.open(p).convert("RGB")] for p in chunk]

        delay = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.multimodal_embed(inputs=inputs, model=model_id, input_type="document")
                break
            except Exception as err:  # voyageai surfaces 429 as ValueError/dict
                msg = str(err).lower()
                retryable = "429" in msg or "rate" in msg or "timeout" in msg
                if attempt == max_retries or not retryable:
                    raise
                # Honor Retry-After header when the SDK exposes it; fall back to backoff.
                retry_after = getattr(err, "retry_after", None)
                wait = float(retry_after) if retry_after else delay
                logger.warning(
                    "Voyage multimodal embed retry %d/%d after %.1fs: %s",
                    attempt,
                    max_retries,
                    wait,
                    err,
                )
                time.sleep(wait)
                delay = min(delay * 2, 30)
        out.extend([list(v) for v in resp.embeddings])
    return out


def make_image_embedder(
    provider: str, model: str | None = None, device_hint: str | None = None
) -> Any:
    """Return a callable `(paths: Iterable[str|Path]) -> list[list[float]]`.

    Image embedders don't share the EmbeddingProvider Protocol because their
    input is paths, not strings. Caller branches on `embedding.modality`.
    """
    key = provider.lower()
    if key == "clip-local":
        m = model or "ViT-B-32"

        def _run(paths: Iterable[str | Path]) -> list[list[float]]:
            return embed_image_batch(paths, model_id=m, device_hint=device_hint)

        return _run
    if key == "voyage":
        m = model or "voyage-multimodal-3"
        # Pre-flight the credential so Phase 4 fails before touching Milvus.
        from .credentials import resolve  # noqa: PLC0415

        if not resolve("VOYAGE_API_KEY", optional=True):
            raise MissingCredentialError(env_var="VOYAGE_API_KEY")

        def _run(paths: Iterable[str | Path]) -> list[list[float]]:
            return embed_image_batch_voyage(paths, model_id=m)

        return _run
    raise BackendUnsupportedError(uri="image-embedding-provider", feature=provider)
