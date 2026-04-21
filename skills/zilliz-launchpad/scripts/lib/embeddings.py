"""API-based embedding providers.

Strategy pattern keyed on provider name. Each strategy is a thin HTTP
wrapper — no local model weights, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .credentials import resolve_required
from .errors import BackendUnsupportedError

_PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "voyage": "VOYAGE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "zilliz-byom": "ZILLIZ_BYOM_KEY",
}


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
    return _PROVIDER_ENV[provider.lower()]
