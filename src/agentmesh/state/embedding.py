import hashlib
import math
import urllib.error
import urllib.request
from typing import Protocol

import orjson

from agentmesh.config import EmbeddingConfig
from agentmesh.core import rust_available, rust_core


class EmbeddingEncoder(Protocol):
    dimensions: int
    provider: str = "unknown"

    def encode(self, text: str) -> list[float]:
        raise NotImplementedError


def is_semantic_encoder(encoder: object) -> bool:
    """Return True if the encoder produces real semantic embeddings.

    HashEmbeddingEncoder uses locality-sensitive hashing which cannot
    capture semantic meaning — it returns near-random cosine similarities
    for conceptually similar but lexically different text.
    """
    if isinstance(encoder, TEIEmbeddingEncoder):
        return True
    return False


class HashEmbeddingEncoder:
    provider = "hash"

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def encode(self, text: str) -> list[float]:
        tokens = _hash_embedding_tokens(text)
        if rust_available() and hasattr(rust_core(), "hash_embedding"):
            vector = [float(value) for value in rust_core().hash_embedding(text, self.dimensions)]
            if any(value != 0.0 for value in vector) or not tokens:
                return vector
        if not tokens:
            return [0.0] * self.dimensions
        bigrams = [
            f"{left}_{right}"
            for left, right in zip(tokens, tokens[1:], strict=False)
        ]
        features = tokens + bigrams
        vector = [0.0] * self.dimensions
        for token in features:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def _hash_embedding_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    ascii_buffer: list[str] = []

    def flush_ascii() -> None:
        if ascii_buffer:
            tokens.append("".join(ascii_buffer))
            ascii_buffer.clear()

    for char in text.lower():
        if char.isascii() and char.isalnum():
            ascii_buffer.append(char)
            continue
        flush_ascii()
        if _is_cjk(char):
            tokens.append(char)
    flush_ascii()
    return tokens


def _is_cjk(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if rust_available() and hasattr(rust_core(), "cosine_similarity_f32"):
        return float(rust_core().cosine_similarity_f32(left, right))
    if len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


class TEIEmbeddingEncoder:
    provider = "tei"

    def __init__(
        self,
        *,
        base_url: str,
        model: str = "BAAI/bge-small-zh-v1.5",
        dimensions: int = 512,
        timeout_seconds: float = 15.0,
        fallback: EmbeddingEncoder | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback or HashEmbeddingEncoder(dimensions=dimensions)

    def encode(self, text: str) -> list[float]:
        try:
            return self._encode_with_tei(text)
        except Exception:
            return self.fallback.encode(text)

    def _encode_with_tei(self, text: str) -> list[float]:
        request = urllib.request.Request(
            f"{self.base_url}/embed",
            data=orjson.dumps({"inputs": text}),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = orjson.loads(response.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"TEI embedding request failed: {exc}") from exc
        vector = _extract_embedding(payload)
        self.dimensions = len(vector)
        return vector


def create_embedding_encoder(config: EmbeddingConfig) -> EmbeddingEncoder:
    fallback = HashEmbeddingEncoder(dimensions=config.dimensions)
    if config.tei_configured and config.base_url is not None:
        return TEIEmbeddingEncoder(
            base_url=config.base_url,
            model=config.model,
            dimensions=config.dimensions,
            timeout_seconds=config.timeout_seconds,
            fallback=fallback,
        )
    return fallback


def _extract_embedding(payload: object) -> list[float]:
    candidate = payload
    if isinstance(payload, dict):
        candidate = payload.get("embedding") or payload.get("embeddings") or payload.get("data")
    if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
        first = candidate[0]
        candidate = first.get("embedding") or first.get("embeddings")
    if isinstance(candidate, list) and candidate and isinstance(candidate[0], list):
        candidate = candidate[0]
    if not isinstance(candidate, list):
        raise RuntimeError("TEI response did not contain an embedding vector")
    return [float(value) for value in candidate]
