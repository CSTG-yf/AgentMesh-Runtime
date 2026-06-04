import hashlib
import math
import re

from agentmesh.core import rust_available, rust_core


class HashEmbeddingEncoder:
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def encode(self, text: str) -> list[float]:
        if rust_available():
            return [float(value) for value in rust_core().hash_embedding(text, self.dimensions)]
        tokens = re.findall(r"[a-z0-9]+", text.lower())
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


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if rust_available():
        return float(rust_core().cosine_similarity_f32(left, right))
    if len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
