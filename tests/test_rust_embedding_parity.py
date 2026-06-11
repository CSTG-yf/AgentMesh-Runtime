import hashlib
import math

import pytest

from agentmesh.core import rust_available, rust_core
from agentmesh.state.embedding import (
    HashEmbeddingEncoder,
    _hash_embedding_tokens,
    cosine_similarity,
)

requires_rust_core = pytest.mark.skipif(
    not rust_available(),
    reason="agentmesh_core Rust extension is not installed",
)


def _python_reference_hash_embedding(text: str, dimensions: int) -> list[float]:
    tokens = _hash_embedding_tokens(text)
    if not tokens:
        return [0.0] * dimensions
    bigrams = [
        f"{left}_{right}"
        for left, right in zip(tokens, tokens[1:], strict=False)
    ]
    features = tokens + bigrams
    vector = [0.0] * dimensions
    for token in features:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


@requires_rust_core
def test_rust_hash_embedding_matches_python_reference() -> None:
    text = "agent state passing memory 留学择校因素"
    expected = _python_reference_hash_embedding(text, 384)
    actual = [float(value) for value in rust_core().hash_embedding(text, 384)]

    assert actual == expected


def test_hash_embedding_is_deterministic() -> None:
    encoder = HashEmbeddingEncoder(dimensions=384)
    left = encoder.encode("agent state passing memory")
    right = encoder.encode("agent state passing memory")

    assert left == right
    assert len(left) == 384
    assert abs(cosine_similarity(left, right) - 1.0) < 1e-6
