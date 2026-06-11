from pathlib import Path
from typing import Any

import orjson

from agentmesh.config import EmbeddingConfig
from agentmesh.state.embedding import (
    HashEmbeddingEncoder,
    TEIEmbeddingEncoder,
    cosine_similarity,
    create_embedding_encoder,
)
from agentmesh.state.schema import StateType
from agentmesh.state.store import StateStore
from agentmesh.storage.paths import RuntimePaths


def test_hash_embedding_is_deterministic_normalized_and_semantic_enough() -> None:
    encoder = HashEmbeddingEncoder(dimensions=384)

    first = encoder.encode("Agent state passing with shared memory")
    second = encoder.encode("Agent state passing with shared memory")
    similar = encoder.encode("Agent state transfer and shared memory")
    unrelated = encoder.encode("Weather forecast for a mountain trip")

    assert first == second
    assert abs(sum(value * value for value in first) - 1.0) < 1e-9
    assert cosine_similarity(first, similar) > cosine_similarity(first, unrelated)
    assert encoder.encode("") == [0.0] * 384


def test_hash_embedding_handles_chinese_text() -> None:
    encoder = HashEmbeddingEncoder(dimensions=384)

    vector = encoder.encode("留学择校问题的考量因素")
    similar = encoder.encode("留学选校因素和考量")
    unrelated = encoder.encode("快速排序代码实现")

    assert sum(abs(value) for value in vector) > 0
    assert abs(sum(value * value for value in vector) - 1.0) < 1e-9
    assert cosine_similarity(vector, similar) > cosine_similarity(vector, unrelated)


def test_embedding_can_be_written_to_state_store(tmp_path: Path) -> None:
    store = StateStore(RuntimePaths(root=tmp_path))
    embedding = HashEmbeddingEncoder().encode("protocol memory")

    ref = store.put_embedding(trace_id="trace-1", producer="retriever", embedding=embedding)
    record, payload = store.get(ref)

    assert record.state_type == StateType.EMBEDDING
    assert payload == embedding


def test_tei_embedding_encoder_calls_embed_endpoint(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return orjson.dumps([[0.1, 0.2, 0.3]])

    def fake_urlopen(request: Any, timeout: float) -> Response:
        captured["url"] = request.full_url
        captured["body"] = orjson.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    encoder = TEIEmbeddingEncoder(
        base_url="http://127.0.0.1:8080",
        timeout_seconds=7,
    )

    assert encoder.encode("中文语义检索") == [0.1, 0.2, 0.3]
    assert captured == {
        "url": "http://127.0.0.1:8080/embed",
        "body": {"inputs": "中文语义检索"},
        "timeout": 7,
    }


def test_create_embedding_encoder_uses_tei_when_configured() -> None:
    encoder = create_embedding_encoder(
        EmbeddingConfig(
            provider="tei",
            base_url="http://127.0.0.1:8080",
            dimensions=512,
        )
    )

    assert isinstance(encoder, TEIEmbeddingEncoder)
