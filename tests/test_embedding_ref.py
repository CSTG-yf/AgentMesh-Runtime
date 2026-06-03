from pathlib import Path

from agentmesh.state.embedding import HashEmbeddingEncoder, cosine_similarity
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


def test_embedding_can_be_written_to_state_store(tmp_path: Path) -> None:
    store = StateStore(RuntimePaths(root=tmp_path))
    embedding = HashEmbeddingEncoder().encode("protocol memory")

    ref = store.put_embedding(trace_id="trace-1", producer="retriever", embedding=embedding)
    record, payload = store.get(ref)

    assert record.state_type == StateType.EMBEDDING
    assert payload == embedding
