from pathlib import Path

from agentmesh.core import rust_core
from agentmesh.memory.schema import MemoryUnit
from agentmesh.memory.sqlite_store import SQLiteMemoryStore
from agentmesh.state.embedding import HashEmbeddingEncoder
from agentmesh.state.store import StateStore
from agentmesh.storage.paths import RuntimePaths


def test_rust_top_k_cosine_returns_best_indexes() -> None:
    indexes = rust_core().top_k_cosine(
        [1.0, 0.0],
        [
            [0.0, 1.0],
            [0.9, 0.1],
            [0.5, 0.5],
        ],
        2,
    )

    assert indexes == [1, 2]


def test_semantic_search_prefers_related_memory(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    state_store = StateStore(paths)
    encoder = HashEmbeddingEncoder()
    store = SQLiteMemoryStore(paths, state_store, encoder)

    rust_ref = state_store.put_embedding("trace-1", "tester", encoder.encode("rust runtime state"))
    docs_ref = state_store.put_embedding(
        "trace-1",
        "tester",
        encoder.encode("readme documentation"),
    )

    store.put(
        MemoryUnit(
            source_agent="tester",
            task_topic="rust runtime",
            summary="Rust runtime state passing optimization",
            tags=["runtime"],
            embedding_ref=rust_ref,
            provenance_trace_id="trace-1",
        )
    )
    store.put(
        MemoryUnit(
            source_agent="tester",
            task_topic="documentation",
            summary="README documentation writing",
            tags=["docs"],
            embedding_ref=docs_ref,
            provenance_trace_id="trace-1",
        )
    )

    results = store.semantic_search("state runtime optimization", limit=1)

    assert results[0].task_topic == "rust runtime"
