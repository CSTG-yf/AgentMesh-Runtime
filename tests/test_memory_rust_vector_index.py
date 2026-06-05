from pathlib import Path

import pytest

from agentmesh.core import rust_available, rust_core
from agentmesh.memory.schema import MemoryUnit
from agentmesh.memory.sqlite_store import SQLiteMemoryStore
from agentmesh.state.embedding import HashEmbeddingEncoder
from agentmesh.state.store import StateStore
from agentmesh.storage.paths import RuntimePaths

requires_rust_core = pytest.mark.skipif(
    not rust_available(),
    reason="agentmesh_core Rust extension is not installed",
)

@requires_rust_core
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


@requires_rust_core
def test_rust_memory_rank_top_k_combines_similarity_and_metadata() -> None:
    ranked = rust_core().memory_rank_top_k(
        [1.0, 0.0],
        [
            [0.0, 1.0],
            [0.8, 0.2],
            [0.7, 0.3],
        ],
        [1.0, 0.9, 0.9],
        [0.9, 0.9, 0.9],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        2,
    )

    assert ranked[0][0] == 1
    assert ranked[0][1] > ranked[1][1]
    assert ranked[0][2] > 0.9


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


def test_semantic_search_uses_rust_memory_ranking_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = RuntimePaths(root=tmp_path)
    state_store = StateStore(paths)
    encoder = HashEmbeddingEncoder()
    store = SQLiteMemoryStore(paths, state_store, encoder)
    embedding_ref = state_store.put_embedding(
        "trace-1",
        "tester",
        encoder.encode("structured memory ranking"),
    )
    store.put(
        MemoryUnit(
            source_agent="tester",
            task_topic="ranking",
            summary="Structured memory ranking with Rust hot path.",
            tags=["memory"],
            embedding_ref=embedding_ref,
            confidence=0.9,
            validity_score=0.9,
            provenance_trace_id="trace-1",
        )
    )
    calls: list[dict[str, object]] = []

    active_unit = store.active_units()[0]
    assert active_unit.embedding_ref == embedding_ref
    assert store._embedding_payload(active_unit) is not None

    class FakeRustCore:
        def memory_rank_top_k(self, *args: object) -> list[tuple[int, float, float]]:
            calls.append({"args": args})
            return [(0, 0.88, 0.77)]

    monkeypatch.setattr("agentmesh.memory.sqlite_store.rust_available", lambda: True)
    monkeypatch.setattr("agentmesh.memory.sqlite_store.rust_core", lambda: FakeRustCore())
    monkeypatch.setattr("agentmesh.memory.sqlite_store._rust_memory_rank_available", lambda: True)

    results = store.semantic_search_with_scores("memory ranking", limit=1)

    assert calls
    assert results[0].memory.task_topic == "ranking"
    assert results[0].score == 0.88
    assert results[0].semantic_similarity == 0.77
