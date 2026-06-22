from pathlib import Path

from agentmesh.agents.retriever import _memory_reuse_limit, _should_reuse_memory_result
from agentmesh.memory.hybrid_store import HybridMemoryStore
from agentmesh.memory.schema import MemoryUnit
from agentmesh.modes.protocol_mode import run_protocol_mode
from agentmesh.state.embedding import HashEmbeddingEncoder
from agentmesh.state.store import StateStore
from agentmesh.storage.paths import RuntimePaths


def test_hybrid_store_writes_only_important_units_to_global_memory(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    state_store = StateStore(paths)
    encoder = HashEmbeddingEncoder()
    embedding = encoder.encode("important protocol memory reuse")
    embedding_ref = state_store.put_embedding(
        trace_id="trace-1",
        producer="test",
        embedding=embedding,
    )
    store = HybridMemoryStore(paths=paths, state_store=state_store, encoder=encoder)

    low_value = MemoryUnit(
        source_agent="summarizer",
        task_topic="low",
        summary="A valid but low importance memory should stay in the current run only.",
        tags=["protocol"],
        embedding_ref=embedding_ref,
        embedding_vector=embedding,
        confidence=0.8,
        validity_score=0.9,
        importance_score=0.2,
        provenance_trace_id="trace-1",
    )
    important = low_value.model_copy(
        update={
            "memory_id": "mem-important",
            "task_topic": "important",
            "importance_score": 0.9,
        }
    )

    assert store.put(low_value) == {"run_written": True, "global_written": False}
    assert store.put(important) == {"run_written": True, "global_written": True}
    assert len(store.run_store.all_units()) == 2
    assert [unit.memory_id for unit in store.global_store.all_units()] == ["mem-important"]


def test_protocol_mode_reuses_global_memory_across_runtime_paths(tmp_path: Path) -> None:
    first_task = tmp_path / "first.txt"
    first_task.write_text(
        "Analyze protocol memory benchmark state reuse in AgentMesh Runtime.",
        encoding="utf-8",
    )
    first_paths = RuntimePaths(root=tmp_path)

    first = run_protocol_mode(first_task, first_paths)

    assert first.metrics.memory_hit_count == 0
    assert first_paths.global_memory_db.exists()
    cleanup_store = HybridMemoryStore(
        paths=first_paths,
        state_store=StateStore(first_paths),
        encoder=HashEmbeddingEncoder(),
    )
    for unit in cleanup_store.run_store.all_units():
        cleanup_store.run_store.archive(unit.memory_id, "test isolates global reuse")

    second_task = tmp_path / "second.txt"
    second_task.write_text(
        "Explain how AgentMesh Runtime reuses protocol memory for benchmark state tasks.",
        encoding="utf-8",
    )
    second_paths = RuntimePaths(root=tmp_path)
    second = run_protocol_mode(second_task, second_paths)

    assert second.metrics.memory_hit_count == 1


def test_protocol_mode_records_generic_code_task_memory(tmp_path: Path) -> None:
    task = tmp_path / "code_task.txt"
    task.write_text(
        "Write Python code that validates a small task.",
        encoding="utf-8",
    )
    paths = RuntimePaths(root=tmp_path)

    result = run_protocol_mode(task, paths, load_configured_llm=False)

    assert "executor" in result.metrics.selected_agents
    assert result.metrics.memory_query_count == 1
    assert paths.memory_db.exists()


def test_memory_retriever_limits_general_and_explicit_reuse_queries() -> None:
    assert _memory_reuse_limit(query="hello", query_tags=["general"]) == 1
    assert _memory_reuse_limit(query="复用之前的 protocol memory", query_tags=["protocol"]) == 2
    assert _memory_reuse_limit(query="Explain protocol routing", query_tags=["protocol"]) == 1
    assert (
        _memory_reuse_limit(
            query="Explain protocol routing",
            query_tags=["protocol"],
            default_limit=2,
        )
        == 2
    )


def test_memory_retriever_requires_stronger_similarity_for_unmatched_tags() -> None:
    assert not _should_reuse_memory_result(
        score=0.69,
        semantic_similarity=0.71,
        tag_overlap_score=0.0,
        query_tags=["dynamic-programming"],
    )
    assert _should_reuse_memory_result(
        score=0.71,
        semantic_similarity=0.73,
        tag_overlap_score=0.0,
        query_tags=["dynamic-programming"],
    )
    assert not _should_reuse_memory_result(
        score=0.59,
        semantic_similarity=0.54,
        tag_overlap_score=0.0,
        query_tags=["general"],
    )
