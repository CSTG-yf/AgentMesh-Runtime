from pathlib import Path

from agentmesh.memory.policy import MemoryWritePolicy
from agentmesh.memory.schema import MemoryUnit
from agentmesh.memory.scorer import MemoryScorer
from agentmesh.memory.sqlite_store import SQLiteMemoryStore
from agentmesh.state.embedding import HashEmbeddingEncoder
from agentmesh.state.store import StateStore
from agentmesh.storage.paths import RuntimePaths


def test_memory_store_puts_and_searches_by_keyword_tag_and_semantic(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    state_store = StateStore(paths)
    encoder = HashEmbeddingEncoder()
    embedding_ref = state_store.put_embedding(
        trace_id="trace-1",
        producer="summarizer",
        embedding=encoder.encode("structured protocol state passing"),
    )
    memory_store = SQLiteMemoryStore(paths=paths, state_store=state_store, encoder=encoder)
    unit = MemoryUnit(
        source_agent="summarizer",
        task_topic="protocol design",
        summary="Structured protocol state passing reduces repeated text.",
        tags=["protocol", "state"],
        evidence_refs=[],
        state_refs=[],
        embedding_ref=embedding_ref,
        confidence=0.9,
        validity_score=0.8,
        provenance_trace_id="trace-1",
    )

    memory_store.put(unit)

    assert memory_store.keyword_search("structured")[0].memory_id == unit.memory_id
    assert memory_store.tag_search("protocol")[0].memory_id == unit.memory_id
    semantic_result = memory_store.semantic_search("agent protocol state transfer")
    assert semantic_result[0].memory_id == unit.memory_id


def test_memory_policy_and_scorer_are_deterministic() -> None:
    unit = MemoryUnit(
        source_agent="summarizer",
        task_topic="benchmark",
        summary="High confidence evidence backed memory.",
        tags=["benchmark"],
        evidence_refs=["state://evidence/state-1"],
        state_refs=[],
        confidence=0.8,
        validity_score=0.9,
        provenance_trace_id="trace-1",
    )

    assert MemoryWritePolicy(min_confidence=0.5).should_write(unit)
    assert MemoryScorer().score(
        semantic_similarity=0.7,
        tag_match_score=1.0,
        confidence=unit.confidence,
        evidence_coverage=1.0,
        time_decay=0.0,
    ) > 0.7
