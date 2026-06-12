from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

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
    assert memory_store.keyword_search("structured-protocol")[0].memory_id == unit.memory_id
    assert memory_store.tag_search("protocol")[0].memory_id == unit.memory_id
    semantic_result = memory_store.semantic_search("agent protocol state transfer")
    assert semantic_result[0].memory_id == unit.memory_id


def test_memory_store_semantic_search_uses_inline_embedding_vector(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    state_store = StateStore(paths)
    encoder = HashEmbeddingEncoder()
    memory_store = SQLiteMemoryStore(paths=paths, state_store=state_store, encoder=encoder)
    unit = MemoryUnit(
        source_agent="summarizer",
        task_topic="inline vector memory",
        summary="Protocol feedback loop benchmark evidence.",
        tags=["protocol", "benchmark"],
        evidence_refs=[],
        state_refs=[],
        embedding_vector=encoder.encode("protocol feedback benchmark"),
        confidence=0.9,
        validity_score=0.9,
        provenance_trace_id="trace-inline",
    )

    memory_store.put(unit)

    results = memory_store.semantic_search("protocol feedback benchmark")
    assert results[0].memory_id == unit.memory_id


def test_memory_store_semantic_search_handles_chinese_query(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    state_store = StateStore(paths)
    encoder = HashEmbeddingEncoder()
    memory_store = SQLiteMemoryStore(paths=paths, state_store=state_store, encoder=encoder)
    relevant = MemoryUnit(
        source_agent="summarizer",
        task_topic="留学择校",
        summary="留学择校问题需要比较学校排名、专业匹配、费用和签证风险。",
        tags=["study_abroad"],
        evidence_refs=[],
        state_refs=[],
        embedding_vector=encoder.encode("留学择校问题的考量因素"),
        confidence=0.9,
        validity_score=0.9,
        provenance_trace_id="trace-cn",
    )
    unrelated = MemoryUnit(
        source_agent="summarizer",
        task_topic="图搜索",
        summary="图搜索通过遍历相邻节点查找目标路径。",
        tags=["code"],
        evidence_refs=[],
        state_refs=[],
        embedding_vector=encoder.encode("图搜索代码实现"),
        confidence=0.9,
        validity_score=0.9,
        provenance_trace_id="trace-sort",
    )

    memory_store.put(unrelated)
    memory_store.put(relevant)

    results = memory_store.semantic_search("留学选校因素", limit=1)
    assert results[0].memory_id == relevant.memory_id


def test_memory_store_semantic_search_prefilters_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = RuntimePaths(root=tmp_path)
    state_store = StateStore(paths)
    encoder = HashEmbeddingEncoder()
    memory_store = SQLiteMemoryStore(paths=paths, state_store=state_store, encoder=encoder)
    for index in range(20):
        memory_store.put(
            MemoryUnit(
                source_agent="summarizer",
                task_topic=f"memory {index}",
                summary=(
                    "needle protocol evidence"
                    if index == 0
                    else f"unrelated filler memory {index}"
                ),
                tags=["target"] if index == 1 else ["general"],
                evidence_refs=[],
                state_refs=[],
                embedding_vector=encoder.encode(f"memory {index}"),
                confidence=0.8,
                validity_score=0.8,
                provenance_trace_id=f"trace-{index}",
            )
        )
    calls = 0
    original = memory_store._embedding_payload

    def counted(unit: MemoryUnit) -> list[float] | None:
        nonlocal calls
        calls += 1
        return original(unit)

    monkeypatch.setattr(memory_store, "_embedding_payload", counted)

    results = memory_store.semantic_search_with_scores(
        "needle protocol evidence",
        query_tags=["target"],
        candidate_limit=5,
    )

    assert results
    assert calls <= 15


def test_memory_store_semantic_search_uses_tag_candidates_beyond_recent_window(
    tmp_path: Path,
) -> None:
    paths = RuntimePaths(root=tmp_path)
    state_store = StateStore(paths)
    encoder = HashEmbeddingEncoder()
    memory_store = SQLiteMemoryStore(paths=paths, state_store=state_store, encoder=encoder)
    old_target = MemoryUnit(
        source_agent="summarizer",
        created_at=datetime.now(UTC) - timedelta(days=30),
        task_topic="old target",
        summary="Old memory selected by tag.",
        tags=["target"],
        evidence_refs=[],
        state_refs=[],
        embedding_vector=encoder.encode("old target"),
        confidence=0.8,
        validity_score=0.8,
        provenance_trace_id="trace-old",
    )
    memory_store.put(old_target)
    for index in range(5):
        memory_store.put(
            MemoryUnit(
                source_agent="summarizer",
                task_topic=f"recent {index}",
                summary=f"Recent unrelated memory {index}.",
                tags=["general"],
                evidence_refs=[],
                state_refs=[],
                embedding_vector=encoder.encode(f"recent {index}"),
                confidence=0.8,
                validity_score=0.8,
                provenance_trace_id=f"trace-recent-{index}",
            )
        )

    results = memory_store.semantic_search_with_scores(
        "no lexical hit",
        query_tags=["target"],
        candidate_limit=1,
    )

    assert any(result.memory.memory_id == old_target.memory_id for result in results)


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


def test_long_term_memory_policy_requires_importance_and_validity() -> None:
    unit = MemoryUnit(
        source_agent="summarizer",
        task_topic="low value",
        summary="This task has a summary but should not enter long term memory.",
        tags=["general"],
        evidence_refs=[],
        state_refs=[],
        confidence=0.8,
        validity_score=0.9,
        importance_score=0.2,
        provenance_trace_id="trace-1",
    )

    policy = MemoryWritePolicy()

    assert policy.should_write(unit)
    assert not policy.should_write_long_term(unit)
    assert policy.should_write_long_term(unit.model_copy(update={"importance_score": 0.8}))
