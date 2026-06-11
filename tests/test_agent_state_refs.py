from pathlib import Path

from agentmesh.agents.executor import ExecutorAgent
from agentmesh.agents.planner import PlannerAgent
from agentmesh.agents.retriever import RetrieverAgent
from agentmesh.memory.hybrid_store import HybridMemoryStore
from agentmesh.memory.schema import MemoryUnit
from agentmesh.modes.protocol_mode import run_protocol_mode
from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.schema import AMPMessage
from agentmesh.runtime.registry import RuntimeContext
from agentmesh.state.embedding import create_embedding_encoder
from agentmesh.state.store import StateStore
from agentmesh.storage.jsonl import read_jsonl
from agentmesh.storage.paths import RuntimePaths


def test_planner_reads_task_from_state_ref_when_params_are_empty(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    trace_id = "trace-state-planner"
    state_store = StateStore(paths)
    task_ref = state_store.put_text(
        trace_id=trace_id,
        producer="runtime",
        text="Write quicksort and output the sorted result.",
    )
    context = RuntimeContext.from_paths(
        paths=paths,
        trace_id=trace_id,
        load_configured_llm=False,
    )

    result = PlannerAgent().handle(
        AMPMessage(
            trace_id=trace_id,
            source_agent="runtime",
            target_agent="planner",
            msg_type=MsgType.INVOKE,
            action="plan.create",
            params={},
            state_refs=[task_ref],
        ),
        context,
    )

    assert result.result["topic"].startswith("Write quicksort")
    assert result.result["decision"]["need_tool_execution"] is True
    record, _ = StateStore(paths).get(task_ref)
    assert "planner" in record.consumers


def test_retriever_reads_query_ref_and_returns_memory_hits(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    trace_id = "trace-state-retriever"
    state_store = StateStore(paths)
    context = RuntimeContext.from_paths(
        paths=paths,
        trace_id=trace_id,
        load_configured_llm=False,
    )
    encoder = create_embedding_encoder(context.config.embedding)
    memory_store = HybridMemoryStore(paths=paths, state_store=state_store, encoder=encoder)
    memory_store.put(
        MemoryUnit(
            source_agent="summarizer",
            task_topic="quicksort prior result",
            summary="Quick sort previously produced sorted output in a sandbox.",
            tags=["code"],
            evidence_refs=[],
            state_refs=[],
            embedding_vector=encoder.encode("quicksort sorted output sandbox"),
            confidence=0.9,
            validity_score=0.9,
            provenance_trace_id="trace-memory",
        )
    )
    query_ref = state_store.put_text(
        trace_id=trace_id,
        producer="runtime",
        text="Need quicksort sorted output.",
    )
    result = RetrieverAgent().handle(
        AMPMessage(
            trace_id=trace_id,
            source_agent="planner",
            target_agent="retriever",
            msg_type=MsgType.INVOKE,
            action="memory.semantic_search",
            params={},
            state_refs=[query_ref],
        ),
        context,
    )

    evidence = result.result["evidence"]
    assert any(item.get("memory_id") for item in evidence)
    assert any(item.get("title") == "quicksort prior result" for item in evidence)
    record, _ = StateStore(paths).get(query_ref)
    assert "retriever" in record.consumers


def test_executor_reads_task_from_state_ref_for_codeact(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    trace_id = "trace-state-executor"
    state_store = StateStore(paths)
    task_ref = state_store.put_text(
        trace_id=trace_id,
        producer="runtime",
        text="[3,1,2] Write quicksort and output the sorted result.",
    )
    context = RuntimeContext.from_paths(
        paths=paths,
        trace_id=trace_id,
        load_configured_llm=False,
    )

    result = ExecutorAgent().handle(
        AMPMessage(
            trace_id=trace_id,
            source_agent="retriever",
            target_agent="executor",
            msg_type=MsgType.INVOKE,
            action="tool.run_python",
            params={},
            state_refs=[task_ref],
        ),
        context,
    )

    assert "data = [3, 1, 2]" in result.result["codeact_code"]
    record, _ = StateStore(paths).get(task_ref)
    assert "executor" in record.consumers


def test_protocol_mode_hands_off_task_by_state_ref_not_full_params(tmp_path: Path) -> None:
    task = tmp_path / "task.txt"
    task_text = "[3,1,2] Write quicksort and output the sorted result."
    task.write_text(task_text, encoding="utf-8")
    paths = RuntimePaths(root=tmp_path)

    result = run_protocol_mode(task_path=task, paths=paths, load_configured_llm=False)

    assert "executor" in result.metrics.selected_agents
    for item in read_jsonl(paths.protocol_agent_io):
        if item.get("trace_id") != result.trace_id:
            continue
        params = item["input"]["params"]
        assert params.get("task") != task_text
        assert task_text not in str(params)
        assert item["state_refs_in"]
