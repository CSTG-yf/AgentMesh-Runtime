from pathlib import Path

from agentmesh.agents.executor import ExecutorAgent
from agentmesh.agents.planner import PlannerAgent
from agentmesh.agents.retriever import RetrieverAgent
from agentmesh.agents.summarizer import SummarizerAgent
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


class InvalidExecutorLLM:
    def complete(self, **kwargs: object) -> str:
        del kwargs
        return 'solution_md = """# DP solution\n## Code\n'


class CapturingLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def complete(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return self.response


def test_planner_reads_task_from_state_ref_when_params_are_empty(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    trace_id = "trace-state-planner"
    state_store = StateStore(paths)
    task_ref = state_store.put_text(
        trace_id=trace_id,
        producer="runtime",
        text="Write Python code that validates a small task.",
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

    assert result.result["topic"].startswith("Write Python code")
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
            task_topic="code validation prior result",
            summary="A prior code task produced validation output in a sandbox.",
            tags=["code"],
            evidence_refs=[],
            state_refs=[],
            embedding_vector=encoder.encode("code validation output sandbox"),
            confidence=0.9,
            validity_score=0.9,
            provenance_trace_id="trace-memory",
        )
    )
    query_ref = state_store.put_text(
        trace_id=trace_id,
        producer="runtime",
        text="Need code validation output.",
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
    assert any(item.get("title") == "code validation prior result" for item in evidence)
    assert "memory_hits" not in result.result
    memory_hit_logs = read_jsonl(paths.protocol_memory_hits)
    assert memory_hit_logs
    assert memory_hit_logs[0]["query"] == "Need code validation output."
    memory_hits = memory_hit_logs[0]["memory_hits"]
    assert memory_hits[0]["task_topic"] == "code validation prior result"
    assert memory_hits[0]["summary"] == (
        "A prior code task produced validation output in a sandbox."
    )
    assert memory_hits[0]["memory_id"]
    assert memory_hits[0]["score"] > 0
    record, _ = StateStore(paths).get(query_ref)
    assert "retriever" in record.consumers


def test_deterministic_retriever_returns_memory_evidence_without_llm(
    tmp_path: Path,
) -> None:
    paths = RuntimePaths(root=tmp_path)
    trace_id = "trace-deterministic-retriever"
    state_store = StateStore(paths)
    client = CapturingLLM("rewritten evidence")
    context = RuntimeContext.from_paths(
        paths=paths,
        trace_id=trace_id,
        llm_client=client,
    )
    encoder = create_embedding_encoder(context.config.embedding)
    memory_store = HybridMemoryStore(paths=paths, state_store=state_store, encoder=encoder)
    evidence_ref = state_store.put_evidence(
        trace_id="trace-memory-source",
        producer="summarizer",
        evidence=[{"title": "validated output", "snippet": "sandbox passed"}],
    )
    unit = MemoryUnit(
        source_agent="summarizer",
        task_topic="code validation prior result",
        summary="A prior code task produced validation output in a sandbox.",
        tags=["code"],
        evidence_refs=[evidence_ref],
        state_refs=[],
        embedding_vector=encoder.encode("code validation output sandbox"),
        confidence=0.9,
        validity_score=0.9,
        provenance_trace_id="trace-memory-source",
    )
    memory_store.put(unit)
    context = context.model_copy(update={"memory_store": memory_store})

    result = RetrieverAgent().handle(
        AMPMessage(
            trace_id=trace_id,
            source_agent="planner",
            target_agent="retriever",
            msg_type=MsgType.INVOKE,
            action="memory.semantic_search",
            params={
                "query": "Need code validation output.",
                "deterministic_retrieval_evidence": True,
            },
        ),
        context,
    )

    assert client.calls == []
    assert len(result.result["evidence"]) == 1
    evidence = result.result["evidence"][0]
    assert evidence["memory_id"]
    assert evidence["source_agent"] == "summarizer"
    assert evidence["provenance_trace_id"] == "trace-memory-source"
    assert evidence["evidence_refs"] == [evidence_ref]
    assert evidence["memory_score"] > 0
    assert evidence["snippet"] == unit.summary


def test_deterministic_retriever_returns_empty_evidence_without_hit(
    tmp_path: Path,
) -> None:
    paths = RuntimePaths(root=tmp_path)
    client = CapturingLLM("rewritten evidence")
    context = RuntimeContext.from_paths(
        paths=paths,
        trace_id="trace-deterministic-empty",
        llm_client=client,
    )

    result = RetrieverAgent().handle(
        AMPMessage(
            trace_id=context.trace_id,
            source_agent="planner",
            target_agent="retriever",
            msg_type=MsgType.INVOKE,
            action="memory.semantic_search",
            params={
                "query": "No matching memory exists.",
                "deterministic_retrieval_evidence": True,
            },
        ),
        context,
    )

    assert client.calls == []
    assert result.result["evidence"] == []


def test_compact_evidence_state_preserves_memory_provenance() -> None:
    from agentmesh.modes.protocol_mode import _compact_evidence_items

    compacted = _compact_evidence_items(
        [
            {
                "title": "prior result",
                "snippet": "validated output",
                "memory_id": "memory-1",
                "memory_score": 0.9,
                "source_agent": "summarizer",
                "provenance_trace_id": "trace-prior",
                "evidence_refs": ["state://evidence/prior-1"],
            }
        ],
        snippet_limit=80,
    )

    assert compacted[0]["source_agent"] == "summarizer"
    assert compacted[0]["provenance_trace_id"] == "trace-prior"
    assert compacted[0]["evidence_refs"] == ["state://evidence/prior-1"]


def test_default_retriever_still_calls_llm(tmp_path: Path) -> None:
    client = CapturingLLM("rewritten evidence")
    context = RuntimeContext.from_paths(
        paths=RuntimePaths(root=tmp_path),
        trace_id="trace-default-retriever",
        llm_client=client,
    )

    RetrieverAgent().handle(
        AMPMessage(
            trace_id=context.trace_id,
            source_agent="planner",
            target_agent="retriever",
            msg_type=MsgType.INVOKE,
            action="memory.semantic_search",
            params={"query": "Find prior evidence."},
        ),
        context,
    )

    assert len(client.calls) == 1


def test_executor_reads_task_from_state_ref_for_codeact(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    trace_id = "trace-state-executor"
    state_store = StateStore(paths)
    task_ref = state_store.put_text(
        trace_id=trace_id,
        producer="runtime",
        text="Write Python code that validates a small task.",
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

    assert "'status': 'validated'" in result.result["codeact_code"]
    record, _ = StateStore(paths).get(task_ref)
    assert "executor" in record.consumers


def test_executor_explicit_empty_evidence_does_not_fallback_to_state(
    tmp_path: Path,
) -> None:
    paths = RuntimePaths(root=tmp_path)
    trace_id = "trace-budgeted-executor"
    state_store = StateStore(paths)
    task_ref = state_store.put_text(
        trace_id=trace_id,
        producer="runtime",
        text="SECRET FULL STATE TASK",
    )
    llm = CapturingLLM("print('ok')")
    context = RuntimeContext.from_paths(
        paths=paths,
        trace_id=trace_id,
        llm_client=llm,
        load_configured_llm=False,
    )

    ExecutorAgent().handle(
        AMPMessage(
            trace_id=trace_id,
            source_agent="runtime",
            target_agent="executor",
            msg_type=MsgType.INVOKE,
            action="tool.run_python",
            params={
                "task": "budgeted task",
                "evidence": "",
                "_disable_state_fallback": True,
            },
            state_refs=[task_ref],
        ),
        context,
    )

    assert llm.calls[0]["variables"] == {"input": "budgeted task"}
    assert "SECRET FULL STATE TASK" not in str(llm.calls[0]["messages"])


def test_executor_empty_evidence_still_falls_back_without_candidate_flag(
    tmp_path: Path,
) -> None:
    paths = RuntimePaths(root=tmp_path)
    trace_id = "trace-baseline-executor"
    state_store = StateStore(paths)
    task_ref = state_store.put_text(
        trace_id=trace_id,
        producer="runtime",
        text="BASELINE STATE CONTEXT",
    )
    llm = CapturingLLM("print('ok')")
    context = RuntimeContext.from_paths(
        paths=paths,
        trace_id=trace_id,
        llm_client=llm,
        load_configured_llm=False,
    )

    ExecutorAgent().handle(
        AMPMessage(
            trace_id=trace_id,
            source_agent="runtime",
            target_agent="executor",
            msg_type=MsgType.INVOKE,
            action="tool.run_python",
            params={"task": "baseline task", "evidence": ""},
            state_refs=[task_ref],
        ),
        context,
    )

    assert llm.calls[0]["variables"] == {
        "input": "baseline task\nBASELINE STATE CONTEXT"
    }


def test_summarizer_explicit_empty_code_result_does_not_fallback_to_state(
    tmp_path: Path,
) -> None:
    paths = RuntimePaths(root=tmp_path)
    trace_id = "trace-budgeted-summarizer"
    state_store = StateStore(paths)
    code_ref = state_store.put_code_result(
        trace_id=trace_id,
        producer="executor",
        result={
            "stdout": "SECRET FULL CODE RESULT",
            "stderr": "",
            "exit_code": 0,
        },
    )
    llm = CapturingLLM("summary")
    context = RuntimeContext.from_paths(
        paths=paths,
        trace_id=trace_id,
        llm_client=llm,
        load_configured_llm=False,
    )

    SummarizerAgent().handle(
        AMPMessage(
            trace_id=trace_id,
            source_agent="executor",
            target_agent="summarizer",
            msg_type=MsgType.INVOKE,
            action="summary.create",
            params={
                "task": "budgeted task",
                "evidence_digest": "",
                "code_result": "",
                "_disable_state_fallback": True,
            },
            state_refs=[code_ref],
        ),
        context,
    )

    assert llm.calls[0]["variables"] == {"input": "Task:\nbudgeted task"}
    assert "SECRET FULL CODE RESULT" not in str(llm.calls[0]["messages"])


def test_summarizer_empty_code_result_still_falls_back_without_candidate_flag(
    tmp_path: Path,
) -> None:
    paths = RuntimePaths(root=tmp_path)
    trace_id = "trace-baseline-summarizer"
    state_store = StateStore(paths)
    code_ref = state_store.put_code_result(
        trace_id=trace_id,
        producer="executor",
        result={"stdout": "BASELINE CODE RESULT", "stderr": "", "exit_code": 0},
    )
    llm = CapturingLLM("summary")
    context = RuntimeContext.from_paths(
        paths=paths,
        trace_id=trace_id,
        llm_client=llm,
        load_configured_llm=False,
    )

    SummarizerAgent().handle(
        AMPMessage(
            trace_id=trace_id,
            source_agent="executor",
            target_agent="summarizer",
            msg_type=MsgType.INVOKE,
            action="summary.create",
            params={"task": "baseline task", "evidence_digest": "", "code_result": ""},
            state_refs=[code_ref],
        ),
        context,
    )

    variables = str(llm.calls[0]["variables"])
    assert "BASELINE CODE RESULT" in variables


def test_executor_rejects_invalid_llm_code_before_codeact(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    trace_id = "trace-invalid-executor-code"
    state_store = StateStore(paths)
    task_ref = state_store.put_text(
        trace_id=trace_id,
        producer="runtime",
        text="Write a DP problem and create a solution file.",
    )
    context = RuntimeContext.from_paths(
        paths=paths,
        trace_id=trace_id,
        llm_client=InvalidExecutorLLM(),
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

    assert 'solution_md = """' not in result.result["codeact_code"]
    assert "'status': 'validated'" in result.result["codeact_code"]
    assert result.result["llm_generated_code"] is False


def test_protocol_mode_hands_off_long_task_by_compact_params(tmp_path: Path) -> None:
    task = tmp_path / "task.txt"
    task_text = "Write Python code that validates a small task. " * 40
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
        if "task" in params:
            assert len(params["task"]) <= 500
        if "query" in params:
            assert len(params["query"]) <= 300
        assert item["state_refs_in"]
