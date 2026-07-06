from pathlib import Path
from typing import Any

from agentmesh.errors import SandboxTimeoutError
from agentmesh.eval.llm_experiment import (
    LLMExperimentProfile,
    ProtocolOptimizationProfile,
)
from agentmesh.llm.client import ChatMessage, LLMClient
from agentmesh.memory.hybrid_store import HybridMemoryStore
from agentmesh.memory.schema import MemoryUnit
from agentmesh.modes.protocol_mode import run_protocol_mode
from agentmesh.runtime.registry import RuntimeContext
from agentmesh.sandbox.runner import SandboxResult, SandboxRunner
from agentmesh.state.embedding import create_embedding_encoder
from agentmesh.state.schema import StateType
from agentmesh.state.store import StateStore
from agentmesh.storage.jsonl import read_jsonl
from agentmesh.storage.paths import RuntimePaths


class AgentEchoLLMClient(LLMClient):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        agent_name: str,
        messages: list[ChatMessage],
        variables: dict[str, object] | None = None,
    ) -> str:
        self.calls.append(
            {"agent_name": agent_name, "messages": messages, "variables": variables or {}}
        )
        if agent_name == "executor":
            return "print('executor model response')"
        return f"{agent_name} model response"


def test_protocol_mode_uses_llm_outputs_as_agent_state(tmp_path: Path) -> None:
    task = tmp_path / "task.txt"
    task.write_text(
        "Design and validate real agent state transfer across several agents.",
        encoding="utf-8",
    )
    paths = RuntimePaths(root=tmp_path)
    llm_client = AgentEchoLLMClient()

    result = run_protocol_mode(task_path=task, paths=paths, llm_client=llm_client)

    assert result.answer == "summarizer model response"
    assert {call["agent_name"] for call in llm_client.calls} == {
        "planner",
        "retriever",
        "executor",
        "summarizer",
        "memory_tagger",
    }

    state_store = StateStore(paths)
    records = state_store.list_by_trace(result.trace_id)
    planner_summaries = [
        state_store.get(record.ref)[1]
        for record in records
        if record.producer == "planner" and record.state_type == StateType.SUMMARY
    ]
    summarizer_summaries = [
        state_store.get(record.ref)[1]
        for record in records
        if record.producer == "summarizer" and record.state_type == StateType.SUMMARY
    ]
    executor_results = [
        state_store.get(record.ref)[1]
        for record in records
        if record.producer == "executor" and record.state_type == StateType.CODE_RESULT
    ]

    assert any("planner model response" in summary for summary in planner_summaries)
    assert summarizer_summaries == ["summarizer model response"]
    assert executor_results[0]["executor_result"]["llm_generated_code"] is True
    assert executor_results[0]["codeact"]["stdout"].strip() == "executor model response"
    assert executor_results[0]["codeact"]["exit_code"] == 0
    calls = {str(call["agent_name"]): call for call in llm_client.calls}
    executor_system = calls["executor"]["messages"][0].content
    summarizer_system = calls["summarizer"]["messages"][0].content
    assert "Do not write the final user answer." in executor_system
    assert "Answer the user directly." in summarizer_system
    assert "required facts" in summarizer_system


def test_protocol_mode_continues_when_sandbox_times_out(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = tmp_path / "task.txt"
    task.write_text("写一个桶排序", encoding="utf-8")
    paths = RuntimePaths(root=tmp_path)

    def timeout_run_python(self: SandboxRunner, code: str):
        del self, code
        raise SandboxTimeoutError("Sandbox execution timed out")

    monkeypatch.setattr(SandboxRunner, "run_python", timeout_run_python)

    result = run_protocol_mode(
        task_path=task,
        paths=paths,
        load_configured_llm=False,
    )

    state_store = StateStore(paths)
    executor_results = [
        state_store.get(record.ref)[1]
        for record in state_store.list_by_trace(result.trace_id)
        if record.producer == "executor" and record.state_type == StateType.CODE_RESULT
    ]

    assert "sandbox exit 124" in result.answer
    assert executor_results[0]["codeact"]["exit_code"] == 124
    assert executor_results[0]["codeact"]["stderr"] == "Sandbox execution timed out"


def test_candidate_profile_applies_auditable_role_context_budgets(
    tmp_path: Path,
) -> None:
    task_text = "Validate code and Return sum 77 and even count 5."
    task = tmp_path / "task.txt"
    task.write_text(task_text, encoding="utf-8")
    paths = RuntimePaths(root=tmp_path)
    llm_client = AgentEchoLLMClient()
    profile = LLMExperimentProfile(
        profile_id="candidate",
        artifact_label="candidate",
        optimization_enabled=True,
        protocol=ProtocolOptimizationProfile(
            planner_max_chars=5,
            retriever_max_chars=5,
            executor_max_chars=5,
            summarizer_max_chars=5,
        ),
    )

    result = run_protocol_mode(
        task_path=task,
        paths=paths,
        llm_client=llm_client,
        experiment_profile=profile,
    )

    assert [audit["role"] for audit in result.metrics.context_audits] == [
        "planner",
        "retriever",
        "executor",
        "summarizer",
    ]
    assert all(
        int(audit["retained_chars"]) <= int(audit["original_chars"])
        for audit in result.metrics.context_audits
    )
    assert result.metrics.context_original_chars >= result.metrics.context_retained_chars
    assert result.metrics.context_safe_fallback_count >= 1
    retriever_audit = next(
        audit
        for audit in result.metrics.context_audits
        if audit["role"] == "retriever"
    )
    assert "planner_intent" in retriever_audit["included_parts"]
    assert retriever_audit["safe_fallback"] is True
    role_calls = {
        str(call["agent_name"]): call
        for call in llm_client.calls
        if call["agent_name"] != "memory_tagger"
    }
    for role in ["planner", "retriever", "executor", "summarizer"]:
        rendered = f"{role_calls[role]['messages']} {role_calls[role]['variables']}"
        assert task_text in rendered
    params_by_action = {
        row["input"]["action"]: row["input"]["params"]
        for row in read_jsonl(paths.protocol_agent_io)
    }
    executor_input = str(role_calls["executor"]["variables"]["input"])
    summarizer_input = str(role_calls["summarizer"]["variables"]["input"])
    assert params_by_action["tool.run_python"]["_disable_state_fallback"] is True
    assert params_by_action["summary.create"]["_disable_state_fallback"] is True
    assert " ".join(executor_input.split()) == params_by_action["tool.run_python"]["task"]
    assert " ".join(summarizer_input.split()) == (
        f"Task: {params_by_action['summary.create']['task']}"
    )
    assert executor_input.count(task_text) == 1
    assert summarizer_input.count(task_text) == 1
    assert summarizer_input.count("sandbox exit 0") == 1


def test_disabled_profile_preserves_no_profile_protocol_params(tmp_path: Path) -> None:
    task = tmp_path / "task.txt"
    task.write_text("Analyze protocol context behavior.", encoding="utf-8")
    no_profile_paths = RuntimePaths(root=tmp_path / "no-profile")
    baseline_paths = RuntimePaths(root=tmp_path / "baseline")

    no_profile = run_protocol_mode(
        task_path=task,
        paths=no_profile_paths,
        load_configured_llm=False,
    )
    baseline = run_protocol_mode(
        task_path=task,
        paths=baseline_paths,
        load_configured_llm=False,
        experiment_profile=LLMExperimentProfile(
            profile_id="baseline",
            artifact_label="baseline",
            optimization_enabled=False,
        ),
    )

    no_profile_params = [
        row["input"]["params"] for row in read_jsonl(no_profile_paths.protocol_agent_io)
    ]
    baseline_params = [
        row["input"]["params"] for row in read_jsonl(baseline_paths.protocol_agent_io)
    ]
    assert baseline_params == no_profile_params
    assert all("_disable_state_fallback" not in params for params in baseline_params)
    assert all(
        "deterministic_retrieval_evidence" not in params
        for params in baseline_params
    )
    assert no_profile.metrics.context_audits == []
    assert baseline.metrics.context_audits == []


def test_candidate_direct_retrieval_keeps_route_without_retriever_llm(
    tmp_path: Path,
) -> None:
    task = tmp_path / "task.txt"
    task.write_text("Remember my preferred editor.", encoding="utf-8")
    client = AgentEchoLLMClient()

    result = run_protocol_mode(
        task_path=task,
        paths=RuntimePaths(root=tmp_path / "candidate-v2"),
        llm_client=client,
        experiment_profile=LLMExperimentProfile(
            profile_id="candidate-v2",
            artifact_label="candidate-v2",
            route_policy_version="quality_safe_v1",
            optimization_enabled=True,
            protocol=ProtocolOptimizationProfile(
                deterministic_retrieval_evidence=True,
            ),
        ),
    )

    called_agents = [str(call["agent_name"]) for call in client.calls]
    assert "retriever" in result.metrics.dynamic_route
    assert result.metrics.memory_query_count == 1
    assert "retriever" not in called_agents
    assert "planner" in called_agents
    assert "summarizer" in called_agents


def test_candidate_direct_retrieval_preserves_provenance_downstream(
    tmp_path: Path,
) -> None:
    task = tmp_path / "task.txt"
    task.write_text("Remember my preferred editor.", encoding="utf-8")
    paths = RuntimePaths(root=tmp_path / "candidate-v2-provenance")
    state_store = StateStore(paths)
    context = RuntimeContext.from_paths(
        paths=paths,
        trace_id="seed-trace",
        load_configured_llm=False,
    )
    encoder = create_embedding_encoder(context.config.embedding)
    source_evidence_ref = state_store.put_evidence(
        trace_id="trace-prior",
        producer="summarizer",
        evidence=[{"title": "preference", "snippet": "Editor is Vim."}],
    )
    memory_store = HybridMemoryStore(
        paths=paths,
        state_store=state_store,
        encoder=encoder,
    )
    memory_store.put(
        MemoryUnit(
            source_agent="summarizer",
            task_topic="preferred editor",
            summary="The preferred editor is Vim.",
            tags=["preference"],
            evidence_refs=[source_evidence_ref],
            state_refs=[],
            embedding_vector=encoder.encode("preferred editor Vim"),
            confidence=0.9,
            validity_score=1.0,
            provenance_trace_id="trace-prior",
        )
    )

    result = run_protocol_mode(
        task_path=task,
        paths=paths,
        llm_client=AgentEchoLLMClient(),
        experiment_profile=LLMExperimentProfile(
            profile_id="candidate-v2-provenance",
            artifact_label="candidate-v2-provenance",
            route_policy_version="quality_safe_v1",
            optimization_enabled=True,
            protocol=ProtocolOptimizationProfile(
                deterministic_retrieval_evidence=True,
            ),
        ),
    )

    summarizer_rows = [
        row
        for row in read_jsonl(paths.protocol_agent_io)
        if row["input"]["action"] == "summary.create"
    ]
    assert '"source_agent":"summarizer"' in summarizer_rows[0]["input"]["params"]["task"]
    assert '"provenance_trace_id":"trace-prior"' in (
        summarizer_rows[0]["input"]["params"]["task"]
    )
    assert source_evidence_ref in summarizer_rows[0]["input"]["params"]["task"]

    evidence_payloads = [
        StateStore(paths).get(record.ref)[1]
        for record in StateStore(paths).list_by_trace(result.trace_id)
        if record.state_type == StateType.EVIDENCE
    ]
    memory_evidence = next(
        item
        for payload in evidence_payloads
        for item in payload
        if item.get("memory_id")
    )
    assert memory_evidence["source_agent"] == "summarizer"
    assert memory_evidence["provenance_trace_id"] == "trace-prior"
    assert memory_evidence["evidence_refs"] == [source_evidence_ref]


def test_candidate_budgets_planner_and_evidence_refinement_contexts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_text = "Validate code benchmark context."
    task = tmp_path / "task.txt"
    task.write_text(task_text, encoding="utf-8")
    paths = RuntimePaths(root=tmp_path)
    max_chars = len(task_text) + 6

    def feedback_result(self: SandboxRunner, code: str) -> SandboxResult:
        del self, code
        return SandboxResult(
            stdout='{"evidence_gaps":["missing baseline details"]}',
            stderr="",
            exit_code=0,
            latency_ms=1,
            backend="test",
        )

    monkeypatch.setattr(SandboxRunner, "run_python", feedback_result)
    result = run_protocol_mode(
        task_path=task,
        paths=paths,
        llm_client=AgentEchoLLMClient(),
        experiment_profile=LLMExperimentProfile(
            profile_id="candidate-refine",
            artifact_label="candidate-refine",
            optimization_enabled=True,
            protocol=ProtocolOptimizationProfile(
                planner_max_chars=max_chars,
                retriever_max_chars=max_chars,
                executor_max_chars=max_chars,
                summarizer_max_chars=max_chars,
            ),
        ),
    )

    audits = {str(audit["role"]): audit for audit in result.metrics.context_audits}
    assert {"planner_refine", "retriever_refine"} <= audits.keys()
    for role in ["planner_refine", "retriever_refine"]:
        assert int(audits[role]["retained_chars"]) <= int(audits[role]["original_chars"])
        assert int(audits[role]["retained_chars"]) <= max_chars
        assert "evidence_gaps" in audits[role]["partial_parts"]
    refine_rows = {
        row["input"]["action"]: row["input"]["params"]
        for row in read_jsonl(paths.protocol_agent_io)
        if row["input"]["action"] in {"plan.refine", "evidence.refine"}
    }
    assert task_text in str(refine_rows["plan.refine"]["task"])
    assert task_text in str(refine_rows["evidence.refine"]["query"])
    assert refine_rows["plan.refine"]["tool_feedback"] == {}
    assert refine_rows["evidence.refine"]["tool_feedback"] == {}
