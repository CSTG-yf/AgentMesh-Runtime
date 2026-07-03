from pathlib import Path
from typing import Any

from agentmesh.errors import SandboxTimeoutError
from agentmesh.eval.llm_experiment import (
    LLMExperimentProfile,
    ProtocolOptimizationProfile,
)
from agentmesh.llm.client import ChatMessage, LLMClient
from agentmesh.modes.protocol_mode import run_protocol_mode
from agentmesh.sandbox.runner import SandboxRunner
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
    assert result.metrics.context_safe_fallback_count == 4
    role_calls = {
        str(call["agent_name"]): call
        for call in llm_client.calls
        if call["agent_name"] != "memory_tagger"
    }
    for role in ["planner", "retriever", "executor", "summarizer"]:
        rendered = f"{role_calls[role]['messages']} {role_calls[role]['variables']}"
        assert task_text in rendered


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
    assert no_profile.metrics.context_audits == []
    assert baseline.metrics.context_audits == []
