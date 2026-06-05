from pathlib import Path
from typing import Any

from agentmesh.llm.client import ChatMessage, LLMClient
from agentmesh.modes.protocol_mode import run_protocol_mode
from agentmesh.state.schema import StateType
from agentmesh.state.store import StateStore
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
