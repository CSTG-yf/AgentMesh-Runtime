from pathlib import Path
from typing import Any

from agentmesh.chat.session import ChatTurn, run_chat_turn
from agentmesh.llm.client import ChatMessage, LLMClient
from agentmesh.runtime.registry import RuntimeContext
from agentmesh.storage.paths import RuntimePaths


class FakeLLMClient(LLMClient):
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
        return "interactive answer"


def test_chat_turn_uses_configurable_initial_prompt(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "interactive.md").write_text(
        "You are AgentMesh. User said: {user_input}",
        encoding="utf-8",
    )
    client = FakeLLMClient()
    context = RuntimeContext.from_paths(
        paths=RuntimePaths(root=tmp_path),
        trace_id="trace-chat",
        llm_client=client,
    )

    result = run_chat_turn("hello", context=context, history=[])

    assert isinstance(result, ChatTurn)
    assert result.assistant == "interactive answer"
    assert client.calls[0]["agent_name"] == "interactive"
    messages = client.calls[0]["messages"]
    assert messages[0].role == "system"
    assert messages[0].content == "You are AgentMesh. User said: hello"
    assert messages[1].role == "user"
    assert messages[1].content == "hello"
