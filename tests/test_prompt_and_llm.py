from pathlib import Path
from typing import Any

from agentmesh.config import AgentMeshConfig
from agentmesh.llm.client import ChatMessage, LLMClient, OpenAICompatibleClient
from agentmesh.prompts.store import PromptTemplateStore
from agentmesh.runtime.registry import RuntimeContext
from agentmesh.storage.paths import RuntimePaths


class FakeLLMClient(LLMClient):
    def __init__(self, response: str = "model response") -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response

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
        return self.response


def test_prompt_template_store_loads_custom_agent_prompt(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "planner.md").write_text(
        "You are {agent_name}. Task: {task}",
        encoding="utf-8",
    )

    store = PromptTemplateStore(prompt_dir=prompt_dir)

    assert store.render("planner", {"agent_name": "planner", "task": "design runtime"}) == (
        "You are planner. Task: design runtime"
    )


def test_openai_compatible_client_builds_masked_chat_request() -> None:
    captured: dict[str, object] = {}

    def transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        captured.update(
            {
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {"choices": [{"message": {"content": "hello from model"}}]}

    config = AgentMeshConfig.from_mapping(
        {
            "AGENTMESH_LLM_BASE_URL": "https://llm.example/v1",
            "AGENTMESH_LLM_API_KEY": "sk-test",
            "AGENTMESH_LLM_MODEL": "qwen-test",
        }
    )
    client = OpenAICompatibleClient(config=config, transport=transport)

    result = client.complete(
        agent_name="interactive",
        messages=[ChatMessage(role="user", content="hello")],
    )

    assert result == "hello from model"
    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["headers"] == {
        "Authorization": "Bearer sk-test",
        "Content-Type": "application/json",
    }
    assert captured["payload"] == {
        "model": "qwen-test",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_runtime_context_can_build_llm_client_and_prompt_store(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "AGENTMESH_LLM_BASE_URL=https://llm.example/v1",
                "AGENTMESH_LLM_API_KEY=sk-test",
                "AGENTMESH_LLM_MODEL=qwen-test",
            ]
        ),
        encoding="utf-8",
    )
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "interactive.md").write_text("System: {user_input}", encoding="utf-8")

    context = RuntimeContext.from_paths(
        paths=RuntimePaths(root=tmp_path),
        trace_id="trace-1",
        llm_client=FakeLLMClient(),
    )

    assert context.config.llm.configured
    assert context.prompts.render("interactive", {"user_input": "hi"}) == "System: hi"
    assert context.llm_client is not None
