from pathlib import Path
from typing import Any

from agentmesh.config import AgentMeshConfig
from agentmesh.llm.client import (
    ChatMessage,
    LLMClient,
    OpenAICompatibleClient,
)
from agentmesh.modes.text_mode import _text_mode_system_prompt, run_text_mode
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


class EchoAgentLLMClient(LLMClient):
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
        return f"{agent_name} model response"


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


def test_default_agent_system_prompts_define_roles_and_boundaries(tmp_path: Path) -> None:
    store = PromptTemplateStore.from_project_root(tmp_path)

    planner = store.render("planner", {"task": "compare text and protocol agents"})
    retriever = store.render("retriever", {"query": "memory reuse"})
    executor = store.render("executor", {"input": "validate evidence"})
    summarizer = store.render("summarizer", {"input": "state://summary/abc"})

    assert "user intent recognition" in planner
    assert "classify the task intent" in planner
    assert "structured plan" in planner
    assert "reusable memory" in retriever
    assert "evidence" in retriever
    assert "Return only Python code" in executor
    assert "sandbox" in executor
    assert "state references" in summarizer
    assert "memory-worthy" in summarizer


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


def test_text_mode_can_use_llm_with_full_text_handoff(tmp_path: Path) -> None:
    task = tmp_path / "task.txt"
    task.write_text("Write quicksort in Python.", encoding="utf-8")
    llm = EchoAgentLLMClient()

    result = run_text_mode(
        task_path=task,
        paths=RuntimePaths(root=tmp_path),
        llm_client=llm,
        load_configured_llm=False,
    )

    assert result.answer == "summarizer model response"
    assert [call["agent_name"] for call in llm.calls] == [
        "planner",
        "retriever",
        "executor",
        "summarizer",
    ]
    retriever_user = llm.calls[1]["messages"][1].content
    summarizer_user = llm.calls[3]["messages"][1].content
    assert "Write quicksort in Python." in retriever_user
    assert "[planner] planner model response" in retriever_user
    assert "[retriever] retriever model response" in summarizer_user
    assert "[executor] executor model response" in summarizer_user
    assert result.metrics.communication_model == "plain_text"
    assert result.metrics.state_transfer_count == 0
    assert result.metrics.transport_send_count == 0


def test_text_mode_preserves_multiline_summarizer_answer(tmp_path: Path) -> None:
    task = tmp_path / "task.txt"
    task.write_text("Write quicksort in Python.", encoding="utf-8")

    class MultilineSummarizerLLM(EchoAgentLLMClient):
        def complete(
            self,
            *,
            agent_name: str,
            messages: list[ChatMessage],
            variables: dict[str, object] | None = None,
        ) -> str:
            self.calls.append(
                {
                    "agent_name": agent_name,
                    "messages": messages,
                    "variables": variables or {},
                }
            )
            if agent_name == "summarizer":
                return "line one\nline two\nline three"
            return f"{agent_name} model response"

    result = run_text_mode(
        task_path=task,
        paths=RuntimePaths(root=tmp_path),
        llm_client=MultilineSummarizerLLM(),
        load_configured_llm=False,
    )

    assert result.answer == "line one\nline two\nline three"


def test_text_mode_executor_prompt_keeps_code_role_boundary(tmp_path: Path) -> None:
    context = RuntimeContext.from_paths(
        paths=RuntimePaths(root=tmp_path),
        trace_id="trace-text-prompt",
        load_configured_llm=False,
    )

    executor_prompt = _text_mode_system_prompt(
        "executor",
        context,
        "Write quicksort in Python.",
    )
    planner_prompt = _text_mode_system_prompt(
        "planner",
        context,
        "Write quicksort in Python.",
    )

    assert "ExecutorAgent-specific rule" in executor_prompt
    assert "return only Python code" in executor_prompt
    assert "ExecutorAgent-specific rule" not in planner_prompt


def test_text_mode_uses_dedicated_llm_timeout_from_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_timeouts: list[float] = []

    def fake_transport(url, headers, payload, timeout_seconds):
        del url, headers, payload
        captured_timeouts.append(timeout_seconds)
        return {"choices": [{"message": {"content": "model text"}}]}

    monkeypatch.setattr(
        "agentmesh.llm.client._default_transport",
        fake_transport,
    )
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "AGENTMESH_LLM_BASE_URL=https://llm.example/v1",
                "AGENTMESH_LLM_API_KEY=sk-test",
                "AGENTMESH_LLM_MODEL=qwen-test",
                "AGENTMESH_LLM_TIMEOUT_SECONDS=15",
                "AGENTMESH_TEXT_LLM_TIMEOUT_SECONDS=90",
            ]
        ),
        encoding="utf-8",
    )
    task = tmp_path / "task.txt"
    task.write_text("Write quicksort in Python.", encoding="utf-8")

    run_text_mode(task_path=task, paths=RuntimePaths(root=tmp_path))

    assert captured_timeouts == [90.0, 90.0, 90.0, 90.0]
