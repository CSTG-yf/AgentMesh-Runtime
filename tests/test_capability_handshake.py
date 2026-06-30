from pathlib import Path

from agentmesh.agents import ExecutorAgent, PlannerAgent, RetrieverAgent, SummarizerAgent
from agentmesh.modes.protocol_mode import run_protocol_mode
from agentmesh.protocol.enums import MsgType
from agentmesh.runtime.registry import AgentRegistry
from agentmesh.storage.jsonl import read_jsonl
from agentmesh.storage.paths import RuntimePaths


def test_agents_register_and_advertise_capabilities() -> None:
    registry = AgentRegistry()
    for agent in [PlannerAgent(), RetrieverAgent(), ExecutorAgent(), SummarizerAgent()]:
        registry.register(agent)

    assert registry.require_capability("plan.create").name == "planner"
    assert registry.require_capability("memory.semantic_search").name == "retriever"
    assert registry.require_capability("tool.run_python").name == "executor"
    assert registry.require_capability("memory.put").name == "summarizer"

    hello = registry.get("planner").hello(trace_id="trace-hello")
    advertise = registry.get("planner").advertise_capabilities(trace_id="trace-hello")

    assert hello.msg_type == MsgType.HELLO
    assert advertise.msg_type == MsgType.CAPABILITY_ADVERTISE
    assert "plan.create" in advertise.capability


def test_protocol_mode_writes_handshake_messages(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "AGENTMESH_PROTOCOL_SKIP_HANDSHAKE_FOR_INPROC=false\n",
        encoding="utf-8",
    )
    task = tmp_path / "task.txt"
    task.write_text("Analyze state passing and shared memory.", encoding="utf-8")
    paths = RuntimePaths(root=tmp_path)

    result = run_protocol_mode(task_path=task, paths=paths)

    messages = read_jsonl(paths.protocol_messages)
    message_types = [item["msg_type"] for item in messages]

    assert result.mode == "protocol"
    assert "HELLO" in message_types
    assert "CAPABILITY_ADVERTISE" in message_types
    assert "PROTOCOL_MAP" in message_types
    assert "STATE_REF" in message_types
    assert "MEMORY_PUT" in message_types
    assert any(item["source_agent"] == "summarizer" for item in messages)
