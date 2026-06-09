from pathlib import Path

from agentmesh.eval.benchmark import run_benchmark
from agentmesh.eval.report import generate_report
from agentmesh.llm.client import ChatMessage, LLMClient
from agentmesh.modes.protocol_mode import run_protocol_mode
from agentmesh.modes.text_mode import run_text_mode
from agentmesh.storage.jsonl import read_jsonl
from agentmesh.storage.paths import RuntimePaths


class TextModeFailingSummarizerLLM(LLMClient):
    def complete(
        self,
        *,
        agent_name: str,
        messages: list[ChatMessage],
        variables: dict[str, object] | None = None,
    ) -> str:
        _ = messages, variables
        if agent_name == "summarizer":
            raise RuntimeError("summarizer timed out")
        if agent_name == "executor":
            return (
                "```python\n"
                "data = [3, 1, 2]\n"
                "sorted_data = sorted(data)\n"
                "print(sorted_data)\n"
                "```"
            )
        return f"{agent_name} output"


def test_text_and_protocol_modes_produce_metrics_and_artifacts(tmp_path: Path) -> None:
    task = tmp_path / "task.txt"
    task.write_text("Create a concise architecture note about AgentMesh Runtime.", encoding="utf-8")
    paths = RuntimePaths(root=tmp_path)

    text_result = run_text_mode(task_path=task, paths=paths)
    protocol_result = run_protocol_mode(task_path=task, paths=paths)

    assert text_result.mode == "text"
    assert text_result.metrics.text_chars > 0
    assert text_result.metrics.communication_model == "plain_text"
    assert text_result.metrics.wire_bytes > 0
    assert text_result.metrics.text_wire_bytes == text_result.metrics.wire_bytes
    assert text_result.metrics.structured_handoff_bytes == 0
    assert text_result.metrics.session_dictionary_bytes == 0
    assert text_result.metrics.typed_envelope_bytes == 0
    assert text_result.metrics.typed_payload_bytes == 0
    assert text_result.metrics.structured_message_bytes == 0
    assert text_result.metrics.compact_structured_message_bytes == 0
    assert text_result.metrics.protocol_bytes == 0
    assert text_result.metrics.state_transfer_count == 0
    assert text_result.metrics.state_transfer_bytes == 0
    assert text_result.metrics.rust_core_enabled is False
    assert text_result.metrics.sandbox_backend == ""
    assert text_result.metrics.transport_type == ""
    assert text_result.metrics.transport_send_count == 0
    assert text_result.metrics.state_shm_transfer_count == 0
    assert text_result.metrics.memory_query_count == 0
    assert text_result.metrics.memory_hit_count == 0
    text_messages = read_jsonl(paths.text_messages)
    text_agent_io = read_jsonl(paths.text_agent_io)
    assert [item["source_agent"] for item in text_messages] == [
        "planner",
        "retriever",
        "executor",
        "summarizer",
    ]
    assert [item["agent"] for item in text_agent_io] == [
        "planner",
        "retriever",
        "executor",
        "summarizer",
    ]
    assert text_messages[0]["content"] == task.read_text(encoding="utf-8")
    assert text_agent_io[0]["input"]["content"] == task.read_text(encoding="utf-8")
    assert text_agent_io[0]["output"]["content"]
    assert text_agent_io[0]["transport"]["communication_model"] == "plain_text_full_context"
    assert all("state://" not in item["input"]["content"] for item in text_agent_io)
    assert all("state://" not in item["content"] for item in text_messages)
    assert all("typed_envelope" not in item["content"] for item in text_messages)
    assert protocol_result.mode == "protocol"
    assert protocol_result.metrics.communication_model == "structured_state_ref"
    assert protocol_result.metrics.wire_bytes > 0
    assert protocol_result.metrics.session_dictionary_bytes > 0
    assert protocol_result.metrics.typed_envelope_bytes > 0
    assert protocol_result.metrics.typed_payload_bytes > 0
    assert protocol_result.metrics.wire_bytes == protocol_result.metrics.typed_envelope_bytes
    assert protocol_result.metrics.structured_message_bytes > 0
    assert protocol_result.metrics.compact_structured_message_bytes > 0
    assert (
        protocol_result.metrics.wire_bytes
        < protocol_result.metrics.compact_structured_message_bytes
    )
    assert protocol_result.metrics.state_transfer_count >= 4
    assert protocol_result.metrics.transport_type == "inproc"
    assert protocol_result.metrics.transport_send_count == 3
    assert protocol_result.metrics.transport_bytes > 0
    assert protocol_result.metrics.transport_avg_latency_ms >= 0
    assert protocol_result.metrics.state_shm_transfer_count == 0
    assert set(protocol_result.metrics.stage_latency_ms) == {
        "setup",
        "state_task_embedding",
        "planner",
        "memory_search",
        "retriever",
        "summarizer",
        "memory_write",
        "artifact_write",
    }
    assert protocol_result.metrics.dynamic_route == ["planner", "retriever", "summarizer"]
    assert "executor" in protocol_result.metrics.skipped_agents
    assert paths.text_messages.exists()
    protocol_agent_io = read_jsonl(paths.protocol_agent_io)
    assert [item["agent"] for item in protocol_agent_io] == [
        "planner",
        "retriever",
        "summarizer",
    ]
    assert protocol_agent_io[0]["input"]["action"] == "plan.create"
    assert protocol_agent_io[0]["output"]["msg_type"] == "RESULT"
    assert protocol_agent_io[0]["transport"]["communication_model"] == "amp_state_ref"
    assert protocol_agent_io[0]["state_refs_in"]
    assert paths.protocol_states.exists()


def test_protocol_mode_can_use_shared_memory_state_payloads(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "AGENTMESH_STATE_PAYLOAD_BACKEND=shm",
                "AGENTMESH_STATE_SHM_THRESHOLD_BYTES=8",
            ]
        ),
        encoding="utf-8",
    )
    task = tmp_path / "task.txt"
    task.write_text("Analyze shared memory state transfer in protocol mode.", encoding="utf-8")
    paths = RuntimePaths(root=tmp_path)

    result = run_protocol_mode(task_path=task, paths=paths)

    assert result.metrics.state_shm_transfer_count > 0
    assert result.metrics.state_shm_transfer_bytes > 0


def test_benchmark_and_report_generate_expected_outputs(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "examples" / "tasks"
    tasks_dir.mkdir(parents=True)
    for name in ["A1.txt", "B1.txt"]:
        (tasks_dir / name).write_text(
            f"Task {name}: explain protocol memory reuse.",
            encoding="utf-8",
        )
    suite = tmp_path / "examples" / "benchmarks" / "continuous_tasks.yaml"
    suite.parent.mkdir(parents=True)
    suite.write_text(
        """
name: tiny_suite
repeat: 5
tasks:
  - id: A1
    group: A
    topic: protocol memory
    input_file: examples/tasks/A1.txt
    tags: [protocol, memory]
  - id: B1
    group: B
    topic: state passing
    input_file: examples/tasks/B1.txt
    tags: [state, protocol]
""".strip(),
        encoding="utf-8",
    )
    paths = RuntimePaths(root=tmp_path)

    summary = run_benchmark(suite_path=suite, paths=paths)
    report_path = generate_report(paths=paths)

    assert summary.total_runs == 10
    assert summary.token_estimator == "mixed_cjk"
    assert summary.text_wire_bytes > 0
    assert summary.protocol_wire_bytes > 0
    assert summary.protocol_session_dictionary_bytes > 0
    assert summary.protocol_typed_envelope_bytes > 0
    assert summary.protocol_typed_payload_bytes > 0
    assert summary.protocol_compact_message_bytes > 0
    assert summary.protocol_json_wire_bytes > 0
    assert summary.transport_send_count > 0
    assert summary.transport_bytes > 0
    assert summary.transport_avg_latency_ms >= 0
    assert summary.state_shm_transfer_count == 0
    assert summary.wire_bytes_reduction_rate != 0
    assert 0.0 <= summary.memory_hit_rate <= 1.0
    assert summary.memory_reused_unit_count >= 0
    assert paths.benchmark_summary.exists()
    assert paths.benchmark_detail.exists()
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "TokenEstimator" in report_text
    assert "TokenSavingRate" in report_text
    assert "WireBytesReductionRate" in report_text
    assert "FeedbackRoundCount" in report_text
    assert "TransportSendCount" in report_text
    assert "StateShmTransferCount" in report_text
    assert "LatestDynamicRoute" in report_text
    assert "Text Agent I/O Sample" in report_text
    assert "Protocol Agent I/O Sample" in report_text
    assert len(read_jsonl(paths.benchmark_detail)) == 20


def test_text_mode_summarizer_fallback_preserves_executor_output(tmp_path: Path) -> None:
    task = tmp_path / "sort.txt"
    task.write_text("Write quicksort and output the sorted result.", encoding="utf-8")
    paths = RuntimePaths(root=tmp_path)

    result = run_text_mode(
        task_path=task,
        paths=paths,
        llm_client=TextModeFailingSummarizerLLM(),
    )

    assert "summarizer LLM unavailable" in result.answer
    assert "print(sorted_data)" in result.answer
    assert result.answer != "processed task with full text context."
    agent_io = read_jsonl(paths.text_agent_io)
    assert agent_io[-1]["agent"] == "summarizer"
    assert "print(sorted_data)" in agent_io[-1]["output"]["content"]
