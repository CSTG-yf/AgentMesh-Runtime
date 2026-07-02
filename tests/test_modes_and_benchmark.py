from pathlib import Path

import orjson

from agentmesh.cli import _benchmark_suite_path
from agentmesh.eval.benchmark import run_benchmark
from agentmesh.eval.metrics import ModeRunResult, RunMetrics
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
                "status = 'validated'\n"
                "print(status)\n"
                "```"
            )
        return f"{agent_name} output"


def test_benchmark_artifacts_are_isolated_by_track(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)

    deterministic = paths.benchmark_suite_summary("suite", track="deterministic")
    llm = paths.benchmark_suite_summary("suite", track="llm")

    assert deterministic != llm
    assert deterministic.as_posix().endswith(
        "benchmarks/suite/deterministic/benchmark_summary.csv"
    )
    assert llm.as_posix().endswith("benchmarks/suite/llm/benchmark_summary.csv")
    assert paths.benchmark_suite_manifest("suite", track="llm").name == "manifest.json"


def test_protocol_mode_runs_ten_continuous_tasks_without_cross_trace_state(
    tmp_path: Path,
) -> None:
    paths = RuntimePaths(root=tmp_path)
    trace_ids: set[str] = set()

    for index in range(10):
        task = tmp_path / f"task-{index}.txt"
        task.write_text(
            f"Analyze protocol state transfer round {index}.",
            encoding="utf-8",
        )
        result = run_protocol_mode(
            task_path=task,
            paths=paths,
            load_configured_llm=False,
        )
        assert result.answer
        assert result.trace_id not in trace_ids
        trace_ids.add(result.trace_id)

    assert len(trace_ids) == 10
    message_trace_ids = {
        str(item["trace_id"])
        for item in read_jsonl(paths.protocol_messages)
    }
    assert trace_ids <= message_trace_ids


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
        "summarizer",
    ]
    assert [item["agent"] for item in text_agent_io] == [
        "planner",
        "retriever",
        "summarizer",
    ]
    assert text_messages[0]["content"] == task.read_text(encoding="utf-8")
    assert text_agent_io[0]["input"]["content"] == task.read_text(encoding="utf-8")
    assert text_agent_io[0]["output"]["content"]
    assert text_agent_io[0]["transport"]["communication_model"] == "plain_text_full_context"
    assert all("state://" not in item["input"]["content"] for item in text_agent_io)
    assert all("state://" not in item["content"] for item in text_messages)
    assert all("typed_envelope" not in item["content"] for item in text_messages)
    assert text_result.metrics.dynamic_route == ["planner", "retriever", "summarizer"]
    assert "executor" in text_result.metrics.skipped_agents
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


def test_modes_strip_utf8_bom_from_task_files(tmp_path: Path) -> None:
    task = tmp_path / "bom_task.txt"
    task.write_text("Summarize BOM-safe task loading.", encoding="utf-8-sig")
    paths = RuntimePaths(root=tmp_path)

    text_result = run_text_mode(task_path=task, paths=paths, load_configured_llm=False)
    protocol_result = run_protocol_mode(task_path=task, paths=paths, load_configured_llm=False)

    assert "\ufeff" not in read_jsonl(paths.text_agent_io)[0]["input"]["content"]
    protocol_params = read_jsonl(paths.protocol_agent_io)[0]["input"]["params"]
    assert "\ufeff" not in str(protocol_params)
    assert text_result.answer
    assert protocol_result.answer


def test_protocol_mode_skips_inproc_handshake_by_default(tmp_path: Path) -> None:
    task = tmp_path / "task.txt"
    task.write_text("Summarize protocol state passing.", encoding="utf-8")
    paths = RuntimePaths(root=tmp_path)

    run_protocol_mode(task_path=task, paths=paths, load_configured_llm=False)

    messages = read_jsonl(paths.protocol_messages)
    assert not any(item.get("msg_type") == "HELLO" for item in messages)
    assert not any(item.get("msg_type") == "CAPABILITY_ADVERTISE" for item in messages)
    assert not any(item.get("action") == "protocol.map" for item in messages)


def test_protocol_mode_can_keep_handshake_for_protocol_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AGENTMESH_PROTOCOL_SKIP_HANDSHAKE_FOR_INPROC", "false")
    task = tmp_path / "task.txt"
    task.write_text("Summarize protocol state passing.", encoding="utf-8")
    paths = RuntimePaths(root=tmp_path)

    run_protocol_mode(task_path=task, paths=paths, load_configured_llm=False)

    messages = read_jsonl(paths.protocol_messages)
    assert any(item.get("msg_type") == "HELLO" for item in messages)
    assert any(item.get("msg_type") == "CAPABILITY_ADVERTISE" for item in messages)
    assert any(item.get("action") == "protocol.map" for item in messages)


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


def test_benchmark_records_reproducible_provenance_and_statistics(
    tmp_path: Path,
) -> None:
    for name in ["task-a.txt", "task-b.txt"]:
        (tmp_path / name).write_text(
            f"Analyze reproducible protocol benchmark for {name}.",
            encoding="utf-8",
        )
    suite = tmp_path / "paired.yaml"
    suite.write_text(
        """
name: paired_suite
repeat: 2
seed: 17
tasks:
  - id: A
    input_file: task-a.txt
  - id: B
    input_file: task-b.txt
""".strip(),
        encoding="utf-8",
    )
    paths = RuntimePaths(root=tmp_path)

    summary = run_benchmark(suite_path=suite, paths=paths, use_llm=False)

    assert summary.schema_version == "2.0"
    assert summary.track == "deterministic"
    assert summary.repeat_count == 2
    assert summary.total_runs == 4
    assert summary.text_latency_stats.count == 4
    assert summary.protocol_latency_stats.count == 4

    manifest = orjson.loads(
        paths.benchmark_suite_manifest(
            "paired_suite",
            track="deterministic",
        ).read_bytes()
    )
    assert manifest["experiment_id"] == summary.experiment_id
    assert manifest["seed"] == 17

    detail = read_jsonl(
        paths.benchmark_suite_detail(
            "paired_suite",
            track="deterministic",
        )
    )
    assert len(detail) == 8
    assert {row["repeat_index"] for row in detail} == {1, 2}
    assert {tuple(row["pair_order"]) for row in detail} == {
        ("text", "protocol"),
        ("protocol", "text"),
    }
    assert all(row["task_sha256"] for row in detail)


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

    summary = run_benchmark(suite_path=suite, paths=paths, use_llm=False)
    report_path = generate_report(
        paths=paths,
        suite_name=summary.suite_name,
        track=summary.track.value,
    )

    assert summary.total_runs == 10
    assert summary.token_estimator == "mixed_cjk"
    assert summary.text_wire_bytes > 0
    assert summary.protocol_wire_bytes > 0
    assert summary.protocol_agent_io_bytes > 0
    assert summary.fair_wire_reduction_rate != 0
    assert summary.agent_io_bytes_reduction_rate != 0
    assert summary.memory_evidence_count >= 0
    assert summary.memory_evidence_bytes >= 0
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
    assert paths.benchmark_suite_summary("tiny_suite", track="deterministic").exists()
    assert paths.benchmark_suite_detail("tiny_suite", track="deterministic").exists()
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "TokenEstimator" in report_text
    assert "TokenSavingRate" in report_text
    assert "WireBytesReductionRate" in report_text
    assert "FairWireReductionRate" in report_text
    assert "AgentIoBytesReductionRate" in report_text
    assert "ProtocolAgentIoBytes" in report_text
    assert "MemoryAvgReusedUnitsPerQuery" in report_text
    assert "MemoryEvidenceCount" in report_text
    assert "MemoryEvidenceBytes" in report_text
    assert "MemoryAvgEvidenceBytesPerQuery" in report_text
    assert "SchemaVersion: 2.0" in report_text
    assert "Track: deterministic" in report_text
    assert "ExperimentId: exp-" in report_text
    assert "Latency P50/P95" in report_text
    assert "MemoryAvgScore" in report_text
    assert "MemoryAvgSemanticSimilarity" in report_text
    assert "MemoryAvgTagOverlapScore" in report_text
    assert "Quality scored pairs" in report_text
    assert "Unscored pairs" in report_text
    assert "Text quality mean / pass rate" in report_text
    assert "Protocol quality mean / pass rate" in report_text
    assert "Quality score delta" in report_text
    assert "QualityPreservationRate" not in report_text
    assert "FeedbackRoundCount" in report_text
    assert "TransportSendCount" in report_text
    assert "StateShmTransferCount" in report_text
    assert "LatestDynamicRoute" in report_text
    assert "Text Agent I/O Sample" in report_text
    assert "Protocol Agent I/O Sample" in report_text
    assert (
        len(
            read_jsonl(
                paths.benchmark_suite_detail("tiny_suite", track="deterministic")
            )
        )
        == 20
    )


def test_cli_benchmark_suite_aliases_resolve_to_builtin_suites(tmp_path: Path) -> None:
    assert _benchmark_suite_path(tmp_path, Path("standard")) == (
        tmp_path / "examples" / "benchmarks" / "continuous_tasks.yaml"
    )
    assert _benchmark_suite_path(tmp_path, Path("long")) == (
        tmp_path / "examples" / "benchmarks" / "long_context_tasks.yaml"
    )
    assert _benchmark_suite_path(tmp_path, Path("showcase")) == (
        tmp_path / "examples" / "benchmarks" / "showcase_benchmark.yaml"
    )
    assert _benchmark_suite_path(tmp_path, Path("custom.yaml")) == tmp_path / "custom.yaml"


def test_benchmark_reports_are_scoped_per_suite(tmp_path: Path, monkeypatch) -> None:
    task = tmp_path / "task.txt"
    task.write_text("Explain protocol memory reuse.", encoding="utf-8")
    suite_a = tmp_path / "suite_a.yaml"
    suite_b = tmp_path / "suite_b.yaml"
    for suite, name in [(suite_a, "suite_a"), (suite_b, "suite_b")]:
        suite.write_text(
            f"""
name: {name}
tasks:
  - id: T1
    group: T
    topic: protocol memory
    input_file: task.txt
""".strip(),
            encoding="utf-8",
        )

    def fake_text_mode(*, task_path, paths, load_configured_llm=True):
        del task_path, paths, load_configured_llm
        return ModeRunResult(
            mode="text",
            trace_id="trace-text",
            answer="text",
            metrics=RunMetrics(estimated_tokens=10, wire_bytes=100),
        )

    def fake_protocol_mode(*, task_path, paths, load_configured_llm=True):
        del task_path, paths, load_configured_llm
        return ModeRunResult(
            mode="protocol",
            trace_id="trace-protocol",
            answer="protocol",
            metrics=RunMetrics(estimated_tokens=5, wire_bytes=50),
        )

    monkeypatch.setattr("agentmesh.eval.benchmark.run_text_mode", fake_text_mode)
    monkeypatch.setattr("agentmesh.eval.benchmark.run_protocol_mode", fake_protocol_mode)

    paths = RuntimePaths(root=tmp_path)
    summary_a = run_benchmark(suite_path=suite_a, paths=paths, use_llm=False)
    report_a = generate_report(
        paths=paths,
        suite_name=summary_a.suite_name,
        track=summary_a.track.value,
    )
    summary_b = run_benchmark(suite_path=suite_b, paths=paths, use_llm=False)
    report_b = generate_report(
        paths=paths,
        suite_name=summary_b.suite_name,
        track=summary_b.track.value,
    )

    assert report_a.exists()
    assert report_b.exists()
    assert paths.benchmark_suite_summary("suite_a", track="deterministic").exists()
    assert paths.benchmark_suite_summary("suite_b", track="deterministic").exists()
    assert paths.benchmark_suite_detail("suite_a", track="deterministic").exists()
    assert paths.benchmark_suite_detail("suite_b", track="deterministic").exists()


def test_benchmark_applies_same_quality_rule_to_both_modes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = tmp_path / "task.txt"
    task.write_text("Return alpha and beta.", encoding="utf-8")
    suite = tmp_path / "quality_suite.yaml"
    suite.write_text(
        """
name: quality_suite
tasks:
  - id: Q1
    input_file: task.txt
    quality:
      rule_id: two-facts-v1
      kind: contains_all
      expected: [alpha, beta]
  - id: Q2
    input_file: task.txt
""".strip(),
        encoding="utf-8",
    )

    def fake_text_mode(*, task_path, paths, load_configured_llm=True):
        del task_path, paths, load_configured_llm
        return ModeRunResult(
            mode="text",
            trace_id="trace-text",
            answer="alpha",
            metrics=RunMetrics(estimated_tokens=10, wire_bytes=100),
        )

    def fake_protocol_mode(*, task_path, paths, load_configured_llm=True):
        del task_path, paths, load_configured_llm
        return ModeRunResult(
            mode="protocol",
            trace_id="trace-protocol",
            answer="alpha beta",
            metrics=RunMetrics(estimated_tokens=5, wire_bytes=50),
        )

    monkeypatch.setattr("agentmesh.eval.benchmark.run_text_mode", fake_text_mode)
    monkeypatch.setattr("agentmesh.eval.benchmark.run_protocol_mode", fake_protocol_mode)

    paths = RuntimePaths(root=tmp_path)
    summary = run_benchmark(suite_path=suite, paths=paths, use_llm=False)

    assert summary.quality_scored_runs == 1
    assert summary.quality_unscored_runs == 1
    assert summary.text_quality_mean == 0.5
    assert summary.protocol_quality_mean == 1.0
    assert summary.text_quality_pass_rate == 0.0
    assert summary.protocol_quality_pass_rate == 1.0
    assert summary.quality_score_delta == 0.5

    rows = read_jsonl(
        paths.benchmark_suite_detail("quality_suite", track="deterministic")
    )
    scored_rows = [row for row in rows if row["task_id"] == "Q1"]
    unscored_rows = [row for row in rows if row["task_id"] == "Q2"]
    assert {row["quality"]["rule_id"] for row in scored_rows} == {"two-facts-v1"}
    assert {row["quality"]["score"] for row in scored_rows} == {0.5, 1.0}
    assert all(not row["quality"]["scored"] for row in unscored_rows)


def test_benchmark_rerun_overwrites_same_suite_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = tmp_path / "task.txt"
    task.write_text("Explain protocol memory reuse.", encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """
name: repeated_suite
tasks:
  - id: T1
    group: T
    topic: protocol memory
    input_file: task.txt
""".strip(),
        encoding="utf-8",
    )

    def fake_text_mode(*, task_path, paths, load_configured_llm=True):
        del task_path, paths, load_configured_llm
        return ModeRunResult(
            mode="text",
            trace_id="trace-text",
            answer="text",
            metrics=RunMetrics(estimated_tokens=10, wire_bytes=100),
        )

    def fake_protocol_mode(*, task_path, paths, load_configured_llm=True):
        del task_path, paths, load_configured_llm
        return ModeRunResult(
            mode="protocol",
            trace_id="trace-protocol",
            answer="protocol",
            metrics=RunMetrics(estimated_tokens=5, wire_bytes=50),
        )

    monkeypatch.setattr("agentmesh.eval.benchmark.run_text_mode", fake_text_mode)
    monkeypatch.setattr("agentmesh.eval.benchmark.run_protocol_mode", fake_protocol_mode)

    paths = RuntimePaths(root=tmp_path)
    run_benchmark(suite_path=suite, paths=paths, use_llm=False)
    run_benchmark(suite_path=suite, paths=paths, use_llm=False)

    assert (
        len(
            read_jsonl(
                paths.benchmark_suite_detail(
                    "repeated_suite",
                    track="deterministic",
                )
            )
        )
        == 2
    )


def test_benchmark_uses_configured_llm_for_both_modes_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = tmp_path / "task.txt"
    task.write_text("Explain protocol memory reuse.", encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """
name: tiny_suite
tasks:
  - id: T1
    group: T
    topic: protocol memory
    input_file: task.txt
""".strip(),
        encoding="utf-8",
    )
    calls: list[tuple[str, bool]] = []

    def fake_text_mode(*, task_path, paths, load_configured_llm=True):
        del task_path, paths
        calls.append(("text", load_configured_llm))
        return ModeRunResult(
            mode="text",
            trace_id="trace-text",
            answer="text",
            metrics=RunMetrics(estimated_tokens=10, wire_bytes=100),
        )

    def fake_protocol_mode(*, task_path, paths, load_configured_llm=True):
        del task_path, paths
        calls.append(("protocol", load_configured_llm))
        return ModeRunResult(
            mode="protocol",
            trace_id="trace-protocol",
            answer="protocol",
            metrics=RunMetrics(
                estimated_tokens=5,
                wire_bytes=50,
                memory_query_count=1,
                memory_query_hit_count=1,
                memory_hit_count=1,
                memory_reused_unit_count=2,
                memory_avg_score=0.6,
                memory_avg_semantic_similarity=0.4,
                memory_avg_tag_overlap_score=0.5,
            ),
        )

    monkeypatch.setattr("agentmesh.eval.benchmark.run_text_mode", fake_text_mode)
    monkeypatch.setattr("agentmesh.eval.benchmark.run_protocol_mode", fake_protocol_mode)

    run_benchmark(suite_path=suite, paths=RuntimePaths(root=tmp_path))

    assert calls == [("text", True), ("protocol", True)]


def test_benchmark_can_disable_configured_llm_for_both_modes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = tmp_path / "task.txt"
    task.write_text("Explain protocol memory reuse.", encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """
name: tiny_suite
tasks:
  - id: T1
    group: T
    topic: protocol memory
    input_file: task.txt
""".strip(),
        encoding="utf-8",
    )
    calls: list[tuple[str, bool]] = []

    def fake_text_mode(*, task_path, paths, load_configured_llm=True):
        del task_path, paths
        calls.append(("text", load_configured_llm))
        return ModeRunResult(
            mode="text",
            trace_id="trace-text",
            answer="text",
            metrics=RunMetrics(estimated_tokens=10, wire_bytes=100),
        )

    def fake_protocol_mode(*, task_path, paths, load_configured_llm=True):
        del task_path, paths
        calls.append(("protocol", load_configured_llm))
        return ModeRunResult(
            mode="protocol",
            trace_id="trace-protocol",
            answer="protocol",
            metrics=RunMetrics(estimated_tokens=5, wire_bytes=50),
        )

    monkeypatch.setattr("agentmesh.eval.benchmark.run_text_mode", fake_text_mode)
    monkeypatch.setattr("agentmesh.eval.benchmark.run_protocol_mode", fake_protocol_mode)

    run_benchmark(suite_path=suite, paths=RuntimePaths(root=tmp_path), use_llm=False)

    assert calls == [("text", False), ("protocol", False)]


def test_benchmark_reports_progress_for_each_task_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = tmp_path / "task.txt"
    task.write_text("Explain protocol memory reuse.", encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """
name: progress_suite
repeat: 2
tasks:
  - id: T1
    group: T
    topic: protocol memory
    input_file: task.txt
""".strip(),
        encoding="utf-8",
    )

    def fake_text_mode(*, task_path, paths, load_configured_llm=True):
        del task_path, paths, load_configured_llm
        return ModeRunResult(
            mode="text",
            trace_id="trace-text",
            answer="text",
            metrics=RunMetrics(estimated_tokens=10, wire_bytes=100, latency_ms=11),
        )

    def fake_protocol_mode(*, task_path, paths, load_configured_llm=True):
        del task_path, paths, load_configured_llm
        return ModeRunResult(
            mode="protocol",
            trace_id="trace-protocol",
            answer="protocol",
            metrics=RunMetrics(estimated_tokens=5, wire_bytes=50, latency_ms=22),
        )

    monkeypatch.setattr("agentmesh.eval.benchmark.run_text_mode", fake_text_mode)
    monkeypatch.setattr("agentmesh.eval.benchmark.run_protocol_mode", fake_protocol_mode)

    events = []
    summary = run_benchmark(
        suite_path=suite,
        paths=RuntimePaths(root=tmp_path),
        use_llm=False,
        progress_callback=events.append,
    )

    assert summary.total_runs == 2
    assert [event.phase for event in events] == [
        "suite_start",
        "mode_complete",
        "mode_complete",
        "mode_complete",
        "mode_complete",
        "suite_complete",
    ]
    mode_events = [event for event in events if event.phase == "mode_complete"]
    assert [(event.current_stage, event.current_task, event.mode) for event in mode_events] == [
        (1, 1, "text"),
        (2, 1, "protocol"),
        (3, 2, "protocol"),
        (4, 2, "text"),
    ]
    assert all(event.task_id == "T1" for event in mode_events)
    assert mode_events[0].latency_ms == 11
    assert mode_events[1].latency_ms == 22


def test_benchmark_cold_start_resets_run_memory_and_reports_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = tmp_path / "task.txt"
    task.write_text("Explain protocol memory reuse.", encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """
name: cold_start_suite
tasks:
  - id: C1
    group: C
    topic: cold baseline
    input_file: task.txt
    tags: [phase-c, cold]
    depends_on: [B1]
    cold_start: true
""".strip(),
        encoding="utf-8",
    )
    paths = RuntimePaths(root=tmp_path)
    paths.ensure()
    paths.memory_db.write_text("stale run memory", encoding="utf-8")
    calls: list[tuple[str, bool, bool]] = []

    def stale_memory_file_present(paths: RuntimePaths) -> bool:
        try:
            return paths.memory_db.read_text(encoding="utf-8") == "stale run memory"
        except UnicodeDecodeError:
            return False
        except FileNotFoundError:
            return False

    def fake_text_mode(*, task_path, paths, load_configured_llm=True):
        del task_path, load_configured_llm
        calls.append(("text", paths.memory_db.exists(), stale_memory_file_present(paths)))
        return ModeRunResult(
            mode="text",
            trace_id="trace-text",
            answer="text",
            metrics=RunMetrics(estimated_tokens=10, wire_bytes=100),
        )

    def fake_protocol_mode(*, task_path, paths, load_configured_llm=True):
        del task_path, load_configured_llm
        calls.append(("protocol", paths.memory_db.exists(), stale_memory_file_present(paths)))
        return ModeRunResult(
            mode="protocol",
            trace_id="trace-protocol",
            answer="protocol",
            metrics=RunMetrics(estimated_tokens=5, wire_bytes=50),
        )

    monkeypatch.setattr("agentmesh.eval.benchmark.run_text_mode", fake_text_mode)
    monkeypatch.setattr("agentmesh.eval.benchmark.run_protocol_mode", fake_protocol_mode)

    events = []
    run_benchmark(
        suite_path=suite,
        paths=paths,
        use_llm=False,
        progress_callback=events.append,
    )

    assert [call[0] for call in calls] == ["text", "protocol"]
    assert all(stale_present is False for _mode, _exists, stale_present in calls)
    assert any(event.phase == "cold_start" for event in events)
    mode_event = next(event for event in events if event.phase == "mode_complete")
    assert mode_event.cold_start is True
    assert mode_event.depends_on == ["B1"]
    assert mode_event.tags == ["phase-c", "cold"]
    detail_rows = read_jsonl(
        paths.benchmark_suite_detail(
            "cold_start_suite",
            track="deterministic",
        )
    )
    assert detail_rows[0]["cold_start"] is True
    assert detail_rows[0]["depends_on"] == ["B1"]
    assert detail_rows[0]["tags"] == ["phase-c", "cold"]


def test_benchmark_summarizes_memory_reuse_quality(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = tmp_path / "task.txt"
    task.write_text("Explain protocol memory reuse.", encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """
name: memory_quality_suite
tasks:
  - id: T1
    group: T
    topic: protocol memory
    input_file: task.txt
""".strip(),
        encoding="utf-8",
    )

    def fake_text_mode(*, task_path, paths, load_configured_llm=True):
        del task_path, paths, load_configured_llm
        return ModeRunResult(
            mode="text",
            trace_id="trace-text",
            answer="text",
            metrics=RunMetrics(estimated_tokens=10, wire_bytes=100),
        )

    def fake_protocol_mode(*, task_path, paths, load_configured_llm=True):
        del task_path, paths, load_configured_llm
        return ModeRunResult(
            mode="protocol",
            trace_id="trace-protocol",
            answer="protocol",
            metrics=RunMetrics(
                estimated_tokens=5,
                wire_bytes=50,
                memory_query_count=1,
                memory_query_hit_count=1,
                memory_reused_unit_count=4,
                memory_avg_score=0.7,
                memory_avg_semantic_similarity=0.3,
                memory_avg_tag_overlap_score=0.5,
            ),
        )

    monkeypatch.setattr("agentmesh.eval.benchmark.run_text_mode", fake_text_mode)
    monkeypatch.setattr("agentmesh.eval.benchmark.run_protocol_mode", fake_protocol_mode)

    summary = run_benchmark(suite_path=suite, paths=RuntimePaths(root=tmp_path), use_llm=False)

    assert summary.memory_avg_reused_units_per_query == 4.0
    assert summary.memory_avg_score == 0.7
    assert summary.memory_avg_semantic_similarity == 0.3
    assert summary.memory_avg_tag_overlap_score == 0.5


def test_text_mode_summarizer_fallback_preserves_executor_output(tmp_path: Path) -> None:
    task = tmp_path / "sort.txt"
    task.write_text("Write validation code and output the validation result.", encoding="utf-8")
    paths = RuntimePaths(root=tmp_path)

    result = run_text_mode(
        task_path=task,
        paths=paths,
        llm_client=TextModeFailingSummarizerLLM(),
    )

    assert "summarizer LLM unavailable" in result.answer
    assert "print(status)" in result.answer
    assert result.answer != "processed task with full text context."
    agent_io = read_jsonl(paths.text_agent_io)
    assert agent_io[-1]["agent"] == "summarizer"
    assert "print(status)" in agent_io[-1]["output"]["content"]
