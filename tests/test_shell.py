from io import StringIO

from rich.console import Console

from agentmesh.eval.compare import CompareAgentOutput, CompareSummary
from agentmesh.eval.benchmark import BenchmarkProgressEvent
from agentmesh.eval.metrics import ModeRunResult, RunMetrics
from agentmesh.memory.schema import MemoryUnit
from agentmesh.memory.search import MemorySearchResult
from agentmesh.shell.commands import parse_shell_line
from agentmesh.shell.render import _aligned_agent_output_rows, render_compare, render_memory
from agentmesh.shell.session import ShellSession
from agentmesh.storage.agent_io import append_protocol_agent_io
from agentmesh.storage.jsonl import append_jsonl
from agentmesh.storage.paths import RuntimePaths


def test_shell_parser_routes_plain_text_to_ask() -> None:
    command = parse_shell_line("Analyze agent communication overhead")

    assert command is not None
    assert command.name == "ask"
    assert command.args == ["Analyze agent communication overhead"]


def test_shell_parser_handles_slash_command_arguments() -> None:
    command = parse_shell_line('/compare --llm "中文 Agent 任务"')

    assert command is not None
    assert command.name == "compare"
    assert command.args == ["--llm", "中文 Agent 任务"]


def test_shell_help_and_config_do_not_fail(tmp_path) -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=120)
    session = ShellSession(paths=RuntimePaths(root=tmp_path), console=console)

    assert session.handle_line("/help")
    assert session.handle_line("/config")

    rendered = output.getvalue()
    assert "AgentMesh shell commands" in rendered
    assert "/benchmark [--no-llm] standard" in rendered
    assert "运行标准评测套件" in rendered
    assert "/benchmark [--no-llm] long" in rendered
    assert "运行长上下文评测套件" in rendered
    assert "/benchmark [--no-llm] <suite.yaml>" in rendered
    assert "运行自定义 YAML 评测套件" in rendered
    assert "llm_configured" in rendered
    assert "api_key" not in rendered.lower()


def test_shell_compare_runs_both_modes(tmp_path) -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=140)
    session = ShellSession(paths=RuntimePaths(root=tmp_path), console=console)

    assert session.handle_line("/compare Analyze structured protocol savings.")

    rendered = output.getvalue()
    assert "Text Mode vs Protocol Mode" in rendered
    assert "token_saving_rate" in rendered


def test_shell_compare_uses_llm_by_default_and_can_disable_it(
    tmp_path,
    monkeypatch,
) -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=140)
    session = ShellSession(paths=RuntimePaths(root=tmp_path), console=console)
    calls: list[bool] = []

    class FakeSummary:
        task_path = "task.txt"
        text = ModeRunResult(
            mode="text",
            trace_id="trace-text",
            answer="text",
            metrics=RunMetrics(message_count=1),
        )
        protocol = ModeRunResult(
            mode="protocol",
            trace_id="trace-protocol",
            answer="protocol",
            metrics=RunMetrics(message_count=1),
        )
        text_agent_outputs = []
        protocol_agent_outputs = []
        token_saving_rate = 0.0
        wire_bytes_reduction_rate = 0.0
        latency_reduction_rate = 0.0
        quality_preservation_rate = 1.0
        memory_hit_rate = 0.0

    def fake_summary(prompt, paths, *, use_llm=True, progress_callback=None):
        del prompt, paths, progress_callback
        calls.append(use_llm)
        return FakeSummary()

    monkeypatch.setattr("agentmesh.shell.session.run_prompt_compare", fake_summary)

    assert session.handle_line("/compare hello")
    assert session.handle_line("/compare --no-llm hello")

    assert calls == [True, False]


def test_shell_benchmark_prints_progress_and_output_paths(tmp_path, monkeypatch) -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=160)
    session = ShellSession(paths=RuntimePaths(root=tmp_path), console=console)

    suite = tmp_path / "examples" / "benchmarks" / "continuous_tasks.yaml"
    suite.parent.mkdir(parents=True)
    suite.write_text(
        """
name: shell_progress_suite
tasks:
  - id: T1
    group: T
    topic: protocol memory
    input_file: task.txt
""".strip(),
        encoding="utf-8",
    )

    class FakeSummary:
        suite_name = "shell_progress_suite"
        total_runs = 1

    def fake_run_benchmark(suite_path, paths, *, use_llm=True, progress_callback=None):
        del suite_path, paths, use_llm
        assert progress_callback is not None
        progress_callback(
            BenchmarkProgressEvent(
                phase="suite_start",
                suite_name="shell_progress_suite",
                total_tasks=1,
                total_stages=2,
                repeat_total=1,
                use_llm=False,
            )
        )
        progress_callback(
            BenchmarkProgressEvent(
                phase="mode_complete",
                suite_name="shell_progress_suite",
                total_tasks=1,
                total_stages=2,
                current_task=1,
                current_stage=1,
                repeat_index=1,
                repeat_total=1,
                task_id="T1",
                group="T",
                topic="protocol memory",
                mode="text",
                trace_id="trace-text",
                latency_ms=12,
                tokens=34,
                bytes=56,
                use_llm=False,
            )
        )
        progress_callback(
            BenchmarkProgressEvent(
                phase="suite_complete",
                suite_name="shell_progress_suite",
                total_tasks=1,
                total_stages=2,
                current_task=1,
                current_stage=2,
                repeat_total=1,
                use_llm=False,
            )
        )
        return FakeSummary()

    def fake_render_benchmark(console, summary):
        console.print(f"rendered benchmark {summary.suite_name}")

    monkeypatch.setattr("agentmesh.shell.session.run_benchmark", fake_run_benchmark)
    monkeypatch.setattr("agentmesh.shell.session.render_benchmark", fake_render_benchmark)

    assert session.handle_line("/benchmark --no-llm standard")

    rendered = output.getvalue()
    assert "benchmark started" in rendered
    assert "stage completed" in rendered
    assert "任务 1/1" in rendered
    assert "id=T1" in rendered
    assert "mode=text" in rendered
    assert "benchmark completed" in rendered
    assert "Benchmark output files" in rendered
    assert "benchmark_summary.csv" in rendered
    assert "benchmark_detail.jsonl" in rendered
    assert "experiment_report.md" in rendered


def test_render_compare_shows_answers_agent_outputs_then_metrics() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=140)
    summary = CompareSummary(
        task_path="task.txt",
        text=ModeRunResult(
            mode="text",
            trace_id="trace-text",
            answer="text final answer",
            metrics=RunMetrics(
                message_count=4,
                estimated_tokens=100,
                wire_bytes=1000,
                agent_io_tokens=120,
                agent_io_bytes=1100,
                per_msg_avg_tokens=30.0,
                answer_quality_score=0.8,
            ),
        ),
        protocol=ModeRunResult(
            mode="protocol",
            trace_id="trace-protocol",
            answer="protocol final answer",
            metrics=RunMetrics(
                message_count=20,
                estimated_tokens=20,
                wire_bytes=400,
                agent_io_tokens=60,
                agent_io_bytes=500,
                per_msg_avg_tokens=3.0,
                protocol_total_bytes=900,
                state_transfer_bytes=256,
                stage_latency_ms={"planner": 10, "retriever": 5},
                answer_quality_score=0.7,
                memory_query_count=2,
                memory_query_hit_count=1,
                memory_reused_unit_count=3,
                memory_avg_score=0.7,
                memory_avg_semantic_similarity=0.5,
                memory_avg_tag_overlap_score=0.25,
            ),
        ),
        text_agent_outputs=[
            CompareAgentOutput(
                mode="text",
                trace_id="trace-text",
                step=1,
                agent="planner",
                output="text planner output",
            )
        ],
        protocol_agent_outputs=[
            CompareAgentOutput(
                mode="protocol",
                trace_id="trace-protocol",
                step=1,
                agent="planner",
                action="plan.create",
                output="protocol planner output",
                state_refs_in=["state://task"],
                state_refs_out=["state://plan"],
            )
        ],
        token_saving_rate=0.8,
        latency_reduction_rate=0.1,
        wire_bytes_reduction_rate=0.6,
        quality_preservation_rate=0.875,
        memory_hit_rate=0.0,
    )

    render_compare(console, summary)

    rendered = output.getvalue()
    assert rendered.index("Final Answers") < rendered.index("Agent Outputs")
    assert rendered.index("Agent Outputs") < rendered.index("Text Mode vs Protocol Mode")
    assert "text final answer" in rendered
    assert "protocol final answer" in rendered
    assert "text planner output" in rendered
    assert "protocol planner output" in rendered
    assert "messages（消息数）" in rendered
    assert "agent I/O tokens（Agent 通信文本 token）" in rendered
    assert "agent I/O bytes（Agent 通信字节）" in rendered
    assert "state transfer bytes（状态传递字节数）" in rendered
    assert "quality preservation rate（回答质量保持率）" in rendered
    assert "Protocol Mode Latency Breakdown" in rendered
    assert "planner" in rendered
    assert "memory reused units（复用记忆条数）" in rendered
    assert "memory avg semantic（平均语义相似度）" in rendered
    assert "token_saving_rate（token 节省率）" in rendered
    assert "memory avg tag overlap" not in rendered


def test_render_memory_shows_semantic_score_and_vector_source_content() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=160)
    unit = MemoryUnit(
        source_agent="summarizer",
        task_topic="sorting",
        summary="Short summary.",
        content="Original vectorized text about sorting stability and ranking.",
        tags=["sort"],
        evidence_refs=[],
        state_refs=[],
        confidence=0.9,
        validity_score=0.9,
        provenance_trace_id="trace-render",
    )

    render_memory(console, [MemorySearchResult(memory=unit, score=0.81234)])

    rendered = output.getvalue()
    assert "0.8123" in rendered
    assert "Original vectorized text about sorting stability and ranking." in rendered


def test_render_compare_does_not_ellipsis_long_agent_log_lines() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=88)
    long_ref = "state://embedding/state-abcdef1234567890abcdef1234567890"
    summary = CompareSummary(
        task_path="task.txt",
        text=ModeRunResult(
            mode="text",
            trace_id="trace-text",
            answer="text final answer",
            metrics=RunMetrics(message_count=1),
        ),
        protocol=ModeRunResult(
            mode="protocol",
            trace_id="trace-protocol",
            answer="protocol final answer",
            metrics=RunMetrics(message_count=1),
        ),
        text_agent_outputs=[],
        protocol_agent_outputs=[
            CompareAgentOutput(
                mode="protocol",
                trace_id="trace-protocol",
                step=1,
                agent="planner",
                action="plan.create",
                input_summary='{"state_refs": ["' + long_ref + '"]}',
                output='{"state_refs": ["' + long_ref + '"]}',
                state_refs_in=[long_ref],
                state_refs_out=[long_ref],
            )
        ],
        token_saving_rate=0.0,
        latency_reduction_rate=0.0,
        wire_bytes_reduction_rate=0.0,
        memory_hit_rate=0.0,
    )

    render_compare(console, summary)

    rendered = output.getvalue()
    assert "..." not in rendered
    assert "state-abcdef123456789" in rendered
    assert "0abcdef1234567890" in rendered


def test_compare_agent_outputs_align_by_agent_when_protocol_skips_steps() -> None:
    text_outputs = [
        CompareAgentOutput(mode="text", trace_id="trace-text", step=1, agent="planner"),
        CompareAgentOutput(mode="text", trace_id="trace-text", step=2, agent="retriever"),
        CompareAgentOutput(mode="text", trace_id="trace-text", step=3, agent="executor"),
        CompareAgentOutput(mode="text", trace_id="trace-text", step=4, agent="summarizer"),
    ]
    protocol_outputs = [
        CompareAgentOutput(mode="protocol", trace_id="trace-protocol", step=1, agent="planner"),
        CompareAgentOutput(mode="protocol", trace_id="trace-protocol", step=2, agent="summarizer"),
    ]

    rows = _aligned_agent_output_rows(text_outputs, protocol_outputs)

    aligned_agents = [
        (left.agent if left else None, right.agent if right else None) for left, right in rows
    ]
    assert aligned_agents == [
        ("planner", "planner"),
        ("retriever", None),
        ("executor", None),
        ("summarizer", "summarizer"),
    ]


def test_shell_trace_shows_agent_io_logs(tmp_path) -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=140)
    paths = RuntimePaths(root=tmp_path)
    session = ShellSession(paths=paths, console=console)
    append_jsonl(
        paths.protocol_agent_io,
        {
            "mode": "protocol",
            "trace_id": "trace-io",
            "agent": "planner",
            "input": {"action": "plan.create"},
            "output": {"result": {"plan": ["step"]}},
        },
    )

    assert session.handle_line("/trace 1")

    rendered = output.getvalue()
    assert "trace-io" in rendered
    assert "plan.create" in rendered


def test_shell_ask_routes_through_protocol_mode(tmp_path, monkeypatch) -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=140)
    session = ShellSession(paths=RuntimePaths(root=tmp_path), console=console)
    calls: list[dict[str, object]] = []

    def fake_protocol_mode(*, task_path, paths, load_configured_llm=True):
        calls.append(
            {
                "task_path": task_path,
                "paths": paths,
                "load_configured_llm": load_configured_llm,
                "task_text": task_path.read_text(encoding="utf-8"),
            }
        )
        return ModeRunResult(
            mode="protocol",
            trace_id="trace-shell-test",
            answer="protocol answer",
            metrics=RunMetrics(message_count=3),
        )

    monkeypatch.setattr("agentmesh.shell.session.run_protocol_mode", fake_protocol_mode)

    assert session.handle_line("/ask Remember protocol state refs.")

    assert len(calls) == 1
    assert calls[0]["paths"] == session.paths
    assert calls[0]["load_configured_llm"] is True
    assert "User: Remember protocol state refs." in str(calls[0]["task_text"])
    assert "protocol answer" in output.getvalue()
    assert session.history[-1].assistant == "protocol answer"


def test_shell_ask_streams_protocol_agent_outputs(tmp_path, monkeypatch) -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=140)
    session = ShellSession(paths=RuntimePaths(root=tmp_path), console=console)

    def fake_protocol_mode(*, task_path, paths, load_configured_llm=True):
        del task_path, load_configured_llm
        append_protocol_agent_io(
            paths=paths,
            trace_id="trace-shell-stream",
            step=1,
            source_agent="runtime",
            agent="planner",
            action="plan.create",
            params={"task": "stream ask"},
            result={"plan": ["streamed planner output"]},
            state_refs_in=["state://task"],
            state_refs_out=["state://plan"],
            result_msg_type="RESULT",
        )
        return ModeRunResult(
            mode="protocol",
            trace_id="trace-shell-stream",
            answer="protocol final answer",
            metrics=RunMetrics(message_count=3),
        )

    monkeypatch.setattr("agentmesh.shell.session.run_protocol_mode", fake_protocol_mode)

    assert session.handle_line("/ask stream ask")

    rendered = output.getvalue()
    assert "START Protocol Mode started" in rendered
    assert "Protocol Mode step 1: planner" in rendered
    assert "streamed planner output" in rendered
    assert "DONE Protocol Mode completed" in rendered
    assert "protocol final answer" in rendered
    assert rendered.index("Protocol Mode step 1: planner") < rendered.index("protocol final answer")


def test_shell_plain_text_routes_through_ask_protocol_mode(tmp_path, monkeypatch) -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=140)
    session = ShellSession(paths=RuntimePaths(root=tmp_path), console=console)
    calls: list[str] = []

    def fake_protocol_mode(*, task_path, paths, load_configured_llm=True):
        del paths
        assert load_configured_llm is True
        task_text = task_path.read_text(encoding="utf-8")
        calls.append(task_text)
        return ModeRunResult(
            mode="protocol",
            trace_id="trace-shell-plain-text",
            answer="plain text ask answer",
            metrics=RunMetrics(message_count=3),
        )

    monkeypatch.setattr("agentmesh.shell.session.run_protocol_mode", fake_protocol_mode)

    assert session.handle_line("directly write python validation code")

    assert len(calls) == 1
    assert "User: directly write python validation code" in calls[0]
    assert "plain text ask answer" in output.getvalue()


def test_shell_ask_prints_protocol_answer_without_rich_markup_stripping(
    tmp_path,
    monkeypatch,
) -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=140)
    session = ShellSession(paths=RuntimePaths(root=tmp_path), console=console)
    answer = (
        "```python\n"
        "pivot = arr[len(arr) // 2]\n"
        "left = [x for x in arr if x < pivot]\n"
        "```\n"
    )

    def fake_protocol_mode(*, task_path, paths, load_configured_llm=True):
        del task_path, paths, load_configured_llm
        return ModeRunResult(
            mode="protocol",
            trace_id="trace-shell-test",
            answer=answer,
            metrics=RunMetrics(message_count=3),
        )

    monkeypatch.setattr("agentmesh.shell.session.run_protocol_mode", fake_protocol_mode)

    assert session.handle_line("/ask write python code")

    rendered = output.getvalue()
    assert "pivot = arr[len(arr) // 2]" in rendered
    assert "left = [x for x in arr if x < pivot]" in rendered


def test_shell_exit_returns_false(tmp_path) -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False)
    session = ShellSession(paths=RuntimePaths(root=tmp_path), console=console)

    assert not session.handle_line("/exit")
