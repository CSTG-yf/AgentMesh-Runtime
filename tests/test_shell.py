from io import StringIO

from rich.console import Console

from agentmesh.eval.metrics import ModeRunResult, RunMetrics
from agentmesh.shell.commands import parse_shell_line
from agentmesh.shell.session import ShellSession
from agentmesh.storage.jsonl import append_jsonl
from agentmesh.storage.paths import RuntimePaths


def test_shell_parser_routes_plain_text_to_compare() -> None:
    command = parse_shell_line("Analyze agent communication overhead")

    assert command is not None
    assert command.name == "compare"
    assert command.args == ["Analyze agent communication overhead"]


def test_shell_parser_handles_slash_command_arguments() -> None:
    command = parse_shell_line('/compare --llm "中文多 Agent 任务"')

    assert command is not None
    assert command.name == "compare"
    assert command.args == ["--llm", "中文多 Agent 任务"]


def test_shell_help_and_config_do_not_fail(tmp_path) -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=120)
    session = ShellSession(paths=RuntimePaths(root=tmp_path), console=console)

    assert session.handle_line("/help")
    assert session.handle_line("/config")

    rendered = output.getvalue()
    assert "AgentMesh shell commands" in rendered
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
        return ModeRunResult(
            mode="protocol",
            trace_id="trace-shell-test",
            answer=answer,
            metrics=RunMetrics(message_count=3),
        )

    monkeypatch.setattr("agentmesh.shell.session.run_protocol_mode", fake_protocol_mode)

    assert session.handle_line("/ask write quicksort")

    rendered = output.getvalue()
    assert "pivot = arr[len(arr) // 2]" in rendered
    assert "left = [x for x in arr if x < pivot]" in rendered


def test_shell_exit_returns_false(tmp_path) -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False)
    session = ShellSession(paths=RuntimePaths(root=tmp_path), console=console)

    assert not session.handle_line("/exit")
