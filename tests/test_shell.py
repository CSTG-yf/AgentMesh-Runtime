from io import StringIO

from rich.console import Console

from agentmesh.shell.commands import parse_shell_line
from agentmesh.shell.session import ShellSession
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


def test_shell_exit_returns_false(tmp_path) -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False)
    session = ShellSession(paths=RuntimePaths(root=tmp_path), console=console)

    assert not session.handle_line("/exit")
