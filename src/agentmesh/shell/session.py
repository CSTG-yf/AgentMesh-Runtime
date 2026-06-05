from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from agentmesh.chat.session import ChatTurn, run_chat_turn
from agentmesh.config import AgentMeshConfig
from agentmesh.core import rust_available
from agentmesh.eval.benchmark import run_benchmark
from agentmesh.eval.compare import run_prompt_compare
from agentmesh.eval.report import generate_report
from agentmesh.memory.sqlite_store import SQLiteMemoryStore
from agentmesh.modes.protocol_mode import run_protocol_mode
from agentmesh.modes.text_mode import run_text_mode
from agentmesh.runtime.registry import RuntimeContext
from agentmesh.shell.commands import ParsedShellCommand, join_prompt, parse_shell_line
from agentmesh.shell.render import (
    render_benchmark,
    render_compare,
    render_config,
    render_help,
    render_memory,
    render_path,
)
from agentmesh.state.embedding import create_embedding_encoder
from agentmesh.state.store import StateStore
from agentmesh.storage.jsonl import read_jsonl
from agentmesh.storage.paths import RuntimePaths

InputFunc = Callable[[str], str]


class ShellSession:
    def __init__(self, *, paths: RuntimePaths, console: Console | None = None) -> None:
        self.paths = paths
        self.console = console or Console()
        self.context = RuntimeContext.from_paths(paths=paths, trace_id="trace-shell")
        self.history: list[ChatTurn] = []

    def run(self, input_func: InputFunc | None = None) -> None:
        prompt = input_func or (lambda label: Prompt.ask(label))
        self.console.print("[bold]AgentMesh shell[/bold] (输入 /help 查看命令，/exit 退出)")
        while True:
            try:
                line = prompt("agentmesh")
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return
            if not self.handle_line(line):
                return

    def handle_line(self, line: str) -> bool:
        command = parse_shell_line(line)
        if command is None:
            return True
        try:
            return self._dispatch(command)
        except Exception as exc:
            self.console.print(f"[red]command failed:[/red] {exc}")
            return True

    def _dispatch(self, command: ParsedShellCommand) -> bool:
        name = command.name
        if name in {"exit", "quit", "q"}:
            return False
        if name in {"help", "h", "?"}:
            render_help(self.console)
            return True
        if name == "compare":
            self._compare(command.args)
            return True
        if name == "ask":
            self._ask(command.args)
            return True
        if name == "run":
            self._run_mode(command.args)
            return True
        if name == "benchmark":
            self._benchmark(command.args)
            return True
        if name == "memory":
            self._memory(command.args)
            return True
        if name == "trace":
            self._trace(command.args)
            return True
        if name == "report":
            report_path = generate_report(self.paths)
            render_path(self.console, "Report written", report_path)
            return True
        if name == "config":
            self._config()
            return True
        self.console.print(f"[yellow]Unknown command:[/yellow] /{name}. Use /help.")
        return True

    def _compare(self, args: list[str]) -> None:
        use_llm = False
        prompt_args: list[str] = []
        for arg in args:
            if arg == "--llm":
                use_llm = True
            elif arg == "--no-llm":
                use_llm = False
            else:
                prompt_args.append(arg)
        prompt = join_prompt(prompt_args)
        if not prompt:
            self.console.print("[red]Usage:[/red] /compare [--llm] <task>")
            return
        summary = run_prompt_compare(prompt, paths=self.paths, use_llm=use_llm)
        render_compare(self.console, summary)

    def _ask(self, args: list[str]) -> None:
        message = join_prompt(args)
        if not message:
            self.console.print("[red]Usage:[/red] /ask <message>")
            return
        turn = run_chat_turn(message, context=self.context, history=self.history)
        self.history.append(turn)
        self.console.print(f"[bold]agentmesh[/bold]: {turn.assistant}")

    def _run_mode(self, args: list[str]) -> None:
        if len(args) != 2 or args[0] not in {"text", "protocol"}:
            self.console.print("[red]Usage:[/red] /run text|protocol <task-file>")
            return
        task_path = Path(args[1])
        if not task_path.is_absolute():
            task_path = self.paths.root / task_path
        result = (
            run_text_mode(task_path, self.paths)
            if args[0] == "text"
            else run_protocol_mode(task_path, self.paths)
        )
        self.console.print(result.model_dump())

    def _benchmark(self, args: list[str]) -> None:
        if len(args) != 1:
            self.console.print("[red]Usage:[/red] /benchmark standard|long|<suite.yaml>")
            return
        suite = _suite_path(self.paths.root, args[0])
        summary = run_benchmark(suite, self.paths)
        render_benchmark(self.console, summary)

    def _memory(self, args: list[str]) -> None:
        if len(args) < 2 or args[0] not in {"--keyword", "--tag", "--semantic"}:
            self.console.print("[red]Usage:[/red] /memory --keyword|--tag|--semantic <query>")
            return
        query = join_prompt(args[1:])
        config = AgentMeshConfig.from_project_root(self.paths.root)
        store = SQLiteMemoryStore(
            self.paths,
            StateStore(self.paths),
            create_embedding_encoder(config.embedding),
        )
        if args[0] == "--keyword":
            results = store.keyword_search(query)
        elif args[0] == "--tag":
            results = store.tag_search(query)
        else:
            results = store.semantic_search(query)
        render_memory(self.console, results)

    def _trace(self, args: list[str]) -> None:
        limit = int(args[0]) if args else 10
        rows = read_jsonl(self.paths.text_trace) + read_jsonl(self.paths.protocol_trace)
        for item in rows[-limit:]:
            self.console.print(item)

    def _config(self) -> None:
        config = AgentMeshConfig.from_project_root(self.paths.root)
        render_config(
            self.console,
            {
                "root": self.paths.root,
                "llm_configured": config.llm.configured,
                "llm_base_url": config.llm.base_url or "",
                "llm_model": config.llm.model or "",
                "embedding_provider": config.embedding.provider,
                "embedding_base_url": config.embedding.base_url or "",
                "embedding_model": config.embedding.model,
                "embedding_dimensions": config.embedding.dimensions,
                "rust_available": rust_available(),
            },
        )


def run_shell(
    paths: RuntimePaths,
    *,
    console: Console | None = None,
    input_func: InputFunc | None = None,
) -> None:
    ShellSession(paths=paths, console=console).run(input_func=input_func)


def _suite_path(root: Path, value: str) -> Path:
    aliases = {
        "standard": root / "examples" / "benchmarks" / "continuous_tasks.yaml",
        "long": root / "examples" / "benchmarks" / "long_context_tasks.yaml",
    }
    path = aliases.get(value, Path(value))
    if not path.is_absolute():
        path = root / path
    return path
