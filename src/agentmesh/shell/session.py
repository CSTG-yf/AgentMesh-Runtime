from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from rich.console import Console
from rich.prompt import Prompt

from agentmesh.chat.session import ChatTurn
from agentmesh.config import AgentMeshConfig
from agentmesh.core import rust_available
from agentmesh.eval.benchmark import run_benchmark
from agentmesh.eval.compare import run_prompt_compare, run_protocol_with_progress
from agentmesh.eval.dashboard import generate_dashboard
from agentmesh.eval.llm_comparison import compare_llm_artifacts
from agentmesh.eval.report import generate_report
from agentmesh.memory.hybrid_store import HybridMemoryStore
from agentmesh.memory.maintenance import MemoryMaintenanceConfig, MemoryMaintenanceWorker
from agentmesh.memory.schema import MemoryUnit
from agentmesh.memory.search import MemorySearchResult
from agentmesh.modes.protocol_mode import run_protocol_mode
from agentmesh.modes.text_mode import run_text_mode
from agentmesh.runtime.registry import RuntimeContext
from agentmesh.shell.commands import ParsedShellCommand, join_prompt, parse_shell_line
from agentmesh.shell.render import (
    render_benchmark,
    render_benchmark_artifacts,
    render_benchmark_progress,
    render_compare,
    render_compare_progress,
    render_config,
    render_help,
    render_llm_comparison,
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
        self.state_store = StateStore(paths)
        self.embedding_encoder = create_embedding_encoder(self.context.config.embedding)
        self.maintenance = MemoryMaintenanceWorker(
            paths=paths,
            state_store=self.state_store,
            encoder=self.embedding_encoder,
            config=MemoryMaintenanceConfig(
                enabled=self.context.config.memory.maintenance_enabled,
                interval_seconds=self.context.config.memory.maintenance_interval_seconds,
                max_items=self.context.config.memory.maintenance_max_items,
            ),
        )

    def run(self, input_func: InputFunc | None = None) -> None:
        prompt = input_func or (lambda label: Prompt.ask(label))
        self.console.print("[bold]AgentMesh shell[/bold] (输入 /help 查看命令，/exit 退出)")
        self.maintenance.start()
        try:
            while True:
                try:
                    line = prompt("agentmesh")
                except (EOFError, KeyboardInterrupt):
                    self.console.print()
                    return
                if not self.handle_line(line):
                    return
        finally:
            self.maintenance.stop()

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
        if name == "benchmark-compare":
            self._benchmark_compare(command.args)
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
        if name == "dashboard":
            self._dashboard(command.args)
            return True
        if name == "config":
            self._config()
            return True
        self.console.print(f"[yellow]Unknown command:[/yellow] /{name}. Use /help.")
        return True

    def _compare(self, args: list[str]) -> None:
        use_llm = True
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
            self.console.print("[red]Usage:[/red] /compare [--no-llm] <task>")
            return
        summary = run_prompt_compare(
            prompt,
            paths=self.paths,
            use_llm=use_llm,
            progress_callback=lambda event: render_compare_progress(self.console, event),
        )
        render_compare(self.console, summary)

    def _ask(self, args: list[str]) -> None:
        message = join_prompt(args)
        if not message:
            self.console.print("[red]Usage:[/red] /ask <message>")
            return
        task_path = self._write_ask_task(message)
        result = run_protocol_with_progress(
            task_path=task_path,
            paths=self.paths,
            use_llm=True,
            progress_callback=lambda event: render_compare_progress(self.console, event),
            runner=run_protocol_mode,
        )
        turn = ChatTurn(user=message, assistant=result.answer)
        self.history.append(turn)
        self.console.print("[bold]agentmesh[/bold]: ", end="")
        self.console.print(turn.assistant, markup=False)

    def _write_ask_task(self, message: str) -> Path:
        self.paths.ensure()
        task_dir = self.paths.latest_run / "user_tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        task_path = task_dir / f"ask-{uuid4().hex[:8]}.txt"
        task_path.write_text(
            _protocol_ask_task_text(message=message, history=self.history),
            encoding="utf-8",
        )
        return task_path

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
        use_llm = True
        if "--no-llm" in args:
            use_llm = False
            args = [arg for arg in args if arg != "--no-llm"]
        if len(args) != 1:
            self.console.print(
                "[red]Usage:[/red] /benchmark [--no-llm] standard|long|showcase|<suite.yaml>"
            )
            return
        suite = _suite_path(self.paths.root, args[0])
        summary = run_benchmark(
            suite,
            self.paths,
            use_llm=use_llm,
            progress_callback=lambda event: render_benchmark_progress(self.console, event),
        )
        render_benchmark(self.console, summary)
        report_path = generate_report(
            self.paths,
            suite_name=summary.suite_name,
            track=summary.track.value,
        )
        render_benchmark_artifacts(
            self.console,
            self.paths,
            suite_name=summary.suite_name,
            track=summary.track.value,
            report_path=report_path,
        )

    def _benchmark_compare(self, args: list[str]) -> None:
        if len(args) != 3:
            self.console.print(
                "[red]Usage:[/red] "
                "/benchmark-compare <suite> <baseline> <candidate>"
            )
            return
        suite_name, baseline_variant, candidate_variant = args
        result = compare_llm_artifacts(
            self.paths,
            suite_name=suite_name,
            baseline_variant=baseline_variant,
            candidate_variant=candidate_variant,
        )
        render_llm_comparison(self.console, result)

    def _dashboard(self, args: list[str]) -> None:
        if len(args) > 1:
            self.console.print("[red]Usage:[/red] /dashboard [output.html]")
            return
        output = Path(args[0]) if args else None
        if output is not None and not output.is_absolute():
            output = self.paths.root / output
        dashboard_path = generate_dashboard(self.paths.root, output)
        render_path(self.console, "Dashboard written", dashboard_path)

    def _memory(self, args: list[str]) -> None:
        if len(args) < 2 or args[0] not in {"--keyword", "--tag", "--semantic"}:
            self.console.print("[red]Usage:[/red] /memory --keyword|--tag|--semantic <query>")
            return
        query = join_prompt(args[1:])
        store = HybridMemoryStore(
            paths=self.paths,
            state_store=self.state_store,
            encoder=self.embedding_encoder,
        )
        results: list[MemoryUnit | MemorySearchResult] = []
        if args[0] == "--keyword":
            results.extend(store.keyword_search(query))
        elif args[0] == "--tag":
            results.extend(store.tag_search(query))
        else:
            results.extend(store.semantic_search_with_scores(query))
        render_memory(self.console, results)

    def _trace(self, args: list[str]) -> None:
        limit = int(args[0]) if args else 10
        rows = (
            read_jsonl(self.paths.text_trace)
            + read_jsonl(self.paths.text_agent_io)
            + read_jsonl(self.paths.protocol_trace)
            + read_jsonl(self.paths.protocol_agent_io)
        )
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
                "memory_maintenance_enabled": config.memory.maintenance_enabled,
                "global_memory_db": self.paths.global_memory_db,
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


def _protocol_ask_task_text(*, message: str, history: list[ChatTurn]) -> str:
    lines = [
        "You are answering inside an interactive AgentMesh shell session.",
        "Use the conversation history as context, then answer the current user request.",
        "",
    ]
    if history:
        lines.append("Conversation history:")
        for turn in history:
            lines.append(f"User: {turn.user}")
            lines.append(f"Assistant: {turn.assistant}")
        lines.append("")
    lines.extend(
        [
            "Current user request:",
            f"User: {message}",
            "",
            "Route this request through the AgentMesh protocol workflow.",
            "Produce a helpful final answer.",
        ]
    )
    return "\n".join(lines)


def _suite_path(root: Path, value: str) -> Path:
    aliases = {
        "standard": root / "examples" / "benchmarks" / "continuous_tasks.yaml",
        "long": root / "examples" / "benchmarks" / "long_context_tasks.yaml",
        "showcase": root / "examples" / "benchmarks" / "showcase_benchmark.yaml",
    }
    path = aliases.get(value, Path(value))
    if not path.is_absolute():
        path = root / path
    return path
