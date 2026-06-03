from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.prompt import Prompt

from agentmesh.chat.session import ChatTurn, run_chat_turn
from agentmesh.eval.benchmark import run_benchmark
from agentmesh.eval.report import generate_report
from agentmesh.memory.sqlite_store import SQLiteMemoryStore
from agentmesh.modes.protocol_mode import run_protocol_mode
from agentmesh.modes.text_mode import run_text_mode
from agentmesh.runtime.registry import RuntimeContext
from agentmesh.state.embedding import HashEmbeddingEncoder
from agentmesh.state.store import StateStore
from agentmesh.storage.jsonl import read_jsonl
from agentmesh.storage.paths import RuntimePaths

app = typer.Typer(help="AgentMesh Runtime CLI")
memory_app = typer.Typer(help="Shared memory commands")
trace_app = typer.Typer(help="Trace commands")
app.add_typer(memory_app, name="memory")
app.add_typer(trace_app, name="trace")
console = Console()
DEFAULT_ROOT = Path(".")
DEFAULT_RUN = Path("runs/latest")


@app.command()
def init(root: Annotated[Path, typer.Option(help="Project root.")] = DEFAULT_ROOT) -> None:
    paths = RuntimePaths(root=root)
    paths.ensure()
    console.print(f"Initialized AgentMesh runtime at {paths.latest_run}")


@app.command()
def run(
    mode: Annotated[str, typer.Option(help="text or protocol")],
    task: Annotated[Path, typer.Option(help="Task file path")],
    root: Annotated[Path, typer.Option(help="Project root.")] = DEFAULT_ROOT,
) -> None:
    paths = RuntimePaths(root=root)
    try:
        if mode == "protocol":
            result = run_protocol_mode(task, paths)
        else:
            result = run_text_mode(task, paths)
    except Exception as exc:
        console.print(f"[red]agentmesh run failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(result.model_dump())


@app.command()
def benchmark(
    suite: Annotated[Path, typer.Option(help="Benchmark suite YAML")],
    root: Annotated[Path, typer.Option(help="Project root.")] = DEFAULT_ROOT,
) -> None:
    paths = RuntimePaths(root=root)
    try:
        summary = run_benchmark(suite, paths)
    except Exception as exc:
        console.print(f"[red]agentmesh benchmark failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(summary.model_dump())


@app.command()
def chat(
    message: Annotated[str | None, typer.Option(help="Single-turn message.")] = None,
    root: Annotated[Path, typer.Option(help="Project root.")] = DEFAULT_ROOT,
) -> None:
    paths = RuntimePaths(root=root)
    context = RuntimeContext.from_paths(paths=paths, trace_id="trace-chat")
    history: list[ChatTurn] = []
    if message:
        turn = run_chat_turn(message, context=context, history=history)
        console.print(turn.assistant)
        return
    console.print("[bold]AgentMesh interactive chat[/bold] (type /exit to quit)")
    while True:
        user_input = Prompt.ask("you")
        if user_input.strip() in {"/exit", "/quit"}:
            return
        turn = run_chat_turn(user_input, context=context, history=history)
        history.append(turn)
        console.print(f"[bold]agentmesh[/bold]: {turn.assistant}")


@app.command()
def report(
    run: Annotated[Path, typer.Option(help="Run directory")] = DEFAULT_RUN,
    root: Annotated[Path, typer.Option(help="Project root.")] = DEFAULT_ROOT,
) -> None:
    paths = RuntimePaths(root=root)
    if run != Path("runs/latest"):
        paths = RuntimePaths(root=run.parent.parent if run.name == "latest" else root)
    path = generate_report(paths)
    console.print(f"Report written to {path}")


@memory_app.command("search")
def memory_search(
    keyword: Annotated[str | None, typer.Option(help="Keyword query")] = None,
    tag: Annotated[str | None, typer.Option(help="Tag query")] = None,
    semantic: Annotated[str | None, typer.Option(help="Semantic query")] = None,
    root: Annotated[Path, typer.Option(help="Project root.")] = DEFAULT_ROOT,
) -> None:
    paths = RuntimePaths(root=root)
    state_store = StateStore(paths)
    store = SQLiteMemoryStore(paths, state_store, HashEmbeddingEncoder())
    if keyword:
        results = store.keyword_search(keyword)
    elif tag:
        results = store.tag_search(tag)
    elif semantic:
        results = store.semantic_search(semantic)
    else:
        console.print("[red]Provide --keyword, --tag, or --semantic.[/red]")
        raise typer.Exit(code=1)
    for item in results:
        console.print(item.model_dump())


@trace_app.command("show")
def trace_show(
    root: Annotated[Path, typer.Option(help="Project root.")] = DEFAULT_ROOT,
) -> None:
    paths = RuntimePaths(root=root)
    for path in [paths.text_trace, paths.protocol_trace]:
        for item in read_jsonl(path):
            console.print(item)
