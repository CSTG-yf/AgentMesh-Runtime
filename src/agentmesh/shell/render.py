from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentmesh.eval.benchmark import BenchmarkSummary
from agentmesh.eval.compare import CompareAgentOutput, CompareProgressEvent, CompareSummary
from agentmesh.memory.schema import MemoryUnit
from agentmesh.memory.search import MemorySearchResult

_AGENT_DISPLAY_ORDER = ["planner", "retriever", "executor", "summarizer"]


def render_help(console: Console) -> None:
    table = Table(title="AgentMesh shell commands")
    table.add_column("Command")
    table.add_column("Usage")
    table.add_column("Purpose")
    rows = [
        ("text", "直接输入消息", "等同 /ask，走 Protocol Mode 交互式 LLM Agent"),
        ("/compare", "/compare [--no-llm] <task>", "流式展示双模式 Agent 输出和指标"),
        ("/ask", "/ask <message>", "流式展示 Protocol Agent 交互"),
        ("/run", "/run text|protocol <task-file>", "运行指定模式"),
        ("/benchmark", "/benchmark standard|long|<suite.yaml>", "运行评测套件"),
        ("/memory", "/memory --keyword|--tag|--semantic <query>", "检索共享记忆"),
        ("/trace", "/trace [limit]", "查看最近通信 trace"),
        ("/report", "/report", "生成实验报告"),
        ("/config", "/config", "查看当前 LLM、Embedding、Rust 配置"),
        ("/exit", "/exit", "退出 shell"),
    ]
    for row in rows:
        table.add_row(*row)
    console.print(table)


def render_compare(console: Console, summary: CompareSummary) -> None:
    console.print(f"Task file: {summary.task_path}")
    answers = Table(title="Final Answers", expand=True, show_lines=True)
    answers.add_column("Text Mode", ratio=1, overflow="fold")
    answers.add_column("Protocol Mode", ratio=1, overflow="fold")
    answers.add_row(
        Text(_clip(summary.text.answer, 6000), overflow="fold"),
        Text(_clip(summary.protocol.answer, 6000), overflow="fold"),
    )
    console.print(answers)

    outputs = Table(title="Agent Outputs", expand=True, show_lines=True)
    outputs.add_column("Text Mode agents", ratio=1, overflow="fold")
    outputs.add_column("Protocol Mode agents", ratio=1, overflow="fold")
    for text_output, protocol_output in _aligned_agent_output_rows(
        summary.text_agent_outputs,
        summary.protocol_agent_outputs,
    ):
        outputs.add_row(
            Text(_format_agent_output(text_output), overflow="fold"),
            Text(_format_agent_output(protocol_output), overflow="fold"),
        )
    console.print(outputs)

    table = Table(title="Text Mode vs Protocol Mode")
    table.add_column("Metric")
    table.add_column("Text")
    table.add_column("Protocol")
    table.add_row(
        "messages",
        str(summary.text.metrics.message_count),
        str(summary.protocol.metrics.message_count),
    )
    table.add_row(
        "estimated tokens",
        str(summary.text.metrics.estimated_tokens),
        str(summary.protocol.metrics.estimated_tokens),
    )
    table.add_row(
        "wire bytes",
        str(summary.text.metrics.wire_bytes),
        str(summary.protocol.metrics.wire_bytes),
    )
    table.add_row(
        "latency ms",
        str(summary.text.metrics.latency_ms),
        str(summary.protocol.metrics.latency_ms),
    )
    table.add_row(
        "state transfers",
        str(summary.text.metrics.state_transfer_count),
        str(summary.protocol.metrics.state_transfer_count),
    )
    table.add_row(
        "memory hit rate",
        _rate_text(summary.text.metrics.memory_hit_rate),
        _rate_text(summary.protocol.metrics.memory_hit_rate),
    )
    table.add_row(
        "memory queries",
        str(summary.text.metrics.memory_query_count),
        str(summary.protocol.metrics.memory_query_count),
    )
    table.add_row(
        "memory query hits",
        str(summary.text.metrics.memory_query_hit_count),
        str(
            summary.protocol.metrics.memory_query_hit_count
            or min(
                summary.protocol.metrics.memory_hit_count,
                summary.protocol.metrics.memory_query_count,
            )
        ),
    )
    table.add_row(
        "memory reused units",
        str(summary.text.metrics.memory_reused_unit_count),
        str(summary.protocol.metrics.memory_reused_unit_count),
    )
    table.add_row(
        "memory reused/query",
        _avg_reused_units_text(summary.text.metrics),
        _avg_reused_units_text(summary.protocol.metrics),
    )
    table.add_row(
        "memory avg score",
        _rate_text(summary.text.metrics.memory_avg_score),
        _rate_text(summary.protocol.metrics.memory_avg_score),
    )
    table.add_row(
        "memory avg semantic",
        _rate_text(summary.text.metrics.memory_avg_semantic_similarity),
        _rate_text(summary.protocol.metrics.memory_avg_semantic_similarity),
    )
    table.add_row(
        "memory avg tag overlap",
        _rate_text(summary.text.metrics.memory_avg_tag_overlap_score),
        _rate_text(summary.protocol.metrics.memory_avg_tag_overlap_score),
    )
    table.add_row("token_saving_rate", _rate_text(summary.token_saving_rate), "")
    table.add_row(
        "wire_bytes_reduction_rate",
        _rate_text(summary.wire_bytes_reduction_rate),
        "",
    )
    table.add_row("latency_reduction_rate", _rate_text(summary.latency_reduction_rate), "")
    console.print(table)


def render_compare_progress(console: Console, event: CompareProgressEvent) -> None:
    if event.kind == "mode_start":
        console.print(f"[cyan]START[/cyan] {event.message}")
        return
    if event.kind == "mode_complete":
        console.print(f"[green]DONE[/green] {event.message}")
        return
    if event.agent_output is None:
        return
    output = event.agent_output
    border_style = "blue" if output.mode == "text" else "magenta"
    title = f"{output.mode.title()} Mode step {output.step}: {output.agent}"
    console.print(
        Panel(
            Text(_format_agent_output(output, include_input=False)),
            title=title,
            border_style=border_style,
        )
    )


def render_benchmark(console: Console, summary: BenchmarkSummary) -> None:
    console.print(summary.model_dump())


def render_memory(console: Console, results: list[MemoryUnit | MemorySearchResult]) -> None:
    if not results:
        console.print("[yellow]No memory matched.[/yellow]")
        return
    table = Table(title="Shared memory")
    table.add_column("ID")
    table.add_column("Source")
    table.add_column("Score")
    table.add_column("Topic")
    table.add_column("Summary")
    table.add_column("Content")
    table.add_column("Tags")
    for item in results:
        unit = item.memory if isinstance(item, MemorySearchResult) else item
        score = f"{item.score:.4f}" if isinstance(item, MemorySearchResult) else ""
        table.add_row(
            unit.memory_id,
            unit.source_agent,
            score,
            unit.task_topic,
            unit.summary,
            unit.content or unit.summary,
            ", ".join(unit.tags),
        )
    console.print(table)


def render_config(console: Console, config: dict[str, Any]) -> None:
    table = Table(title="AgentMesh config")
    table.add_column("Key")
    table.add_column("Value")
    for key, value in config.items():
        table.add_row(key, str(value))
    console.print(table)


def render_path(console: Console, label: str, path: Path) -> None:
    console.print(f"{label}: {path}")


def _format_agent_output(
    output: CompareAgentOutput | None,
    *,
    include_input: bool = True,
) -> str:
    if output is None:
        return ""
    lines = [f"[{output.step}] {output.agent}"]
    if output.action:
        lines.append(f"action: {output.action}")
    if output.state_refs_in:
        lines.append(f"state_refs_in: {', '.join(output.state_refs_in)}")
    if output.state_refs_out:
        lines.append(f"state_refs_out: {', '.join(output.state_refs_out)}")
    if include_input and output.input_summary:
        lines.append("input:")
        lines.append(_clip(output.input_summary, 1000))
    lines.append("output:")
    lines.append(_clip(output.output, 4000))
    return "\n".join(lines)


def _aligned_agent_output_rows(
    text_outputs: list[CompareAgentOutput],
    protocol_outputs: list[CompareAgentOutput],
) -> list[tuple[CompareAgentOutput | None, CompareAgentOutput | None]]:
    text_by_agent = _outputs_by_agent(text_outputs)
    protocol_by_agent = _outputs_by_agent(protocol_outputs)
    ordered_agents = list(_AGENT_DISPLAY_ORDER)
    for output in [*text_outputs, *protocol_outputs]:
        if output.agent not in ordered_agents:
            ordered_agents.append(output.agent)

    rows: list[tuple[CompareAgentOutput | None, CompareAgentOutput | None]] = []
    for agent in ordered_agents:
        text_items = text_by_agent.get(agent, [])
        protocol_items = protocol_by_agent.get(agent, [])
        count = max(len(text_items), len(protocol_items))
        for index in range(count):
            rows.append(
                (
                    text_items[index] if index < len(text_items) else None,
                    protocol_items[index] if index < len(protocol_items) else None,
                )
            )
    if not rows:
        rows.append((None, None))
    return rows


def _outputs_by_agent(
    outputs: list[CompareAgentOutput],
) -> dict[str, list[CompareAgentOutput]]:
    grouped: dict[str, list[CompareAgentOutput]] = {}
    for output in sorted(outputs, key=lambda item: item.step):
        grouped.setdefault(output.agent, []).append(output)
    return grouped


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "\n... [truncated]"


def _rate_text(value: float) -> str:
    return f"{value:.4f}"


def _avg_reused_units_text(metrics: Any) -> str:
    if metrics.memory_query_count == 0:
        return "0.0000"
    return f"{metrics.memory_reused_unit_count / metrics.memory_query_count:.4f}"
