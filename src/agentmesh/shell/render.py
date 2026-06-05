from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from agentmesh.eval.benchmark import BenchmarkSummary
from agentmesh.eval.compare import CompareSummary
from agentmesh.memory.schema import MemoryUnit


def render_help(console: Console) -> None:
    table = Table(title="AgentMesh shell commands")
    table.add_column("Command")
    table.add_column("Usage")
    table.add_column("Purpose")
    rows = [
        ("text", "直接输入任务", "默认执行文本模式与结构化协议模式对比"),
        ("/compare", "/compare [--llm] <task>", "同任务可复现实验对比"),
        ("/ask", "/ask <message>", "调用交互式 LLM Agent"),
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
        str(summary.text.metrics.memory_hit_rate),
        str(summary.protocol.metrics.memory_hit_rate),
    )
    console.print(table)
    console.print(
        {
            "task_path": summary.task_path,
            "token_saving_rate": summary.token_saving_rate,
            "wire_bytes_reduction_rate": summary.wire_bytes_reduction_rate,
            "latency_reduction_rate": summary.latency_reduction_rate,
        }
    )


def render_benchmark(console: Console, summary: BenchmarkSummary) -> None:
    console.print(summary.model_dump())


def render_memory(console: Console, results: list[MemoryUnit]) -> None:
    if not results:
        console.print("[yellow]No memory matched.[/yellow]")
        return
    table = Table(title="Shared memory")
    table.add_column("ID")
    table.add_column("Source")
    table.add_column("Topic")
    table.add_column("Summary")
    table.add_column("Tags")
    for unit in results:
        table.add_row(
            unit.memory_id,
            unit.source_agent,
            unit.task_topic,
            unit.summary,
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

