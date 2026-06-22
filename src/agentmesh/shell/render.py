from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentmesh.eval.benchmark import BenchmarkProgressEvent, BenchmarkSummary
from agentmesh.eval.compare import CompareAgentOutput, CompareProgressEvent, CompareSummary
from agentmesh.memory.schema import MemoryUnit
from agentmesh.memory.search import MemorySearchResult
from agentmesh.storage.paths import RuntimePaths

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
        ("/benchmark", "/benchmark [--no-llm] standard", "运行标准评测套件，覆盖常规连续任务"),
        ("/benchmark", "/benchmark [--no-llm] long", "运行长上下文评测套件，观察状态复用和长任务开销"),
        ("/benchmark", "/benchmark [--no-llm] showcase", "运行展示型 A/B/C 套件，包含 warm reuse 与 cold baseline"),
        ("/benchmark", "/benchmark [--no-llm] <suite.yaml>", "运行自定义 YAML 评测套件"),
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

    table = Table(title="Text Mode vs Protocol Mode - 赛题关键指标", show_lines=True)
    table.add_column("Metric / 指标说明", ratio=2)
    table.add_column("Text", ratio=1)
    table.add_column("Protocol", ratio=1)

    _add_section(table, "通信效率（评分权重 25%）")
    _add_metric(table, "messages（消息数）", summary.text.metrics.message_count,
                summary.protocol.metrics.message_count)
    _add_metric(table, "agent I/O tokens（Agent 通信文本 token）",
                summary.text.metrics.agent_io_tokens,
                summary.protocol.metrics.agent_io_tokens)
    _add_metric(table, "agent I/O bytes（Agent 通信字节）",
                summary.text.metrics.agent_io_bytes,
                summary.protocol.metrics.agent_io_bytes)
    _add_metric(table, "avg tokens per message（每消息平均 token）",
                _rate_text(summary.text.metrics.per_msg_avg_tokens),
                _rate_text(summary.protocol.metrics.per_msg_avg_tokens))
    _add_metric(table, "wire bytes（传输字节数）", summary.text.metrics.wire_bytes,
                summary.protocol.metrics.wire_bytes)
    _add_metric(table, "protocol total bytes（信封 + payload + state）", "",
                summary.protocol.metrics.protocol_total_bytes)
    table.add_row("[bold]token_saving_rate（token 节省率）[/bold]",
                  f"[bold]{_rate_text(summary.token_saving_rate)}[/bold]", "")
    table.add_row("[bold]wire_bytes_reduction_rate（端到端总字节降低率）[/bold]",
                  f"[bold]{_rate_text(summary.wire_bytes_reduction_rate)}[/bold]", "")
    table.add_row("[bold]fair_wire_reduction_rate（仅 wire 层降低率）[/bold]",
                  f"[bold]{_rate_text(summary.fair_wire_reduction_rate)}[/bold]", "")
    table.add_row("[bold]agent_io_bytes_reduction_rate（Agent I/O 字节降低率）[/bold]",
                  f"[bold]{_rate_text(summary.agent_io_bytes_reduction_rate)}[/bold]", "")

    _add_section(table, "状态传递（评分权重 20%）")
    _add_metric(table, "state transfers（状态传递次数）",
                summary.text.metrics.state_transfer_count,
                summary.protocol.metrics.state_transfer_count)
    _add_metric(table, "state transfer bytes（状态传递字节数）",
                summary.text.metrics.state_transfer_bytes,
                summary.protocol.metrics.state_transfer_bytes)

    _add_section(table, "记忆复用（评分权重 20%）")
    _add_metric(table, "memory queries（记忆查询次数）",
                summary.text.metrics.memory_query_count,
                summary.protocol.metrics.memory_query_count)
    _add_metric(table, "memory hit rate（记忆命中率）",
                _rate_text(summary.text.metrics.memory_hit_rate),
                _rate_text(summary.protocol.metrics.memory_hit_rate))
    _add_metric(table, "memory reused units（复用记忆条数）",
                summary.text.metrics.memory_reused_unit_count,
                summary.protocol.metrics.memory_reused_unit_count)
    _add_metric(table, "memory evidence bytes（注入给 Agent 的记忆证据字节）",
                summary.text.metrics.memory_evidence_bytes,
                summary.protocol.metrics.memory_evidence_bytes)
    _add_metric(table, "memory evidence/query（每次查询注入证据字节）", "-",
                _rate_text(summary.memory_avg_evidence_bytes_per_query))
    _add_metric(table, "memory avg semantic（平均语义相似度）",
                _rate_text(summary.text.metrics.memory_avg_semantic_similarity),
                _rate_text(summary.protocol.metrics.memory_avg_semantic_similarity))
    _add_metric(table, "quality preservation rate（回答质量保持率）",
                _rate_text(1.0), _rate_text(summary.quality_preservation_rate))

    _add_section(table, "系统完整性（评分权重 20%）")
    protocol_route = (
        " -> ".join(summary.protocol.metrics.dynamic_route)
        if summary.protocol.metrics.dynamic_route
        else "-"
    )
    _add_metric(table, "agent route（执行链路）",
                " -> ".join(summary.text.metrics.dynamic_route), protocol_route)
    _add_metric(table, "selected agents（已启用 Agent）",
                ", ".join(summary.text.metrics.selected_agents),
                ", ".join(summary.protocol.metrics.selected_agents))
    _add_metric(table, "latency ms（总耗时）", summary.text.metrics.latency_ms,
                summary.protocol.metrics.latency_ms)
    _add_metric(table, "stage breakdown（各阶段耗时）", "-", "[see below]")
    table.add_row("[bold]latency_reduction_rate（耗时降低率）[/bold]",
                  f"[bold]{_rate_text(summary.latency_reduction_rate)}[/bold]", "")

    console.print(table)
    _render_latency_breakdown(console, summary)


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


def _render_latency_breakdown(console: Console, summary: CompareSummary) -> None:
    stages = summary.protocol.metrics.stage_latency_ms
    if not stages:
        return
    console.print("Protocol Mode Latency Breakdown")
    table = Table(title="Protocol Mode Latency Breakdown")
    table.add_column("Stage")
    table.add_column("ms", justify="right")
    table.add_column("%", justify="right")
    total = summary.protocol.metrics.latency_ms or sum(stages.values()) or 1
    for stage, ms in sorted(stages.items(), key=lambda item: item[1], reverse=True):
        table.add_row(stage, str(ms), f"{ms / total * 100:.1f}%")
    console.print(table)


def render_benchmark(console: Console, summary: BenchmarkSummary) -> None:
    table = Table(title=f"Benchmark: {summary.suite_name} ({summary.total_runs} runs)", show_lines=True)
    table.add_column("Metric / 指标说明", ratio=2)
    table.add_column("Text", ratio=1)
    table.add_column("Protocol", ratio=1)

    _add_section(table, "通信效率（评分权重 25%）")
    _add_metric(table, "agent I/O tokens（Agent 通信文本 token）",
                summary.text_agent_io_tokens, summary.protocol_agent_io_tokens)
    _add_metric(table, "agent I/O bytes（Agent 通信字节）",
                summary.text_agent_io_bytes, summary.protocol_agent_io_bytes)
    _add_metric(table, "avg tokens per message（每消息平均 token）",
                f"{summary.text_per_msg_avg_tokens:.2f}" if summary.text_per_msg_avg_tokens else "-",
                f"{summary.protocol_per_msg_avg_tokens:.2f}" if summary.protocol_per_msg_avg_tokens else "-")
    _add_metric(table, "wire bytes（传输字节）",
                summary.text_wire_bytes, summary.protocol_wire_bytes)
    _add_metric(table, "total protocol bytes（信封 + payload + state）", "",
                summary.protocol_total_bytes)
    table.add_row("[bold]token saving rate（token 节省率）[/bold]",
                  f"[bold]{summary.token_saving_rate:.4f}[/bold]", "")
    table.add_row("[bold]wire bytes reduction rate（端到端总字节降低率）[/bold]",
                  f"[bold]{summary.wire_bytes_reduction_rate:.4f}[/bold]", "")
    table.add_row("[bold]fair wire reduction rate（仅 wire 层降低率）[/bold]",
                  f"[bold]{summary.fair_wire_reduction_rate:.4f}[/bold]", "")
    table.add_row("[bold]agent I/O bytes reduction rate（Agent 通信字节降低率）[/bold]",
                  f"[bold]{summary.agent_io_bytes_reduction_rate:.4f}[/bold]", "")

    _add_section(table, "状态传递（评分权重 20%）")
    _add_metric(table, "state transfer bytes（状态传递总字节）", "-",
                summary.protocol_state_payload_bytes)
    _add_metric(table, "state SHM transfers（共享内存传递次数）", "-",
                summary.state_shm_transfer_count)

    _add_section(table, "记忆复用（评分权重 20%）")
    _add_metric(table, "memory hit rate（记忆命中率）", "-",
                f"{summary.memory_hit_rate:.4f}")
    _add_metric(table, "memory reused units（复用记忆条数）", "-",
                summary.memory_reused_unit_count)
    _add_metric(table, "memory evidence bytes（注入给 Agent 的记忆证据字节）", "-",
                summary.memory_evidence_bytes)
    _add_metric(table, "memory evidence/query（每次查询注入证据字节）", "-",
                f"{summary.memory_avg_evidence_bytes_per_query:.2f}")
    _add_metric(table, "memory avg semantic（平均语义相似度）", "-",
                f"{summary.memory_avg_semantic_similarity:.4f}"
                if summary.memory_avg_semantic_similarity else "-")
    table.add_row("[bold]quality preservation rate（质量保持率）[/bold]",
                  "1.0000", f"[bold]{summary.quality_preservation_rate:.4f}[/bold]")

    _add_section(table, "系统完整性（评分权重 20%）")
    _add_metric(table, "feedback rounds（反馈优化轮次）", "-",
                summary.feedback_round_count)
    _add_metric(table, "Rust core enabled runs（Rust 加速运行数）", "-",
                summary.rust_core_enabled_runs)
    table.add_row("[bold]latency reduction rate（耗时降低率）[/bold]",
                  f"[bold]{summary.latency_reduction_rate:.4f}[/bold]", "")
    console.print(table)


def render_benchmark_progress(console: Console, event: BenchmarkProgressEvent) -> None:
    if event.phase == "suite_start":
        mode = "真实 LLM" if event.use_llm else "离线 no-llm"
        console.print(
            "[bold blue]benchmark started[/bold blue]: "
            f"{event.suite_name}，共 {event.total_tasks} 条任务 / "
            f"{event.total_stages} 个阶段，模式：{mode}",
            soft_wrap=True,
        )
        return
    if event.phase == "suite_complete":
        console.print(
            "[bold green]benchmark completed[/bold green]: "
            f"{event.suite_name}，已完成 {event.total_tasks} 条任务 / "
            f"{event.total_stages} 个阶段",
            soft_wrap=True,
        )
        return
    if event.phase == "cold_start":
        label = event.task_id or f"task-{event.current_task}"
        console.print(
            "[yellow]memory reset[/yellow]: "
            f"任务 {event.current_task}/{event.total_tasks} id={label} "
            "启用 cold_start，已清空本次 run 的记忆库",
            soft_wrap=True,
        )
        return
    if event.phase != "mode_complete":
        return
    label = event.task_id or f"task-{event.current_task}"
    topic = f"，topic={event.topic}" if event.topic else ""
    group = f"，group={event.group}" if event.group else ""
    depends_on = f"，depends_on={','.join(event.depends_on)}" if event.depends_on else ""
    tags = f"，tags={','.join(event.tags)}" if event.tags else ""
    cold_start = "，cold_start=true" if event.cold_start else ""
    console.print(
        "[green]stage completed[/green]: "
        f"[{event.current_stage}/{event.total_stages}] "
        f"任务 {event.current_task}/{event.total_tasks} "
        f"round {event.repeat_index}/{event.repeat_total} "
        f"id={label}{group}{topic}{depends_on}{tags}{cold_start}，"
        f"mode={event.mode}，latency={event.latency_ms}ms，"
        f"tokens={event.tokens}，bytes={event.bytes}，trace={event.trace_id}",
        soft_wrap=True,
    )


def render_benchmark_artifacts(
    console: Console,
    paths: RuntimePaths,
    *,
    suite_name: str,
    report_path: Path | None = None,
) -> None:
    console.print("[bold]Benchmark output files[/bold]")
    console.print(
        f"summary csv: {paths.benchmark_suite_summary(suite_name).resolve()}",
        soft_wrap=True,
    )
    console.print(
        f"detail jsonl: {paths.benchmark_suite_detail(suite_name).resolve()}",
        soft_wrap=True,
    )
    if report_path is not None:
        console.print(f"experiment report: {report_path.resolve()}", soft_wrap=True)


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


def _add_section(table: Table, title: str) -> None:
    table.add_row(f"[bold cyan]{title}[/bold cyan]", "", "")


def _add_metric(table: Table, label: str, text_val: object, protocol_val: object) -> None:
    table.add_row(
        label,
        str(text_val) if text_val is not None and str(text_val) else "-",
        str(protocol_val) if protocol_val is not None and str(protocol_val) else "-",
    )
