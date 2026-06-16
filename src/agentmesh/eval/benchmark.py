import csv
import shutil
from pathlib import Path
from collections.abc import Callable
from typing import Any

import yaml
from pydantic import BaseModel

from agentmesh.errors import BenchmarkConfigError
from agentmesh.eval.metrics import TOKEN_ESTIMATOR, RunMetrics
from agentmesh.modes.protocol_mode import run_protocol_mode
from agentmesh.modes.text_mode import run_text_mode
from agentmesh.storage.jsonl import append_jsonl
from agentmesh.storage.paths import RuntimePaths


class BenchmarkSummary(BaseModel):
    suite_name: str
    total_runs: int
    token_estimator: str
    token_saving_rate: float
    wire_bytes_reduction_rate: float
    text_agent_io_tokens: int = 0
    protocol_agent_io_tokens: int = 0
    text_per_msg_avg_tokens: float = 0.0
    protocol_per_msg_avg_tokens: float = 0.0
    text_agent_io_bytes: int = 0
    protocol_total_bytes: int = 0
    text_wire_bytes: int
    protocol_wire_bytes: int
    protocol_session_dictionary_bytes: int
    protocol_typed_envelope_bytes: int
    protocol_typed_payload_bytes: int
    protocol_compact_message_bytes: int
    protocol_json_wire_bytes: int
    protocol_state_payload_bytes: int
    latency_reduction_rate: float
    memory_hit_rate: float
    memory_reused_unit_count: int
    memory_avg_reused_units_per_query: float
    memory_avg_score: float
    memory_avg_semantic_similarity: float
    memory_avg_tag_overlap_score: float
    quality_preservation_rate: float
    rust_core_enabled_runs: int
    rust_sandbox_backend_runs: int
    transport_send_count: int
    transport_bytes: int
    transport_avg_latency_ms: float
    transport_p99_latency_ms: float
    state_shm_transfer_count: int
    state_shm_transfer_bytes: int
    feedback_round_count: int
    planner_refine_count: int
    retriever_refine_count: int
    tool_feedback_count: int


class BenchmarkProgressEvent(BaseModel):
    phase: str
    suite_name: str
    total_tasks: int
    total_stages: int
    current_task: int = 0
    current_stage: int = 0
    repeat_index: int = 0
    repeat_total: int = 0
    task_id: str = ""
    group: str = ""
    topic: str = ""
    mode: str = ""
    trace_id: str = ""
    latency_ms: int = 0
    tokens: int = 0
    bytes: int = 0
    use_llm: bool = True


def run_benchmark(
    suite_path: Path,
    paths: RuntimePaths,
    *,
    use_llm: bool = True,
    progress_callback: Callable[[BenchmarkProgressEvent], None] | None = None,
) -> BenchmarkSummary:
    suite = _load_suite(suite_path)
    if paths.latest_run.exists():
        shutil.rmtree(paths.latest_run)
    paths.ensure()
    repeat = int(suite.get("repeat", 1))
    tasks = suite.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise BenchmarkConfigError("Benchmark suite requires tasks")
    suite_name = str(suite.get("name", suite_path.stem))
    total_tasks = repeat * len(tasks)
    total_stages = total_tasks * 2
    _emit_progress(
        progress_callback,
        BenchmarkProgressEvent(
            phase="suite_start",
            suite_name=suite_name,
            total_tasks=total_tasks,
            total_stages=total_stages,
            repeat_total=repeat,
            use_llm=use_llm,
        ),
    )
    text_tokens = 0
    protocol_tokens = 0
    text_msg_count = 0
    protocol_msg_count = 0
    text_wire_bytes = 0
    protocol_wire_bytes = 0
    text_agent_io_bytes = 0
    protocol_total_bytes = 0
    protocol_session_dictionary_bytes = 0
    protocol_typed_envelope_bytes = 0
    protocol_typed_payload_bytes = 0
    protocol_compact_message_bytes = 0
    protocol_json_wire_bytes = 0
    protocol_state_payload_bytes = 0
    text_latency = 0
    protocol_latency = 0
    memory_queries = 0
    memory_query_hits = 0
    memory_reused_units = 0
    memory_score_weighted_total = 0.0
    memory_semantic_weighted_total = 0.0
    memory_tag_overlap_weighted_total = 0.0
    text_quality = 0.0
    protocol_quality = 0.0
    rust_core_enabled_runs = 0
    rust_sandbox_backend_runs = 0
    transport_send_count = 0
    transport_bytes = 0
    transport_latency_weighted_total = 0.0
    transport_p99_latency_ms = 0.0
    state_shm_transfer_count = 0
    state_shm_transfer_bytes = 0
    feedback_round_count = 0
    planner_refine_count = 0
    retriever_refine_count = 0
    tool_feedback_count = 0
    logical_runs = 0

    for _round in range(repeat):
        for task_index, task in enumerate(tasks, start=1):
            current_task = _round * len(tasks) + task_index
            input_file = paths.root / str(task["input_file"])
            text_result = run_text_mode(
                task_path=input_file,
                paths=paths,
                load_configured_llm=use_llm,
            )
            _emit_progress(
                progress_callback,
                _progress_event(
                    suite_name=suite_name,
                    total_tasks=total_tasks,
                    total_stages=total_stages,
                    current_task=current_task,
                    current_stage=(current_task - 1) * 2 + 1,
                    repeat_index=_round + 1,
                    repeat_total=repeat,
                    task=task,
                    result=text_result,
                    use_llm=use_llm,
                ),
            )
            protocol_result = run_protocol_mode(
                task_path=input_file,
                paths=paths,
                load_configured_llm=use_llm,
            )
            _emit_progress(
                progress_callback,
                _progress_event(
                    suite_name=suite_name,
                    total_tasks=total_tasks,
                    total_stages=total_stages,
                    current_task=current_task,
                    current_stage=current_task * 2,
                    repeat_index=_round + 1,
                    repeat_total=repeat,
                    task=task,
                    result=protocol_result,
                    use_llm=use_llm,
                ),
            )
            logical_runs += 1
            text_tokens += _token_metric(text_result.metrics)
            protocol_tokens += _token_metric(protocol_result.metrics)
            text_msg_count += text_result.metrics.message_count
            protocol_msg_count += protocol_result.metrics.message_count
            text_wire_bytes += text_result.metrics.wire_bytes
            protocol_wire_bytes += protocol_result.metrics.wire_bytes
            text_agent_io_bytes += text_result.metrics.agent_io_bytes
            protocol_total_bytes += _byte_metric(protocol_result.metrics)
            protocol_session_dictionary_bytes += (
                protocol_result.metrics.session_dictionary_bytes
            )
            protocol_typed_envelope_bytes += protocol_result.metrics.typed_envelope_bytes
            protocol_typed_payload_bytes += protocol_result.metrics.typed_payload_bytes
            protocol_compact_message_bytes += (
                protocol_result.metrics.compact_structured_message_bytes
            )
            protocol_json_wire_bytes += protocol_result.metrics.structured_message_bytes
            protocol_state_payload_bytes += protocol_result.metrics.state_transfer_bytes
            text_latency += text_result.metrics.latency_ms
            protocol_latency += protocol_result.metrics.latency_ms
            memory_queries += protocol_result.metrics.memory_query_count
            memory_query_hits += protocol_result.metrics.memory_query_hit_count or min(
                protocol_result.metrics.memory_hit_count,
                protocol_result.metrics.memory_query_count,
            )
            memory_reused_units += protocol_result.metrics.memory_reused_unit_count
            memory_score_weighted_total += (
                protocol_result.metrics.memory_avg_score
                * protocol_result.metrics.memory_reused_unit_count
            )
            memory_semantic_weighted_total += (
                protocol_result.metrics.memory_avg_semantic_similarity
                * protocol_result.metrics.memory_reused_unit_count
            )
            memory_tag_overlap_weighted_total += (
                protocol_result.metrics.memory_avg_tag_overlap_score
                * protocol_result.metrics.memory_reused_unit_count
            )
            text_quality += text_result.metrics.answer_quality_score
            protocol_quality += protocol_result.metrics.answer_quality_score
            rust_core_enabled_runs += int(protocol_result.metrics.rust_core_enabled)
            rust_sandbox_backend_runs += int(protocol_result.metrics.sandbox_backend == "rust")
            transport_send_count += protocol_result.metrics.transport_send_count
            transport_bytes += protocol_result.metrics.transport_bytes
            transport_latency_weighted_total += (
                protocol_result.metrics.transport_avg_latency_ms
                * protocol_result.metrics.transport_send_count
            )
            transport_p99_latency_ms = max(
                transport_p99_latency_ms,
                protocol_result.metrics.transport_p99_latency_ms,
            )
            state_shm_transfer_count += protocol_result.metrics.state_shm_transfer_count
            state_shm_transfer_bytes += protocol_result.metrics.state_shm_transfer_bytes
            feedback_round_count += protocol_result.metrics.feedback_round_count
            planner_refine_count += protocol_result.metrics.planner_refine_count
            retriever_refine_count += protocol_result.metrics.retriever_refine_count
            tool_feedback_count += protocol_result.metrics.tool_feedback_count
            for result in [text_result, protocol_result]:
                append_jsonl(
                    paths.benchmark_detail,
                    {
                        "task_id": task["id"],
                        "group": task.get("group", ""),
                        "topic": task.get("topic", ""),
                        "mode": result.mode,
                        "trace_id": result.trace_id,
                        "metrics": result.metrics.model_dump(),
                    },
                )

    summary = BenchmarkSummary(
        suite_name=suite_name,
        total_runs=logical_runs,
        token_estimator=TOKEN_ESTIMATOR,
        token_saving_rate=_rate(text_tokens, protocol_tokens),
        wire_bytes_reduction_rate=_rate(
            text_agent_io_bytes or text_wire_bytes,
            protocol_total_bytes or protocol_wire_bytes,
        ),
        text_agent_io_tokens=text_tokens,
        protocol_agent_io_tokens=protocol_tokens,
        text_per_msg_avg_tokens=(
            text_tokens / text_msg_count if text_msg_count else 0.0
        ),
        protocol_per_msg_avg_tokens=(
            protocol_tokens / protocol_msg_count if protocol_msg_count else 0.0
        ),
        text_agent_io_bytes=text_agent_io_bytes,
        protocol_total_bytes=protocol_total_bytes,
        text_wire_bytes=text_wire_bytes,
        protocol_wire_bytes=protocol_wire_bytes,
        protocol_session_dictionary_bytes=protocol_session_dictionary_bytes,
        protocol_typed_envelope_bytes=protocol_typed_envelope_bytes,
        protocol_typed_payload_bytes=protocol_typed_payload_bytes,
        protocol_compact_message_bytes=protocol_compact_message_bytes,
        protocol_json_wire_bytes=protocol_json_wire_bytes,
        protocol_state_payload_bytes=protocol_state_payload_bytes,
        latency_reduction_rate=_rate(text_latency, protocol_latency),
        memory_hit_rate=memory_query_hits / memory_queries if memory_queries else 0.0,
        memory_reused_unit_count=memory_reused_units,
        memory_avg_reused_units_per_query=(
            memory_reused_units / memory_queries if memory_queries else 0.0
        ),
        memory_avg_score=(
            memory_score_weighted_total / memory_reused_units
            if memory_reused_units
            else 0.0
        ),
        memory_avg_semantic_similarity=(
            memory_semantic_weighted_total / memory_reused_units
            if memory_reused_units
            else 0.0
        ),
        memory_avg_tag_overlap_score=(
            memory_tag_overlap_weighted_total / memory_reused_units
            if memory_reused_units
            else 0.0
        ),
        quality_preservation_rate=protocol_quality / text_quality if text_quality else 0.0,
        rust_core_enabled_runs=rust_core_enabled_runs,
        rust_sandbox_backend_runs=rust_sandbox_backend_runs,
        transport_send_count=transport_send_count,
        transport_bytes=transport_bytes,
        transport_avg_latency_ms=(
            transport_latency_weighted_total / transport_send_count
            if transport_send_count
            else 0.0
        ),
        transport_p99_latency_ms=transport_p99_latency_ms,
        state_shm_transfer_count=state_shm_transfer_count,
        state_shm_transfer_bytes=state_shm_transfer_bytes,
        feedback_round_count=feedback_round_count,
        planner_refine_count=planner_refine_count,
        retriever_refine_count=retriever_refine_count,
        tool_feedback_count=tool_feedback_count,
    )
    _write_summary(paths.benchmark_summary, summary)
    _emit_progress(
        progress_callback,
        BenchmarkProgressEvent(
            phase="suite_complete",
            suite_name=suite_name,
            total_tasks=total_tasks,
            total_stages=total_stages,
            current_task=total_tasks,
            current_stage=total_stages,
            repeat_total=repeat,
            use_llm=use_llm,
        ),
    )
    return summary


def _emit_progress(
    progress_callback: Callable[[BenchmarkProgressEvent], None] | None,
    event: BenchmarkProgressEvent,
) -> None:
    if progress_callback is not None:
        progress_callback(event)


def _progress_event(
    *,
    suite_name: str,
    total_tasks: int,
    total_stages: int,
    current_task: int,
    current_stage: int,
    repeat_index: int,
    repeat_total: int,
    task: dict[str, Any],
    result: Any,
    use_llm: bool,
) -> BenchmarkProgressEvent:
    return BenchmarkProgressEvent(
        phase="mode_complete",
        suite_name=suite_name,
        total_tasks=total_tasks,
        total_stages=total_stages,
        current_task=current_task,
        current_stage=current_stage,
        repeat_index=repeat_index,
        repeat_total=repeat_total,
        task_id=str(task.get("id", "")),
        group=str(task.get("group", "")),
        topic=str(task.get("topic", "")),
        mode=str(result.mode),
        trace_id=str(result.trace_id),
        latency_ms=result.metrics.latency_ms,
        tokens=_token_metric(result.metrics),
        bytes=_byte_metric(result.metrics),
        use_llm=use_llm,
    )


def _load_suite(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BenchmarkConfigError(f"Benchmark suite not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise BenchmarkConfigError("Benchmark suite must be a mapping")
    return loaded


def _rate(baseline: int, candidate: int) -> float:
    if baseline <= 0:
        return 0.0
    return (baseline - candidate) / baseline


def _token_metric(metrics: RunMetrics) -> int:
    return metrics.agent_io_tokens or metrics.estimated_tokens


def _byte_metric(metrics: RunMetrics) -> int:
    return metrics.protocol_total_bytes or metrics.agent_io_bytes or metrics.wire_bytes


def _write_summary(path: Path, summary: BenchmarkSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.model_dump().keys()))
        writer.writeheader()
        writer.writerow(summary.model_dump())
