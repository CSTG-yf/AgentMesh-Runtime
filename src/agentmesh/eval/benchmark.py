import csv
import shutil
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from agentmesh.errors import BenchmarkConfigError
from agentmesh.eval.metrics import TOKEN_ESTIMATOR
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


def run_benchmark(suite_path: Path, paths: RuntimePaths) -> BenchmarkSummary:
    suite = _load_suite(suite_path)
    if paths.latest_run.exists():
        shutil.rmtree(paths.latest_run)
    paths.ensure()
    repeat = int(suite.get("repeat", 1))
    tasks = suite.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise BenchmarkConfigError("Benchmark suite requires tasks")
    text_tokens = 0
    protocol_tokens = 0
    text_wire_bytes = 0
    protocol_wire_bytes = 0
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
        for task in tasks:
            input_file = paths.root / str(task["input_file"])
            text_result = run_text_mode(task_path=input_file, paths=paths)
            protocol_result = run_protocol_mode(
                task_path=input_file,
                paths=paths,
                load_configured_llm=False,
            )
            logical_runs += 1
            text_tokens += text_result.metrics.estimated_tokens
            protocol_tokens += protocol_result.metrics.estimated_tokens
            text_wire_bytes += text_result.metrics.wire_bytes
            protocol_wire_bytes += protocol_result.metrics.wire_bytes
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
        suite_name=str(suite.get("name", suite_path.stem)),
        total_runs=logical_runs,
        token_estimator=TOKEN_ESTIMATOR,
        token_saving_rate=_rate(text_tokens, protocol_tokens),
        wire_bytes_reduction_rate=_rate(text_wire_bytes, protocol_wire_bytes),
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
    return summary


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


def _write_summary(path: Path, summary: BenchmarkSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.model_dump().keys()))
        writer.writeheader()
        writer.writerow(summary.model_dump())
