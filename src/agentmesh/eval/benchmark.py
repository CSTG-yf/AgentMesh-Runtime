import csv
import hashlib
import os
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import orjson
import yaml
from pydantic import BaseModel, Field

from agentmesh.config import AgentMeshConfig
from agentmesh.core import rust_available
from agentmesh.errors import BenchmarkConfigError
from agentmesh.eval.experiment import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkTrack,
    ExperimentManifest,
    build_experiment_manifest,
    model_fingerprint,
    paired_mode_order,
    prompt_tree_sha256,
)
from agentmesh.eval.llm_experiment import LLMExperimentProfile
from agentmesh.eval.metrics import TOKEN_ESTIMATOR, ModeRunResult, RunMetrics
from agentmesh.eval.quality_spec import QualityResult, QualitySpec, evaluate_quality
from agentmesh.eval.statistics import DistributionStats, distribution_stats
from agentmesh.modes.protocol_mode import run_protocol_mode
from agentmesh.modes.text_mode import run_text_mode
from agentmesh.storage.jsonl import append_jsonl
from agentmesh.storage.paths import RuntimePaths


class BenchmarkSummary(BaseModel):
    schema_version: str = BENCHMARK_SCHEMA_VERSION
    experiment_id: str
    track: BenchmarkTrack
    repeat_count: int
    seed: int
    suite_name: str
    artifact_variant: str = ""
    quality_rules_sha256: str = ""
    total_runs: int
    token_estimator: str
    token_saving_rate: float
    wire_bytes_reduction_rate: float
    fair_wire_reduction_rate: float = 0.0
    agent_io_bytes_reduction_rate: float = 0.0
    text_agent_io_tokens: int = 0
    protocol_agent_io_tokens: int = 0
    text_per_msg_avg_tokens: float = 0.0
    protocol_per_msg_avg_tokens: float = 0.0
    text_agent_io_bytes: int = 0
    protocol_agent_io_bytes: int = 0
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
    memory_evidence_count: int = 0
    memory_evidence_bytes: int = 0
    memory_avg_evidence_bytes_per_query: float = 0.0
    memory_avg_reused_units_per_query: float
    memory_avg_score: float
    memory_avg_semantic_similarity: float
    memory_avg_tag_overlap_score: float
    quality_scored_runs: int
    quality_unscored_runs: int
    text_quality_mean: float | None
    protocol_quality_mean: float | None
    text_quality_pass_rate: float | None
    protocol_quality_pass_rate: float | None
    quality_score_delta: float | None
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
    text_latency_stats: DistributionStats
    protocol_latency_stats: DistributionStats
    text_token_stats: DistributionStats
    protocol_token_stats: DistributionStats


class BenchmarkDetailRecord(BaseModel):
    schema_version: str = BENCHMARK_SCHEMA_VERSION
    experiment_id: str
    track: BenchmarkTrack
    repeat_index: int
    pair_index: int
    pair_order: tuple[str, str]
    task_id: str
    task_sha256: str
    group: str = ""
    topic: str = ""
    tags: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    cold_start: bool = False
    mode: str
    trace_id: str
    metrics: RunMetrics
    quality: QualityResult


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
    tags: list[str] = []
    depends_on: list[str] = []
    cold_start: bool = False
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
    profile: LLMExperimentProfile | None = None,
) -> BenchmarkSummary:
    previous_global_memory_setting = os.environ.get(
        "AGENTMESH_DISABLE_GLOBAL_MEMORY"
    )
    os.environ["AGENTMESH_DISABLE_GLOBAL_MEMORY"] = "1"
    try:
        return _run_benchmark_impl(
            suite_path,
            paths,
            use_llm=use_llm,
            progress_callback=progress_callback,
            profile=profile,
        )
    finally:
        _restore_global_memory_setting(previous_global_memory_setting)


def _run_benchmark_impl(
    suite_path: Path,
    paths: RuntimePaths,
    *,
    use_llm: bool = True,
    progress_callback: Callable[[BenchmarkProgressEvent], None] | None = None,
    profile: LLMExperimentProfile | None = None,
) -> BenchmarkSummary:
    suite = _load_suite(suite_path)
    _reset_runtime_working_dirs(paths)
    paths.ensure()
    _reset_run_memory(paths)
    repeat = profile.repeat if profile is not None else int(suite.get("repeat", 1))
    tasks = suite.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise BenchmarkConfigError("Benchmark suite requires tasks")
    suite_name = str(suite.get("name", suite_path.stem))
    track = BenchmarkTrack.LLM if use_llm else BenchmarkTrack.DETERMINISTIC
    seed = int(suite.get("seed", 0))
    runtime_config = AgentMeshConfig.from_project_root(paths.root)
    if profile is not None and use_llm and not runtime_config.llm.configured:
        raise BenchmarkConfigError(
            "LLM experiment profile requires base_url, api_key, and model"
        )
    quality_rules_hash = _quality_rules_sha256(tasks)
    prompt_dir = runtime_config.prompt_dir or paths.root / "prompts"
    manifest = build_experiment_manifest(
        suite_path=suite_path,
        suite_name=suite_name,
        track=track,
        repeat_count=repeat,
        seed=seed,
        environment={
            "model": runtime_config.llm.model or "",
            "rust_core": str(rust_available()),
        },
        profile_id=profile.profile_id if profile else "",
        profile_sha256=profile.sha256 if profile else "",
        prompt_version=profile.prompt_version if profile else "",
        route_policy_version=profile.route_policy_version if profile else "",
        model_fingerprint=model_fingerprint(
            runtime_config.llm.base_url, runtime_config.llm.model
        ),
        prompt_tree_sha256=prompt_tree_sha256(prompt_dir),
        quality_rules_sha256=quality_rules_hash,
    )
    variant = profile.artifact_label if profile else None
    detail_path = paths.benchmark_suite_detail(suite_name, track.value, variant)
    summary_path = paths.benchmark_suite_summary(suite_name, track.value, variant)
    report_path = paths.benchmark_suite_report(suite_name, track.value, variant)
    manifest_path = paths.benchmark_suite_manifest(suite_name, track.value, variant)
    _reset_suite_artifacts(summary_path, detail_path, report_path, manifest_path)
    _write_manifest(manifest_path, manifest)
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
    protocol_agent_io_bytes = 0
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
    memory_evidence_count = 0
    memory_evidence_bytes = 0
    memory_score_weighted_total = 0.0
    memory_semantic_weighted_total = 0.0
    memory_tag_overlap_weighted_total = 0.0
    quality_scored_runs = 0
    quality_unscored_runs = 0
    text_quality_scores: list[float] = []
    protocol_quality_scores: list[float] = []
    text_quality_passes: list[float] = []
    protocol_quality_passes: list[float] = []
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
    text_latency_samples: list[int] = []
    protocol_latency_samples: list[int] = []
    text_token_samples: list[int] = []
    protocol_token_samples: list[int] = []

    for _round in range(repeat):
        for task_index, task in enumerate(tasks, start=1):
            current_task = _round * len(tasks) + task_index
            if task.get("cold_start"):
                _reset_run_memory(paths)
                _emit_progress(
                    progress_callback,
                    BenchmarkProgressEvent(
                        phase="cold_start",
                        suite_name=suite_name,
                        total_tasks=total_tasks,
                        total_stages=total_stages,
                        current_task=current_task,
                        current_stage=(current_task - 1) * 2,
                        repeat_index=_round + 1,
                        repeat_total=repeat,
                        task_id=str(task.get("id", "")),
                        group=str(task.get("group", "")),
                        topic=str(task.get("topic", "")),
                        tags=_list_of_str(task.get("tags")),
                        depends_on=_list_of_str(task.get("depends_on")),
                        cold_start=True,
                        use_llm=use_llm,
                    ),
                )
            input_file = paths.root / str(task["input_file"])
            raw_quality = task.get("quality")
            quality_spec = (
                QualitySpec.model_validate(raw_quality)
                if isinstance(raw_quality, dict)
                else None
            )
            pair_index = _round * len(tasks) + task_index - 1
            pair_order = paired_mode_order(seed=seed, pair_index=pair_index)
            results: dict[str, ModeRunResult] = {}
            for stage_offset, mode in enumerate(pair_order, start=1):
                if mode == "text":
                    result = run_text_mode(
                        task_path=input_file,
                        paths=paths,
                        load_configured_llm=use_llm,
                    )
                elif profile is None:
                    result = run_protocol_mode(
                        task_path=input_file,
                        paths=paths,
                        load_configured_llm=use_llm,
                    )
                else:
                    result = run_protocol_mode(
                        task_path=input_file,
                        paths=paths,
                        load_configured_llm=use_llm,
                        experiment_profile=profile,
                    )
                results[mode] = result
                _emit_progress(
                    progress_callback,
                    _progress_event(
                        suite_name=suite_name,
                        total_tasks=total_tasks,
                        total_stages=total_stages,
                        current_task=current_task,
                        current_stage=(current_task - 1) * 2 + stage_offset,
                        repeat_index=_round + 1,
                        repeat_total=repeat,
                        task=task,
                        result=result,
                        use_llm=use_llm,
                    ),
                )
            text_result = results["text"]
            protocol_result = results["protocol"]
            if profile is not None and (
                text_result.metrics.llm_error_count
                or protocol_result.metrics.llm_error_count
            ):
                raise BenchmarkConfigError(
                    f"LLM provider failure for task {task.get('id', '')}"
                )
            quality_results = {
                "text": evaluate_quality(text_result.answer, quality_spec),
                "protocol": evaluate_quality(protocol_result.answer, quality_spec),
            }
            if quality_spec is None:
                quality_unscored_runs += 1
            else:
                quality_scored_runs += 1
                text_quality = quality_results["text"]
                protocol_quality = quality_results["protocol"]
                if text_quality.score is None or protocol_quality.score is None:
                    raise BenchmarkConfigError("Configured quality rule returned no score")
                text_quality_scores.append(text_quality.score)
                protocol_quality_scores.append(protocol_quality.score)
                text_quality_passes.append(float(text_quality.passed))
                protocol_quality_passes.append(float(protocol_quality.passed))
            logical_runs += 1
            text_token_sample = _token_metric(text_result.metrics)
            protocol_token_sample = _token_metric(protocol_result.metrics)
            text_tokens += text_token_sample
            protocol_tokens += protocol_token_sample
            text_token_samples.append(text_token_sample)
            protocol_token_samples.append(protocol_token_sample)
            text_msg_count += text_result.metrics.message_count
            protocol_msg_count += protocol_result.metrics.message_count
            text_wire_bytes += text_result.metrics.wire_bytes
            protocol_wire_bytes += protocol_result.metrics.wire_bytes
            text_agent_io_bytes += text_result.metrics.agent_io_bytes
            protocol_agent_io_bytes += protocol_result.metrics.agent_io_bytes
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
            text_latency_samples.append(text_result.metrics.latency_ms)
            protocol_latency_samples.append(protocol_result.metrics.latency_ms)
            memory_queries += protocol_result.metrics.memory_query_count
            memory_query_hits += protocol_result.metrics.memory_query_hit_count or min(
                protocol_result.metrics.memory_hit_count,
                protocol_result.metrics.memory_query_count,
            )
            memory_reused_units += protocol_result.metrics.memory_reused_unit_count
            memory_evidence_count += protocol_result.metrics.memory_evidence_count
            memory_evidence_bytes += protocol_result.metrics.memory_evidence_bytes
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
            task_sha256 = hashlib.sha256(input_file.read_bytes()).hexdigest()
            for result in [text_result, protocol_result]:
                detail_record = BenchmarkDetailRecord(
                    experiment_id=manifest.experiment_id,
                    track=track,
                    repeat_index=_round + 1,
                    pair_index=pair_index,
                    pair_order=pair_order,
                    task_id=str(task["id"]),
                    task_sha256=task_sha256,
                    group=str(task.get("group", "")),
                    topic=str(task.get("topic", "")),
                    tags=_list_of_str(task.get("tags")),
                    depends_on=_list_of_str(task.get("depends_on")),
                    cold_start=bool(task.get("cold_start", False)),
                    mode=result.mode,
                    trace_id=result.trace_id,
                    metrics=result.metrics,
                    quality=quality_results[result.mode],
                )
                append_jsonl(
                    detail_path,
                    detail_record.model_dump(mode="json"),
                )

    summary = BenchmarkSummary(
        experiment_id=manifest.experiment_id,
        track=track,
        repeat_count=repeat,
        seed=seed,
        suite_name=suite_name,
        artifact_variant=variant or "",
        quality_rules_sha256=quality_rules_hash,
        total_runs=logical_runs,
        token_estimator=TOKEN_ESTIMATOR,
        token_saving_rate=_rate(text_tokens, protocol_tokens),
        wire_bytes_reduction_rate=_rate(
            text_agent_io_bytes or text_wire_bytes,
            protocol_total_bytes or protocol_wire_bytes,
        ),
        fair_wire_reduction_rate=_rate(text_wire_bytes, protocol_wire_bytes),
        agent_io_bytes_reduction_rate=_rate(
            text_agent_io_bytes,
            protocol_agent_io_bytes,
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
        protocol_agent_io_bytes=protocol_agent_io_bytes,
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
        memory_evidence_count=memory_evidence_count,
        memory_evidence_bytes=memory_evidence_bytes,
        memory_avg_evidence_bytes_per_query=(
            memory_evidence_bytes / memory_queries if memory_queries else 0.0
        ),
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
        quality_scored_runs=quality_scored_runs,
        quality_unscored_runs=quality_unscored_runs,
        text_quality_mean=_mean_or_none(text_quality_scores),
        protocol_quality_mean=_mean_or_none(protocol_quality_scores),
        text_quality_pass_rate=_mean_or_none(text_quality_passes),
        protocol_quality_pass_rate=_mean_or_none(protocol_quality_passes),
        quality_score_delta=_quality_delta(
            text_quality_scores,
            protocol_quality_scores,
        ),
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
        text_latency_stats=distribution_stats(text_latency_samples),
        protocol_latency_stats=distribution_stats(protocol_latency_samples),
        text_token_stats=distribution_stats(text_token_samples),
        protocol_token_stats=distribution_stats(protocol_token_samples),
    )
    _write_summary(summary_path, summary)
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
        tags=_list_of_str(task.get("tags")),
        depends_on=_list_of_str(task.get("depends_on")),
        cold_start=bool(task.get("cold_start", False)),
        mode=str(result.mode),
        trace_id=str(result.trace_id),
        latency_ms=result.metrics.latency_ms,
        tokens=_token_metric(result.metrics),
        bytes=_byte_metric(result.metrics),
        use_llm=use_llm,
    )


def _list_of_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _reset_run_memory(paths: RuntimePaths) -> None:
    db = paths.memory_db
    if not db.exists():
        return
    if not _looks_like_sqlite(db):
        _unlink_if_available(db)
        return
    try:
        with sqlite3.connect(db) as conn:
            tables = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if "memory_fts" in tables:
                conn.execute("DELETE FROM memory_fts")
            if "memory_units" in tables:
                conn.execute("DELETE FROM memory_units")
            conn.execute("VACUUM")
    except sqlite3.DatabaseError:
        _unlink_if_available(db)


def _reset_runtime_working_dirs(paths: RuntimePaths) -> None:
    for path in [paths.text_dir, paths.protocol_dir, paths.sandbox_dir]:
        if path.exists():
            shutil.rmtree(path)
    _reset_data_dir(paths)


def _reset_data_dir(paths: RuntimePaths) -> None:
    if paths.state_payload_dir.exists():
        shutil.rmtree(paths.state_payload_dir, ignore_errors=True)
    _reset_run_memory(paths)
    _unlink_if_available(paths.state_index)


def _reset_suite_artifacts(*paths: Path) -> None:
    for path in paths:
        _unlink_if_available(path)


def _looks_like_sqlite(path: Path) -> bool:
    try:
        return path.read_bytes()[:16] == b"SQLite format 3\x00"
    except OSError:
        return False


def _unlink_if_available(path: Path) -> None:
    if not path.exists():
        return
    try:
        path.unlink()
    except PermissionError:
        return


def _restore_global_memory_setting(previous_value: str | None) -> None:
    if previous_value is None:
        os.environ.pop("AGENTMESH_DISABLE_GLOBAL_MEMORY", None)
    else:
        os.environ["AGENTMESH_DISABLE_GLOBAL_MEMORY"] = previous_value


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


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _quality_delta(
    text_scores: list[float],
    protocol_scores: list[float],
) -> float | None:
    text_mean = _mean_or_none(text_scores)
    protocol_mean = _mean_or_none(protocol_scores)
    if text_mean is None or protocol_mean is None:
        return None
    return protocol_mean - text_mean


def _token_metric(metrics: RunMetrics) -> int:
    return metrics.agent_io_tokens or metrics.estimated_tokens


def _byte_metric(metrics: RunMetrics) -> int:
    return metrics.protocol_total_bytes or metrics.agent_io_bytes or metrics.wire_bytes


def _write_summary(path: Path, summary: BenchmarkSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        key: _csv_value(value)
        for key, value in summary.model_dump(mode="json").items()
    }
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list)):
        return orjson.dumps(value).decode("utf-8")
    return value


def _write_manifest(path: Path, manifest: ExperimentManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        orjson.dumps(
            manifest.model_dump(mode="json"),
            option=orjson.OPT_INDENT_2,
        )
    )


def _quality_rules_sha256(tasks: list[object]) -> str:
    rules: list[dict[str, object]] = []
    for task in tasks:
        if isinstance(task, dict):
            rules.append(
                {
                    "task_id": str(task.get("id", "")),
                    "quality": task.get("quality"),
                }
            )
    canonical = orjson.dumps(rules, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(canonical).hexdigest()
