import json
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from inspect import signature
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from agentmesh.eval.benchmark import _rate
from agentmesh.eval.metrics import ModeRunResult
from agentmesh.modes.protocol_mode import run_protocol_mode
from agentmesh.modes.text_mode import run_text_mode
from agentmesh.storage.jsonl import read_jsonl
from agentmesh.storage.paths import RuntimePaths


class CompareAgentOutput(BaseModel):
    mode: Literal["text", "protocol"]
    trace_id: str
    step: int
    agent: str
    action: str = ""
    input_summary: str = ""
    output: str = ""
    state_refs_in: list[str] = Field(default_factory=list)
    state_refs_out: list[str] = Field(default_factory=list)


class CompareProgressEvent(BaseModel):
    kind: Literal["mode_start", "agent_output", "mode_complete"]
    mode: Literal["text", "protocol"]
    message: str
    agent_output: CompareAgentOutput | None = None


ProgressCallback = Callable[[CompareProgressEvent], None]


class CompareSummary(BaseModel):
    task_path: str
    text: ModeRunResult
    protocol: ModeRunResult
    text_agent_outputs: list[CompareAgentOutput] = Field(default_factory=list)
    protocol_agent_outputs: list[CompareAgentOutput] = Field(default_factory=list)
    token_saving_rate: float
    latency_reduction_rate: float
    wire_bytes_reduction_rate: float
    memory_hit_rate: float
    memory_query_count: int = 0
    memory_query_hit_count: int = 0
    memory_reused_unit_count: int = 0
    memory_avg_reused_units_per_query: float = 0.0
    memory_avg_score: float = 0.0
    memory_avg_semantic_similarity: float = 0.0
    memory_avg_tag_overlap_score: float = 0.0


def run_prompt_compare(
    prompt: str,
    paths: RuntimePaths,
    *,
    use_llm: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> CompareSummary:
    paths.ensure()
    task_dir = paths.latest_run / "user_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    task_path = task_dir / f"prompt-{uuid4().hex[:8]}.txt"
    task_path.write_text(prompt, encoding="utf-8")

    if progress_callback is None:
        text_result = run_text_mode(
            task_path=task_path,
            paths=paths,
            load_configured_llm=use_llm,
        )
        protocol_result = run_protocol_mode(
            task_path=task_path,
            paths=paths,
            load_configured_llm=use_llm,
        )
    else:
        text_result, protocol_result = _run_modes_with_progress(
            task_path=task_path,
            paths=paths,
            use_llm=use_llm,
            progress_callback=progress_callback,
        )
    return _build_summary(
        task_path=task_path,
        paths=paths,
        text_result=text_result,
        protocol_result=protocol_result,
    )


def _run_modes_with_progress(
    *,
    task_path: Path,
    paths: RuntimePaths,
    use_llm: bool,
    progress_callback: ProgressCallback,
) -> tuple[ModeRunResult, ModeRunResult]:
    text_offset = len(read_jsonl(paths.text_agent_io))
    protocol_offset = len(read_jsonl(paths.protocol_agent_io))
    text_trace_id = f"trace-compare-text-{uuid4().hex[:12]}"
    protocol_trace_id = f"trace-compare-protocol-{uuid4().hex[:12]}"
    progress_callback(
        CompareProgressEvent(kind="mode_start", mode="text", message="Text Mode started")
    )
    progress_callback(
        CompareProgressEvent(
            kind="mode_start",
            mode="protocol",
            message="Protocol Mode started",
        )
    )

    results: dict[str, ModeRunResult] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        text_future = executor.submit(
            run_text_mode,
            **_runner_kwargs(
                run_text_mode,
                task_path=task_path,
                paths=paths,
                load_configured_llm=use_llm,
                trace_id=text_trace_id,
            ),
        )
        protocol_future = executor.submit(
            run_protocol_mode,
            **_runner_kwargs(
                run_protocol_mode,
                task_path=task_path,
                paths=paths,
                load_configured_llm=use_llm,
                trace_id=protocol_trace_id,
            ),
        )
        expected_text_trace_id = (
            text_trace_id if _accepts_kwarg(run_text_mode, "trace_id") else None
        )
        expected_protocol_trace_id = (
            protocol_trace_id if _accepts_kwarg(run_protocol_mode, "trace_id") else None
        )
        futures: dict[Future[ModeRunResult], Literal["text", "protocol"]] = {
            text_future: "text",
            protocol_future: "protocol",
        }
        while futures:
            text_offset, protocol_offset = _emit_new_agent_outputs(
                paths=paths,
                text_offset=text_offset,
                protocol_offset=protocol_offset,
                text_trace_id=expected_text_trace_id,
                protocol_trace_id=expected_protocol_trace_id,
                progress_callback=progress_callback,
            )
            done, _ = wait(list(futures), timeout=0.1, return_when=FIRST_COMPLETED)
            if done:
                text_offset, protocol_offset = _emit_new_agent_outputs(
                    paths=paths,
                    text_offset=text_offset,
                    protocol_offset=protocol_offset,
                    text_trace_id=expected_text_trace_id,
                    protocol_trace_id=expected_protocol_trace_id,
                    progress_callback=progress_callback,
                )
            for future in done:
                mode = futures.pop(future)
                results[mode] = future.result()
                progress_callback(
                    CompareProgressEvent(
                        kind="mode_complete",
                        mode=mode,
                        message=f"{mode.title()} Mode completed",
                    )
                )
        _emit_new_agent_outputs(
            paths=paths,
            text_offset=text_offset,
            protocol_offset=protocol_offset,
            text_trace_id=expected_text_trace_id,
            protocol_trace_id=expected_protocol_trace_id,
            progress_callback=progress_callback,
        )
    return results["text"], results["protocol"]


def run_protocol_with_progress(
    *,
    task_path: Path,
    paths: RuntimePaths,
    use_llm: bool = True,
    progress_callback: ProgressCallback,
    runner: Callable[..., ModeRunResult] = run_protocol_mode,
) -> ModeRunResult:
    protocol_offset = len(read_jsonl(paths.protocol_agent_io))
    trace_id = f"trace-ask-protocol-{uuid4().hex[:12]}"
    expected_trace_id = trace_id if _accepts_kwarg(runner, "trace_id") else None
    progress_callback(
        CompareProgressEvent(
            kind="mode_start",
            mode="protocol",
            message="Protocol Mode started",
        )
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            runner,
            **_runner_kwargs(
                runner,
                task_path=task_path,
                paths=paths,
                load_configured_llm=use_llm,
                trace_id=trace_id,
            ),
        )
        while True:
            protocol_offset = _emit_new_protocol_outputs(
                paths=paths,
                protocol_offset=protocol_offset,
                trace_id=expected_trace_id,
                progress_callback=progress_callback,
            )
            done, _ = wait([future], timeout=0.1, return_when=FIRST_COMPLETED)
            if not done:
                continue
            protocol_offset = _emit_new_protocol_outputs(
                paths=paths,
                protocol_offset=protocol_offset,
                trace_id=expected_trace_id,
                progress_callback=progress_callback,
            )
            result = future.result()
            progress_callback(
                CompareProgressEvent(
                    kind="mode_complete",
                    mode="protocol",
                    message="Protocol Mode completed",
                )
            )
            return result


def _build_summary(
    *,
    task_path: Path,
    paths: RuntimePaths,
    text_result: ModeRunResult,
    protocol_result: ModeRunResult,
) -> CompareSummary:
    return CompareSummary(
        task_path=str(task_path),
        text=text_result,
        protocol=protocol_result,
        text_agent_outputs=_load_agent_outputs(
            paths=paths,
            mode="text",
            trace_id=text_result.trace_id,
        ),
        protocol_agent_outputs=_load_agent_outputs(
            paths=paths,
            mode="protocol",
            trace_id=protocol_result.trace_id,
        ),
        token_saving_rate=_rate(
            text_result.metrics.estimated_tokens,
            protocol_result.metrics.estimated_tokens,
        ),
        latency_reduction_rate=_rate(
            text_result.metrics.latency_ms,
            protocol_result.metrics.latency_ms,
        ),
        wire_bytes_reduction_rate=_rate(
            text_result.metrics.wire_bytes,
            protocol_result.metrics.wire_bytes,
        ),
        memory_hit_rate=protocol_result.metrics.memory_hit_rate,
        memory_query_count=protocol_result.metrics.memory_query_count,
        memory_query_hit_count=(
            protocol_result.metrics.memory_query_hit_count
            or min(
                protocol_result.metrics.memory_hit_count,
                protocol_result.metrics.memory_query_count,
            )
        ),
        memory_reused_unit_count=protocol_result.metrics.memory_reused_unit_count,
        memory_avg_reused_units_per_query=(
            protocol_result.metrics.memory_reused_unit_count
            / protocol_result.metrics.memory_query_count
            if protocol_result.metrics.memory_query_count
            else 0.0
        ),
        memory_avg_score=protocol_result.metrics.memory_avg_score,
        memory_avg_semantic_similarity=protocol_result.metrics.memory_avg_semantic_similarity,
        memory_avg_tag_overlap_score=protocol_result.metrics.memory_avg_tag_overlap_score,
    )


def _emit_new_agent_outputs(
    *,
    paths: RuntimePaths,
    text_offset: int,
    protocol_offset: int,
    text_trace_id: str | None = None,
    protocol_trace_id: str | None = None,
    progress_callback: ProgressCallback,
) -> tuple[int, int]:
    text_rows = read_jsonl(paths.text_agent_io)
    protocol_rows = read_jsonl(paths.protocol_agent_io)
    for row in text_rows[text_offset:]:
        if text_trace_id is not None and str(row.get("trace_id", "")) != text_trace_id:
            continue
        output = _agent_output_from_text_row(row)
        if output is not None:
            progress_callback(
                CompareProgressEvent(
                    kind="agent_output",
                    mode="text",
                    message=f"Text Mode {output.agent} output",
                    agent_output=output,
                )
            )
    for row in protocol_rows[protocol_offset:]:
        if protocol_trace_id is not None and str(row.get("trace_id", "")) != protocol_trace_id:
            continue
        output = _agent_output_from_protocol_row(row)
        if output is not None:
            progress_callback(
                CompareProgressEvent(
                    kind="agent_output",
                    mode="protocol",
                    message=f"Protocol Mode {output.agent} output",
                    agent_output=output,
                )
            )
    return len(text_rows), len(protocol_rows)


def _emit_new_protocol_outputs(
    *,
    paths: RuntimePaths,
    protocol_offset: int,
    trace_id: str | None = None,
    progress_callback: ProgressCallback,
) -> int:
    protocol_rows = read_jsonl(paths.protocol_agent_io)
    for row in protocol_rows[protocol_offset:]:
        if trace_id is not None and str(row.get("trace_id", "")) != trace_id:
            continue
        output = _agent_output_from_protocol_row(row)
        if output is not None:
            progress_callback(
                CompareProgressEvent(
                    kind="agent_output",
                    mode="protocol",
                    message=f"Protocol Mode {output.agent} output",
                    agent_output=output,
                )
            )
    return len(protocol_rows)


def _load_agent_outputs(
    *,
    paths: RuntimePaths,
    mode: Literal["text", "protocol"],
    trace_id: str,
) -> list[CompareAgentOutput]:
    rows = read_jsonl(paths.text_agent_io if mode == "text" else paths.protocol_agent_io)
    outputs: list[CompareAgentOutput] = []
    for row in rows:
        if str(row.get("trace_id", "")) != trace_id:
            continue
        output = (
            _agent_output_from_text_row(row)
            if mode == "text"
            else _agent_output_from_protocol_row(row)
        )
        if output is not None:
            outputs.append(output)
    return sorted(outputs, key=lambda item: item.step)


def _agent_output_from_text_row(row: dict[str, Any]) -> CompareAgentOutput | None:
    output = row.get("output")
    input_value = row.get("input")
    if not isinstance(output, dict) or not isinstance(input_value, dict):
        return None
    return CompareAgentOutput(
        mode="text",
        trace_id=str(row.get("trace_id", "")),
        step=_int_value(row.get("step")),
        agent=str(row.get("agent", "")),
        input_summary=str(input_value.get("content", "")),
        output=str(output.get("content", "")),
    )


def _agent_output_from_protocol_row(row: dict[str, Any]) -> CompareAgentOutput | None:
    output = row.get("output")
    input_value = row.get("input")
    if not isinstance(output, dict) or not isinstance(input_value, dict):
        return None
    result = output.get("result", {})
    return CompareAgentOutput(
        mode="protocol",
        trace_id=str(row.get("trace_id", "")),
        step=_int_value(row.get("step")),
        agent=str(row.get("agent", "")),
        action=str(input_value.get("action", "")),
        input_summary=_json_text(
            {
                "action": input_value.get("action", ""),
                "params": input_value.get("params", {}),
                "state_refs": input_value.get("state_refs", []),
            }
        ),
        output=_json_text(result),
        state_refs_in=_string_list(row.get("state_refs_in")),
        state_refs_out=_string_list(row.get("state_refs_out")),
    )


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _runner_kwargs(
    runner: Callable[..., ModeRunResult],
    **kwargs: Any,
) -> dict[str, Any]:
    if _accepts_kwarg(runner, "trace_id"):
        return kwargs
    return {key: value for key, value in kwargs.items() if key != "trace_id"}


def _accepts_kwarg(runner: Callable[..., ModeRunResult], name: str) -> bool:
    try:
        parameters = signature(runner).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values()
    )
