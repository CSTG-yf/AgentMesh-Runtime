import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from agentmesh.core import rust_available
from agentmesh.eval.metrics import ModeRunResult, RunMetrics, estimate_tokens
from agentmesh.eval.quality import deterministic_quality_score
from agentmesh.llm.client import LLMClient
from agentmesh.memory.hybrid_store import HybridMemoryStore
from agentmesh.memory.schema import MemoryUnit
from agentmesh.memory.tagger import LLMMemoryTagger
from agentmesh.protocol.codec import (
    encode_message,
    encode_message_compact,
    encode_payload_compact,
)
from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.envelope import measure_typed_envelopes
from agentmesh.protocol.schema import AMPMessage
from agentmesh.runtime.orchestrator import default_registry
from agentmesh.runtime.registry import RuntimeContext
from agentmesh.sandbox.runner import SandboxRunner
from agentmesh.state.embedding import create_embedding_encoder
from agentmesh.state.store import StateStore
from agentmesh.storage.jsonl import append_jsonl
from agentmesh.storage.paths import RuntimePaths


def run_protocol_mode(
    task_path: Path,
    paths: RuntimePaths,
    llm_client: LLMClient | None = None,
    load_configured_llm: bool = True,
) -> ModeRunResult:
    paths.ensure()
    trace_id = f"trace-{uuid4().hex[:12]}"
    task = task_path.read_text(encoding="utf-8")
    start = time.perf_counter()
    stage_latency_ms: dict[str, int] = {}

    def mark_stage(name: str, stage_start: float) -> None:
        stage_latency_ms[name] = int((time.perf_counter() - stage_start) * 1000)

    stage_start = time.perf_counter()
    state_store = StateStore(paths)
    context = RuntimeContext.from_paths(
        paths=paths,
        task_path=task_path,
        trace_id=trace_id,
        llm_client=llm_client,
        load_configured_llm=load_configured_llm,
    )
    encoder = create_embedding_encoder(context.config.embedding)
    memory_store = HybridMemoryStore(paths=paths, state_store=state_store, encoder=encoder)
    registry = default_registry()
    messages: list[AMPMessage] = []
    mark_stage("setup", stage_start)

    stage_start = time.perf_counter()
    for agent in registry.all():
        messages.extend([agent.hello(trace_id), agent.advertise_capabilities(trace_id)])
    messages.append(
        AMPMessage(
            trace_id=trace_id,
            source_agent="runtime",
            target_agent="runtime",
            msg_type=MsgType.CAPABILITY_QUERY,
            action="system.capability_query",
            params={"required": ["plan.create", "memory.semantic_search", "tool.run_python"]},
        )
    )
    messages.append(
        AMPMessage(
            trace_id=trace_id,
            source_agent="runtime",
            target_agent="runtime",
            msg_type=MsgType.PROTOCOL_MAP,
            action="protocol.map",
            result={
                "planner": "plan.create",
                "retriever": "memory.semantic_search",
                "executor": "tool.run_python",
                "summarizer": "summary.create",
                "llm_configured": context.config.llm.configured,
                "llm_client_available": context.llm_client is not None,
            },
        )
    )

    task_ref = state_store.put_text(
        trace_id=trace_id,
        producer="runtime",
        text=task,
        metadata={"task_path": str(task_path)},
    )
    query_embedding = encoder.encode(task)
    query_ref = state_store.put_embedding(
        trace_id=trace_id,
        producer="runtime",
        embedding=query_embedding,
        parent_state_refs=[task_ref],
    )
    messages.append(
        AMPMessage(
            trace_id=trace_id,
            source_agent="runtime",
            target_agent="planner",
            msg_type=MsgType.STATE_REF,
            action="state.put_text",
            state_refs=[task_ref, query_ref],
            result={"state_count": 2},
        )
    )
    mark_stage("state_task_embedding", stage_start)

    stage_start = time.perf_counter()
    planner_invoke = AMPMessage(
        trace_id=trace_id,
        source_agent="runtime",
        target_agent="planner",
        msg_type=MsgType.INVOKE,
        action="plan.create",
        params={"task": task},
        state_refs=[task_ref, query_ref],
    )
    messages.append(planner_invoke)
    planner_result = registry.get("planner").handle(planner_invoke, context)
    messages.append(planner_result)
    plan_ref = state_store.put_summary(
        trace_id=trace_id,
        producer="planner",
        summary=_planner_summary(planner_result),
        parent_state_refs=[task_ref, query_ref],
        metadata={"llm_used": _non_empty_text(planner_result.result.get("llm_plan")) is not None},
    )
    messages.append(
        AMPMessage(
            trace_id=trace_id,
            source_agent="planner",
            target_agent="retriever",
            msg_type=MsgType.STATE_REF,
            action="state.put_summary",
            state_refs=[plan_ref],
            result={"state_count": 1},
        )
    )
    mark_stage("planner", stage_start)

    stage_start = time.perf_counter()
    query_tags = _derive_tags(task)
    memory_hits = memory_store.semantic_search(task, query_tags=query_tags)
    for hit in memory_hits:
        memory_store.increment_reuse(hit.memory_id)
    mark_stage("memory_search", stage_start)

    stage_start = time.perf_counter()
    retriever_invoke = AMPMessage(
        trace_id=trace_id,
        source_agent="planner",
        target_agent="retriever",
        msg_type=MsgType.INVOKE,
        action="memory.semantic_search",
        params={"query": task},
        state_refs=[task_ref, query_ref, plan_ref],
    )
    messages.append(retriever_invoke)
    retriever_result = registry.get("retriever").handle(retriever_invoke, context)
    messages.append(retriever_result)
    evidence = list(retriever_result.result["evidence"])
    evidence.extend(
        {"title": unit.task_topic, "snippet": unit.summary, "memory_id": unit.memory_id}
        for unit in memory_hits
    )
    evidence_ref = state_store.put_evidence(
        trace_id=trace_id,
        producer="retriever",
        evidence=evidence,
        parent_state_refs=[task_ref, query_ref, plan_ref],
        metadata={
            "llm_used": any(item.get("title") == "llm-evidence" for item in evidence),
        },
    )
    messages.append(
        AMPMessage(
            trace_id=trace_id,
            source_agent="retriever",
            target_agent="executor",
            msg_type=MsgType.STATE_REF,
            action="state.put_evidence",
            state_refs=[evidence_ref],
            result={"state_count": 1},
        )
    )
    mark_stage("retriever", stage_start)

    stage_start = time.perf_counter()
    evidence_digest = _evidence_digest(evidence)
    executor_invoke = AMPMessage(
        trace_id=trace_id,
        source_agent="retriever",
        target_agent="executor",
        msg_type=MsgType.INVOKE,
        action="tool.run_python",
        params={"task": task, "evidence": evidence_digest},
        state_refs=[evidence_ref],
    )
    messages.append(executor_invoke)
    executor_result = registry.get("executor").handle(executor_invoke, context)
    messages.append(executor_result)
    mark_stage("executor", stage_start)

    stage_start = time.perf_counter()
    codeact_code = str(executor_result.result.get("codeact_code") or "")
    sandbox_result = SandboxRunner(paths.sandbox_dir).run_python(codeact_code)
    mark_stage("sandbox", stage_start)

    stage_start = time.perf_counter()
    code_result_payload = sandbox_result.model_dump()
    code_result_payload["executor_result"] = executor_result.result
    code_result_payload["codeact"] = {
        "code": codeact_code,
        "generated_by_llm": bool(executor_result.result.get("llm_generated_code")),
        "stdout": sandbox_result.stdout,
        "stderr": sandbox_result.stderr,
        "exit_code": sandbox_result.exit_code,
    }
    code_result_ref = state_store.put_code_result(
        trace_id=trace_id,
        producer="executor",
        result=code_result_payload,
        parent_state_refs=[evidence_ref],
        metadata={
            "llm_used": bool(executor_result.result.get("llm_generated_code")),
            "codeact": True,
        },
    )
    messages.append(
        AMPMessage(
            trace_id=trace_id,
            source_agent="executor",
            target_agent="summarizer",
            msg_type=MsgType.STATE_REF,
            action="state.put_code_result",
            state_refs=[code_result_ref],
            result={"state_count": 1},
        )
    )
    mark_stage("code_result_state", stage_start)

    stage_start = time.perf_counter()
    code_result_summary = _code_result_summary(code_result_payload)
    summarizer_invoke = AMPMessage(
        trace_id=trace_id,
        source_agent="executor",
        target_agent="summarizer",
        msg_type=MsgType.INVOKE,
        action="summary.create",
        params={"code_result": code_result_summary},
        state_refs=[task_ref, plan_ref, evidence_ref, code_result_ref],
    )
    messages.append(summarizer_invoke)
    summarizer_result = registry.get("summarizer").handle(summarizer_invoke, context)
    messages.append(summarizer_result)
    model_summary = _non_empty_text(summarizer_result.result.get("llm_summary"))
    summary_text = model_summary or (
        "Protocol Mode answer: structured collaboration completed with "
        f"{len(evidence)} evidence item(s) and sandbox exit {sandbox_result.exit_code}."
    )
    summary_ref = state_store.put_summary(
        trace_id=trace_id,
        producer="summarizer",
        summary=summary_text,
        parent_state_refs=[task_ref, plan_ref, evidence_ref, code_result_ref],
        metadata={"llm_used": model_summary is not None},
    )
    memory_embedding = encoder.encode(summary_text)
    memory_embedding_ref = state_store.put_embedding(
        trace_id=trace_id,
        producer="summarizer",
        embedding=memory_embedding,
        parent_state_refs=[summary_ref],
    )
    mark_stage("summarizer", stage_start)

    stage_start = time.perf_counter()
    classification = LLMMemoryTagger(context.llm_client).classify(
        task=task,
        summary=summary_text,
        evidence_count=len(evidence),
    )
    unit = MemoryUnit(
        source_agent="summarizer",
        task_topic=classification.topic,
        summary=summary_text,
        tags=classification.tags,
        evidence_refs=[evidence_ref],
        state_refs=[task_ref, plan_ref, code_result_ref, summary_ref],
        embedding_ref=memory_embedding_ref,
        embedding_vector=memory_embedding,
        confidence=0.8,
        validity_score=1.0 if sandbox_result.exit_code == 0 else 0.3,
        provenance_trace_id=trace_id,
        importance_score=classification.importance_score,
        memory_type=classification.memory_type,
        domain=classification.domain,
    )
    write_result = memory_store.put(unit)
    if write_result["run_written"]:
        messages.append(
            AMPMessage(
                trace_id=trace_id,
                source_agent="summarizer",
                target_agent="runtime",
                msg_type=MsgType.MEMORY_PUT,
                action="memory.put",
                result={
                    "memory_id": unit.memory_id,
                    "long_term_written": write_result["global_written"],
                    "importance_score": unit.importance_score,
                },
                state_refs=unit.state_refs + unit.evidence_refs + [memory_embedding_ref],
            )
        )
    mark_stage("memory_write", stage_start)

    stage_start = time.perf_counter()
    for message in messages:
        append_jsonl(paths.protocol_messages, message.model_dump(mode="json"))
    protocol_bytes = sum(len(encode_message(message)) for message in messages)
    compact_protocol_bytes = sum(len(encode_message_compact(message)) for message in messages)
    typed_envelope_stats = measure_typed_envelopes(messages)
    handoff_bytes = _handoff_wire_bytes(messages)
    state_records = state_store.list_by_trace(trace_id)
    state_transfer_bytes = sum(record.size_bytes for record in state_records)
    mark_stage("artifact_write", stage_start)
    latency_ms = int((time.perf_counter() - start) * 1000)
    metrics = RunMetrics(
        message_count=len(messages),
        text_chars=len(task),
        estimated_tokens=estimate_tokens(task),
        communication_model="structured_state_ref",
        wire_bytes=typed_envelope_stats.wire_bytes,
        structured_handoff_bytes=handoff_bytes,
        session_dictionary_bytes=typed_envelope_stats.session_dictionary_bytes,
        typed_envelope_bytes=typed_envelope_stats.typed_envelope_bytes,
        typed_payload_bytes=typed_envelope_stats.typed_payload_bytes,
        structured_message_bytes=protocol_bytes,
        compact_structured_message_bytes=compact_protocol_bytes,
        protocol_bytes=protocol_bytes,
        state_transfer_count=len(state_records),
        state_transfer_bytes=state_transfer_bytes,
        rust_core_enabled=rust_available(),
        sandbox_backend=sandbox_result.backend,
        memory_query_count=1,
        memory_hit_count=1 if memory_hits else 0,
        latency_ms=latency_ms,
        stage_latency_ms=stage_latency_ms,
        answer_quality_score=deterministic_quality_score(summary_text),
    )
    append_jsonl(paths.protocol_trace, {"trace_id": trace_id, "metrics": metrics.model_dump()})
    return ModeRunResult(mode="protocol", trace_id=trace_id, answer=summary_text, metrics=metrics)


def _derive_tags(text: str) -> list[str]:
    lowered = text.lower()
    tags: list[str] = []
    for candidate in ["protocol", "state", "memory", "benchmark", "agent", "runtime"]:
        if candidate in lowered:
            tags.append(candidate)
    return tags or ["general"]


def _planner_summary(message: AMPMessage) -> str:
    steps = message.result.get("plan", [])
    if not isinstance(steps, list):
        steps = []
    deterministic_plan = "; ".join(str(step) for step in steps)
    llm_plan = _non_empty_text(message.result.get("llm_plan"))
    if llm_plan is None:
        return deterministic_plan
    if not deterministic_plan:
        return llm_plan
    return f"{deterministic_plan}\n\nLLM plan:\n{llm_plan}"


def _non_empty_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _handoff_wire_bytes(messages: list[AMPMessage]) -> int:
    total = 0
    for message in messages:
        if message.msg_type != MsgType.STATE_REF or not message.state_refs:
            continue
        total += len(
            encode_payload_compact(
                {
                    "source_agent": message.source_agent,
                    "target_agent": message.target_agent,
                    "msg_type": message.msg_type.value,
                    "action": message.action,
                    "state_refs": message.state_refs,
                }
            )
        )
    return total


def _evidence_digest(evidence: list[dict[str, Any]]) -> str:
    snippets: list[str] = []
    for item in evidence[:5]:
        title = str(item.get("title", "evidence"))
        snippet = str(item.get("snippet", ""))[:240]
        snippets.append(f"{title}: {snippet}")
    return "\n".join(snippets)


def _code_result_summary(payload: dict[str, Any]) -> str:
    codeact = payload.get("codeact")
    if not isinstance(codeact, dict):
        return "CodeAct execution result unavailable."
    stdout = str(codeact.get("stdout", "")).strip()
    stderr = str(codeact.get("stderr", "")).strip()
    exit_code = codeact.get("exit_code")
    return (
        f"CodeAct exit_code={exit_code}; "
        f"stdout={stdout[:500] or '<empty>'}; "
        f"stderr={stderr[:300] or '<empty>'}"
    )
