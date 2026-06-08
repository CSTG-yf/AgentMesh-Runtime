import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import orjson

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
from agentmesh.runtime.decision import PlannerDecision
from agentmesh.runtime.orchestrator import default_registry
from agentmesh.runtime.registry import RuntimeContext
from agentmesh.runtime.scheduler import ProtocolScheduler
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
    return ProtocolRunSession(
        task_path=task_path,
        paths=paths,
        llm_client=llm_client,
        load_configured_llm=load_configured_llm,
    ).run()


class ProtocolRunSession:
    def __init__(
        self,
        *,
        task_path: Path,
        paths: RuntimePaths,
        llm_client: LLMClient | None,
        load_configured_llm: bool,
    ) -> None:
        self.task_path = task_path
        self.paths = paths
        self.llm_client = llm_client
        self.load_configured_llm = load_configured_llm

    def run(self) -> ModeRunResult:
        return _run_protocol_mode_impl(
            task_path=self.task_path,
            paths=self.paths,
            llm_client=self.llm_client,
            load_configured_llm=self.load_configured_llm,
        )


def _run_protocol_mode_impl(
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
    context = RuntimeContext.from_paths(
        paths=paths,
        task_path=task_path,
        trace_id=trace_id,
        llm_client=llm_client,
        load_configured_llm=load_configured_llm,
    )
    state_store = StateStore(
        paths,
        payload_backend=context.config.state.payload_backend,
        shm_threshold_bytes=context.config.state.shm_threshold_bytes,
    )
    encoder = create_embedding_encoder(context.config.embedding)
    memory_store = HybridMemoryStore(paths=paths, state_store=state_store, encoder=encoder)
    registry = default_registry()
    messages: list[AMPMessage] = []
    protocol_map = _protocol_action_map()
    scheduler = ProtocolScheduler(
        registry=registry,
        context=context,
        messages=messages,
        protocol_map=protocol_map,
    )
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
            params={"required": list(protocol_map.values())},
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
                **protocol_map,
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
    planner_result = scheduler.invoke(
        source_agent="runtime",
        action="plan.create",
        params={"task": task},
        state_refs=[task_ref, query_ref],
    )
    decision = _planner_decision(planner_result, task)
    plan_summary = _planner_summary(planner_result)
    plan_ref = state_store.put_summary(
        trace_id=trace_id,
        producer="planner",
        summary=plan_summary,
        parent_state_refs=[task_ref, query_ref],
        metadata={"llm_used": _non_empty_text(planner_result.result.get("llm_plan")) is not None},
    )
    next_agent = "retriever" if decision.need_retrieval else "summarizer"
    if decision.need_tool_execution and not decision.need_retrieval:
        next_agent = "executor"
    messages.append(
        AMPMessage(
            trace_id=trace_id,
            source_agent="planner",
            target_agent=next_agent,
            msg_type=MsgType.STATE_REF,
            action="state.put_summary",
            state_refs=[plan_ref],
            result={"state_count": 1},
        )
    )
    mark_stage("planner", stage_start)

    query_tags = _derive_tags(task)
    memory_hits = []
    evidence: list[dict[str, Any]] = []
    evidence_ref: str | None = None
    refined_plan_ref: str | None = None
    refined_plan_summary = ""
    refined_evidence_ref: str | None = None
    code_result_ref: str | None = None
    code_result_payload: dict[str, Any] | None = None
    code_result_summary = ""
    tool_feedback: dict[str, Any] | None = None
    feedback_round_count = 0
    planner_refine_count = 0
    retriever_refine_count = 0
    tool_feedback_count = 0
    sandbox_backend = ""
    memory_query_count = 0

    if decision.need_retrieval:
        stage_start = time.perf_counter()
        memory_hits = memory_store.semantic_search(task, query_tags=query_tags)
        for hit in memory_hits:
            memory_store.increment_reuse(hit.memory_id)
        memory_query_count = 1
        mark_stage("memory_search", stage_start)

        stage_start = time.perf_counter()
        retriever_result = scheduler.invoke(
            source_agent="planner",
            action="memory.semantic_search",
            params={"query": task},
            state_refs=[task_ref, query_ref, plan_ref],
        )
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
                target_agent="executor" if decision.need_tool_execution else "summarizer",
                msg_type=MsgType.STATE_REF,
                action="state.put_evidence",
                state_refs=[evidence_ref],
                result={"state_count": 1},
            )
        )
        mark_stage("retriever", stage_start)

    if decision.need_tool_execution:
        stage_start = time.perf_counter()
        evidence_digest = _evidence_digest(evidence)
        executor_result = scheduler.invoke(
            source_agent="retriever" if evidence_ref else "planner",
            action="tool.run_python",
            params={"task": task, "evidence": evidence_digest},
            state_refs=[ref for ref in [evidence_ref] if ref],
        )
        mark_stage("executor", stage_start)

        stage_start = time.perf_counter()
        codeact_code = str(executor_result.result.get("codeact_code") or "")
        sandbox_result = SandboxRunner(paths.sandbox_dir).run_python(codeact_code)
        sandbox_backend = sandbox_result.backend
        mark_stage("sandbox", stage_start)

        stage_start = time.perf_counter()
        tool_feedback = _tool_feedback(task, sandbox_result.model_dump())
        tool_feedback_count = 1
        code_result_payload = sandbox_result.model_dump()
        code_result_payload["executor_result"] = executor_result.result
        code_result_payload["tool_feedback"] = tool_feedback
        code_result_payload["codeact"] = {
            "code": codeact_code,
            "generated_by_llm": bool(executor_result.result.get("llm_generated_code")),
            "stdout": sandbox_result.stdout,
            "stderr": sandbox_result.stderr,
            "exit_code": sandbox_result.exit_code,
        }
        parent_refs = [ref for ref in [evidence_ref] if ref]
        code_result_ref = state_store.put_code_result(
            trace_id=trace_id,
            producer="executor",
            result=code_result_payload,
            parent_state_refs=parent_refs,
            metadata={
                "llm_used": bool(executor_result.result.get("llm_generated_code")),
                "codeact": True,
                "tool_feedback": True,
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

        recommended_actions = {
            str(item.get("target_capability", ""))
            for item in tool_feedback.get("recommended_next_actions", [])
            if isinstance(item, dict)
        }
        feedback_refs = [ref for ref in [plan_ref, evidence_ref, code_result_ref] if ref]
        if "plan.refine" in recommended_actions:
            stage_start = time.perf_counter()
            refined_plan_result = scheduler.invoke(
                source_agent="executor",
                action="plan.refine",
                params={"task": task, "tool_feedback": tool_feedback},
                state_refs=feedback_refs,
            )
            refined_plan_ref = state_store.put_summary(
                trace_id=trace_id,
                producer="planner",
                summary="; ".join(refined_plan_result.result.get("refined_plan", [])),
                parent_state_refs=feedback_refs,
                metadata={"refined": True},
            )
            refined_plan_summary = str(
                refined_plan_result.result.get("refined_plan", refined_plan_summary)
            )
            feedback_round_count = 1
            planner_refine_count = 1
            mark_stage("planner_refine", stage_start)
        should_refine_evidence = (
            "evidence.refine" in recommended_actions
            or "memory.semantic_search" in recommended_actions
        )
        if should_refine_evidence:
            stage_start = time.perf_counter()
            refined_evidence_result = scheduler.invoke(
                source_agent="planner" if refined_plan_ref else "executor",
                action="evidence.refine",
                params={"query": task, "tool_feedback": tool_feedback},
                state_refs=feedback_refs + ([refined_plan_ref] if refined_plan_ref else []),
            )
            refined_items = list(refined_evidence_result.result.get("evidence", []))
            evidence.extend(refined_items)
            refined_evidence_ref = state_store.put_evidence(
                trace_id=trace_id,
                producer="retriever",
                evidence=refined_items,
                parent_state_refs=feedback_refs + ([refined_plan_ref] if refined_plan_ref else []),
                metadata={"refined": True},
            )
            feedback_round_count = 1
            retriever_refine_count = 1
            mark_stage("retriever_refine", stage_start)
        code_result_summary = _code_result_summary(code_result_payload)

    stage_start = time.perf_counter()
    summary_state_refs = [
        ref
        for ref in [
            task_ref,
            plan_ref,
            refined_plan_ref,
            evidence_ref,
            refined_evidence_ref,
            code_result_ref,
        ]
        if ref
    ]
    summarizer_source = scheduler.selected_agents[-1] if scheduler.selected_agents else "planner"
    summarizer_result = scheduler.invoke(
        source_agent=summarizer_source,
        action="summary.create",
        params={
            "code_result": code_result_summary,
            "tool_feedback": tool_feedback or {},
            "dynamic_route": scheduler.selected_agents,
            "state_context": {
                "task": task,
                "plan": plan_summary,
                "refined_plan": refined_plan_summary,
                "evidence": evidence,
                "code_result": code_result_summary,
                "tool_feedback": tool_feedback or {},
                "dynamic_route": scheduler.selected_agents,
            },
        },
        state_refs=summary_state_refs,
    )
    model_summary = _non_empty_text(summarizer_result.result.get("llm_summary"))
    summary_text = model_summary or (
        "Protocol Mode answer: structured collaboration completed with "
        f"{len(evidence)} evidence item(s) and route {' -> '.join(scheduler.selected_agents)}."
    )
    summary_ref = state_store.put_summary(
        trace_id=trace_id,
        producer="summarizer",
        summary=summary_text,
        parent_state_refs=summary_state_refs,
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
        evidence_refs=[ref for ref in [evidence_ref, refined_evidence_ref] if ref],
        state_refs=[
            ref
            for ref in [task_ref, plan_ref, refined_plan_ref, code_result_ref, summary_ref]
            if ref
        ],
        embedding_ref=memory_embedding_ref,
        embedding_vector=memory_embedding,
        confidence=0.8,
        validity_score=_validity_score(code_result_payload),
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
    transport_stats = scheduler.transport_metrics()
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
        sandbox_backend=sandbox_backend,
        transport_type=transport_stats.transport_type,
        transport_send_count=transport_stats.send_count,
        transport_bytes=transport_stats.total_bytes,
        transport_avg_latency_ms=transport_stats.avg_latency_ms,
        transport_p99_latency_ms=transport_stats.p99_latency_ms,
        state_shm_transfer_count=state_store.shm_transfer_count(trace_id),
        state_shm_transfer_bytes=state_store.shm_transfer_bytes(trace_id),
        memory_query_count=memory_query_count,
        memory_hit_count=1 if memory_hits else 0,
        memory_query_hit_count=1 if memory_hits else 0,
        memory_reused_unit_count=len(memory_hits),
        latency_ms=latency_ms,
        stage_latency_ms=stage_latency_ms,
        answer_quality_score=deterministic_quality_score(summary_text),
        dynamic_route=scheduler.selected_agents,
        selected_agents=list(dict.fromkeys(scheduler.selected_agents)),
        skipped_agents=[
            agent
            for agent in ["planner", "retriever", "executor", "summarizer"]
            if agent not in scheduler.selected_agents
        ],
        feedback_round_count=feedback_round_count,
        planner_refine_count=planner_refine_count,
        retriever_refine_count=retriever_refine_count,
        tool_feedback_count=tool_feedback_count,
    )
    append_jsonl(paths.protocol_trace, {"trace_id": trace_id, "metrics": metrics.model_dump()})
    state_store.close()
    return ModeRunResult(mode="protocol", trace_id=trace_id, answer=summary_text, metrics=metrics)


def _derive_tags(text: str) -> list[str]:
    lowered = text.lower()
    tags: list[str] = []
    for candidate in ["protocol", "state", "memory", "benchmark", "agent", "runtime"]:
        if candidate in lowered:
            tags.append(candidate)
    return tags or ["general"]


def _protocol_action_map() -> dict[str, str]:
    return {
        "planner": "plan.create",
        "retriever": "memory.semantic_search",
        "executor": "tool.run_python",
        "summarizer": "summary.create",
        "planner_refine": "plan.refine",
        "retriever_refine": "evidence.refine",
    }


def _planner_decision(message: AMPMessage, task: str) -> PlannerDecision:
    try:
        return PlannerDecision.model_validate(message.result.get("decision")).normalized()
    except Exception:
        return PlannerDecision.from_task(task)


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


def _tool_feedback(task: str, sandbox_payload: dict[str, Any]) -> dict[str, Any]:
    stdout = str(sandbox_payload.get("stdout", "")).strip()
    stderr = str(sandbox_payload.get("stderr", "")).strip()
    exit_code = int(sandbox_payload.get("exit_code", 1))
    parsed_stdout = _stdout_json(stdout)
    evidence_gaps = _string_list(parsed_stdout.get("evidence_gaps"))
    validated_claims = _string_list(parsed_stdout.get("validated_claims"))
    failed_claims = _string_list(parsed_stdout.get("failed_claims"))
    recommended_next_actions = _action_list(parsed_stdout.get("recommended_next_actions"))
    if _requires_feedback_refine(task):
        _append_unique(evidence_gaps, "verify benchmark baseline and communication metrics")
    if exit_code == 0:
        _append_unique(validated_claims, "sandbox execution completed")
    else:
        _append_unique(failed_claims, "sandbox execution failed")
    if evidence_gaps and not _has_capability(recommended_next_actions, "plan.refine"):
        recommended_next_actions.extend(
            [
                {
                    "target_capability": "plan.refine",
                    "reason": "tool feedback identified missing validation context",
                },
                {
                    "target_capability": "evidence.refine",
                    "reason": "evidence gaps should be checked before summarization",
                },
            ]
        )
    return {
        "status": "success" if exit_code == 0 else "error",
        "exit_code": exit_code,
        "stdout_summary": stdout[:500],
        "stderr_summary": stderr[:300],
        "validated_claims": validated_claims,
        "failed_claims": failed_claims,
        "evidence_gaps": evidence_gaps,
        "recommended_next_actions": recommended_next_actions,
    }


def _requires_feedback_refine(task: str) -> bool:
    lowered = task.lower()
    return any(word in lowered for word in ["benchmark", "验证", "评测", "通信开销"])


def _stdout_json(stdout: str) -> dict[str, Any]:
    if not stdout:
        return {}
    try:
        loaded = orjson.loads(stdout)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _action_list(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    actions: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        capability = item.get("target_capability")
        if not isinstance(capability, str) or not capability:
            continue
        actions.append(
            {
                "target_capability": capability,
                "reason": str(item.get("reason", "tool requested follow-up")),
            }
        )
    return actions


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _has_capability(actions: list[dict[str, str]], capability: str) -> bool:
    return any(action.get("target_capability") == capability for action in actions)


def _validity_score(code_result_payload: dict[str, Any] | None) -> float:
    if code_result_payload is None:
        return 1.0
    try:
        return 1.0 if int(code_result_payload.get("exit_code", 1)) == 0 else 0.3
    except Exception:
        return 0.3


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
