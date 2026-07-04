import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import orjson

from agentmesh.core import rust_available
from agentmesh.errors import SandboxTimeoutError
from agentmesh.eval.llm_experiment import LLMExperimentProfile
from agentmesh.eval.metrics import ModeRunResult, RunMetrics, estimate_tokens
from agentmesh.llm.client import InstrumentedLLMClient, LLMClient
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
from agentmesh.runtime.context_budget import ContextAudit, ContextPart, pack_context
from agentmesh.runtime.decision import PlannerDecision
from agentmesh.runtime.orchestrator import default_registry
from agentmesh.runtime.registry import RuntimeContext
from agentmesh.runtime.scheduler import ProtocolScheduler
from agentmesh.sandbox.runner import SandboxResult, SandboxRunner
from agentmesh.state.embedding import create_embedding_encoder
from agentmesh.state.store import StateStore
from agentmesh.storage.jsonl import append_jsonl, read_jsonl
from agentmesh.storage.paths import RuntimePaths


def run_protocol_mode(
    task_path: Path,
    paths: RuntimePaths,
    llm_client: LLMClient | None = None,
    load_configured_llm: bool = True,
    trace_id: str | None = None,
    experiment_profile: LLMExperimentProfile | None = None,
) -> ModeRunResult:
    return ProtocolRunSession(
        task_path=task_path,
        paths=paths,
        llm_client=llm_client,
        load_configured_llm=load_configured_llm,
        trace_id=trace_id,
        experiment_profile=experiment_profile,
    ).run()


class ProtocolRunSession:
    def __init__(
        self,
        *,
        task_path: Path,
        paths: RuntimePaths,
        llm_client: LLMClient | None,
        load_configured_llm: bool,
        trace_id: str | None,
        experiment_profile: LLMExperimentProfile | None = None,
    ) -> None:
        self.task_path = task_path
        self.paths = paths
        self.llm_client = llm_client
        self.load_configured_llm = load_configured_llm
        self.trace_id = trace_id
        self.experiment_profile = experiment_profile

    def run(self) -> ModeRunResult:
        return _run_protocol_mode_impl(
            task_path=self.task_path,
            paths=self.paths,
            llm_client=self.llm_client,
            load_configured_llm=self.load_configured_llm,
            trace_id=self.trace_id,
            experiment_profile=self.experiment_profile,
        )


def _run_protocol_mode_impl(
    task_path: Path,
    paths: RuntimePaths,
    llm_client: LLMClient | None = None,
    load_configured_llm: bool = True,
    trace_id: str | None = None,
    experiment_profile: LLMExperimentProfile | None = None,
) -> ModeRunResult:
    paths.ensure()
    trace_id = trace_id or f"trace-{uuid4().hex[:12]}"
    task = task_path.read_text(encoding="utf-8-sig")
    start = time.perf_counter()
    stage_latency_ms: dict[str, int] = {}
    context_audits: list[ContextAudit] = []
    optimization_enabled = bool(
        experiment_profile is not None and experiment_profile.optimization_enabled
    )

    def budget_context(
        role: str,
        max_chars: int,
        parts: list[ContextPart],
    ) -> str:
        packed = pack_context(role=role, max_chars=max_chars, parts=parts)
        context_audits.append(packed.audit)
        return packed.text

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
    instrumented = (
        InstrumentedLLMClient(context.llm_client)
        if context.llm_client is not None
        else None
    )
    if instrumented is not None:
        context = context.model_copy(update={"llm_client": instrumented})
    protocol_budget = context.config.protocol
    state_store = StateStore(
        paths,
        payload_backend=context.config.state.payload_backend,
        shm_threshold_bytes=context.config.state.shm_threshold_bytes,
    )
    encoder = create_embedding_encoder(context.config.embedding)
    memory_store = HybridMemoryStore(paths=paths, state_store=state_store, encoder=encoder)
    registry = default_registry()
    context = context.model_copy(update={
        "capability_to_agent": registry.capability_owner_map(),
        "state_store": state_store,
        "embedding_encoder": encoder,
        "memory_store": memory_store,
    })
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
    if not context.config.protocol.skip_handshake_for_inproc:
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
    messages.append(
        AMPMessage(
            trace_id=trace_id,
            source_agent="runtime",
            target_agent="planner",
            msg_type=MsgType.STATE_REF,
            action="state.put_text",
            state_refs=[task_ref],
            result={"state_count": 1},
        )
    )
    mark_stage("state_task_embedding", stage_start)

    stage_start = time.perf_counter()
    planner_task = (
        budget_context(
            "planner",
            experiment_profile.protocol.planner_max_chars,
            [ContextPart(name="task", text=task, required=True)],
        )
        if optimization_enabled and experiment_profile is not None
        else _compact_text(task, 300)
    )
    planner_result = scheduler.invoke(
        source_agent="runtime",
        action="plan.create",
        params={"task_chars": len(task), "task": planner_task},
        state_refs=[task_ref],
    )
    decision = _planner_decision(planner_result, task, context.capability_to_agent)
    plan_summary = _planner_summary(planner_result)
    planner_direct_answer = _planner_direct_answer(planner_result, decision)
    plan_ref = state_store.put_summary(
        trace_id=trace_id,
        producer="planner",
        summary=_compact_text(plan_summary, protocol_budget.state_summary_max_chars),
        parent_state_refs=[task_ref],
        metadata={
            "llm_used": _non_empty_text(planner_result.result.get("llm_plan")) is not None,
            "full_chars": len(plan_summary),
            "compacted": len(plan_summary) > protocol_budget.state_summary_max_chars,
        },
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

    memory_hits: list[str] = []
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
        memory_query_count = 1
        mark_stage("memory_search", stage_start)

        stage_start = time.perf_counter()
        retriever_query = (
            budget_context(
                "retriever",
                experiment_profile.protocol.retriever_max_chars,
                [
                    ContextPart(name="task", text=task, required=True),
                    ContextPart(
                        name="planner_intent",
                        text=plan_summary,
                        required=True,
                    ),
                ],
            )
            if optimization_enabled and experiment_profile is not None
            else _compact_text(task, 300)
        )
        retriever_result = scheduler.invoke(
            source_agent="planner",
            action="memory.semantic_search",
            params={"query": retriever_query, "task_chars": len(task)},
            state_refs=[task_ref, plan_ref],
        )
        evidence = list(retriever_result.result["evidence"])
        memory_hits = _memory_units_from_evidence(evidence)
        compact_evidence = _compact_evidence_items(
            evidence,
            snippet_limit=protocol_budget.evidence_snippet_max_chars,
        )
        evidence_ref = state_store.put_evidence(
            trace_id=trace_id,
            producer="retriever",
            evidence=compact_evidence,
            parent_state_refs=[task_ref, plan_ref],
            metadata={
                "llm_used": any(item.get("title") == "llm-evidence" for item in evidence),
                "full_count": len(evidence),
                "compacted": True,
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
        evidence_digest = _evidence_digest(
            evidence,
            snippet_limit=protocol_budget.evidence_snippet_max_chars,
            max_items=3,
        )
        executor_task = (
            budget_context(
                "executor",
                experiment_profile.protocol.executor_max_chars,
                [
                    ContextPart(name="task", text=task, required=True),
                    ContextPart(
                        name="execution_requirements",
                        text="Execute and validate the task safely.",
                        required=True,
                    ),
                    ContextPart(name="evidence_digest", text=evidence_digest),
                ],
            )
            if optimization_enabled and experiment_profile is not None
            else _compact_text(task, protocol_budget.agent_log_param_max_chars)
        )
        executor_result = scheduler.invoke(
            source_agent="retriever" if evidence_ref else "planner",
            action="tool.run_python",
            params={
                "task": executor_task,
                "evidence": "" if optimization_enabled else evidence_digest,
                "evidence_count": len(evidence),
                **(
                    {"_disable_state_fallback": True}
                    if optimization_enabled
                    else {}
                ),
            },
            state_refs=[ref for ref in [task_ref, evidence_ref] if ref],
        )
        mark_stage("executor", stage_start)

        stage_start = time.perf_counter()
        codeact_code = str(executor_result.result.get("codeact_code") or "")
        try:
            sandbox_result = SandboxRunner(paths.sandbox_dir).run_python(codeact_code)
        except SandboxTimeoutError as exc:
            sandbox_result = SandboxResult(
                stdout="",
                stderr=str(exc),
                exit_code=124,
                latency_ms=0,
                backend="timeout",
            )
        sandbox_backend = sandbox_result.backend
        mark_stage("sandbox", stage_start)

        stage_start = time.perf_counter()
        tool_feedback = _tool_feedback(task, sandbox_result.model_dump())
        tool_feedback_count = 1
        code_result_payload = sandbox_result.model_dump()
        generated_file_writes = _write_generated_files(
            root=paths.root,
            files=executor_result.result.get("generated_files"),
        )
        code_result_payload["executor_result"] = executor_result.result
        code_result_payload["generated_files"] = generated_file_writes
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
            result=_compact_code_result_payload(
                code_result_payload,
                text_limit=protocol_budget.state_summary_max_chars,
            ),
            parent_state_refs=parent_refs,
            metadata={
                "llm_used": bool(executor_result.result.get("llm_generated_code")),
                "codeact": True,
                "tool_feedback": True,
                "full_stdout_chars": len(str(code_result_payload.get("stdout", ""))),
                "full_stderr_chars": len(str(code_result_payload.get("stderr", ""))),
                "compacted": True,
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
            planner_refine_task = (
                budget_context(
                    "planner_refine",
                    experiment_profile.protocol.planner_max_chars,
                    [
                        ContextPart(name="task", text=task, required=True),
                        *_refinement_feedback_parts(tool_feedback),
                    ],
                )
                if optimization_enabled and experiment_profile is not None
                else _compact_text(task, 300)
            )
            refined_plan_result = scheduler.invoke(
                source_agent="executor",
                action="plan.refine",
                params={
                    "task": planner_refine_task,
                    "tool_feedback": {} if optimization_enabled else tool_feedback,
                },
                state_refs=[task_ref] + feedback_refs,
            )
            refined_plan_ref = state_store.put_summary(
                trace_id=trace_id,
                producer="planner",
                summary=_compact_text(
                    "; ".join(refined_plan_result.result.get("refined_plan", [])),
                    protocol_budget.state_summary_max_chars,
                ),
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
            retriever_refine_query = (
                budget_context(
                    "retriever_refine",
                    experiment_profile.protocol.retriever_max_chars,
                    [
                        ContextPart(name="task", text=task, required=True),
                        *_refinement_feedback_parts(tool_feedback),
                    ],
                )
                if optimization_enabled and experiment_profile is not None
                else _compact_text(task, 300)
            )
            refined_evidence_result = scheduler.invoke(
                source_agent="planner" if refined_plan_ref else "executor",
                action="evidence.refine",
                params={
                    "query": retriever_refine_query,
                    "tool_feedback": {} if optimization_enabled else tool_feedback,
                },
                state_refs=(
                    [task_ref]
                    + feedback_refs
                    + ([refined_plan_ref] if refined_plan_ref else [])
                ),
            )
            refined_items = list(refined_evidence_result.result.get("evidence", []))
            evidence.extend(refined_items)
            refined_evidence_ref = state_store.put_evidence(
                trace_id=trace_id,
                producer="retriever",
                evidence=_compact_evidence_items(
                    refined_items,
                    snippet_limit=protocol_budget.evidence_snippet_max_chars,
                ),
                parent_state_refs=feedback_refs + ([refined_plan_ref] if refined_plan_ref else []),
                metadata={"refined": True, "full_count": len(refined_items), "compacted": True},
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
    summarizer_evidence = _evidence_digest(
        evidence,
        snippet_limit=protocol_budget.evidence_snippet_max_chars,
        max_items=3,
    )
    summarizer_task = (
        budget_context(
            "summarizer",
            experiment_profile.protocol.summarizer_max_chars,
            [
                ContextPart(name="task", text=task, required=True),
                ContextPart(
                    name="execution_result",
                    text=code_result_summary or "No tool execution result.",
                    required=True,
                ),
                ContextPart(name="evidence_digest", text=summarizer_evidence),
            ],
        )
        if optimization_enabled and experiment_profile is not None
        else _compact_text(task, 500)
    )
    summarizer_result = scheduler.invoke(
        source_agent=summarizer_source,
        action="summary.create",
        params={
            "task": summarizer_task,
            "evidence_digest": "" if optimization_enabled else summarizer_evidence,
            "evidence_count": len(evidence),
            "code_result": "" if optimization_enabled else code_result_summary,
            "tool_feedback": tool_feedback or {},
            "dynamic_route": scheduler.selected_agents,
            **(
                {"_disable_state_fallback": True}
                if optimization_enabled
                else {}
            ),
        },
        state_refs=summary_state_refs,
    )
    model_summary = _non_empty_text(summarizer_result.result.get("llm_summary"))
    summary_text = model_summary or planner_direct_answer or (
        _fallback_summary(
            evidence_count=len(evidence),
            route=scheduler.selected_agents,
            code_result_summary=code_result_summary,
        )
    )
    summary_ref = state_store.put_summary(
        trace_id=trace_id,
        producer="summarizer",
        summary=_compact_text(summary_text, protocol_budget.state_summary_max_chars),
        parent_state_refs=summary_state_refs,
        metadata={
            "llm_used": model_summary is not None,
            "full_chars": len(summary_text),
            "compacted": len(summary_text) > protocol_budget.state_summary_max_chars,
        },
    )
    compact_memory_summary = _compact_text(summary_text, 320)
    memory_content = summary_text
    classification = LLMMemoryTagger(context.llm_client).classify(
        task=task,
        summary=summary_text,
        evidence_count=len(evidence),
    )
    # Embed topic + summary together so the vector captures both "what task"
    # and "what answer", making it retrievable from either direction.
    embedding_source = f"{classification.topic}: {compact_memory_summary}"
    memory_embedding = encoder.encode(embedding_source)
    mark_stage("summarizer", stage_start)

    stage_start = time.perf_counter()
    unit = MemoryUnit(
        source_agent="summarizer",
        task_topic=classification.topic,
        summary=compact_memory_summary,
        content=memory_content,
        tags=classification.tags,
        evidence_refs=[ref for ref in [evidence_ref, refined_evidence_ref] if ref],
        state_refs=[
            ref
            for ref in [task_ref, plan_ref, refined_plan_ref, code_result_ref, summary_ref]
            if ref
        ],
        embedding_ref=None,
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
                state_refs=unit.state_refs + unit.evidence_refs,
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
    agent_io_tokens, agent_io_bytes = _protocol_agent_io_stats(
        paths=paths,
        trace_id=trace_id,
    )
    transport_stats = scheduler.transport_metrics()
    memory_quality = _memory_quality_from_log(paths=paths, trace_id=trace_id)
    memory_evidence_bytes = _memory_evidence_bytes(evidence)
    mark_stage("artifact_write", stage_start)
    latency_ms = int((time.perf_counter() - start) * 1000)
    metrics = RunMetrics(
        llm_call_count=instrumented.stats.call_count if instrumented else 0,
        llm_error_count=instrumented.stats.error_count if instrumented else 0,
        message_count=len(messages),
        text_chars=len(task),
        estimated_tokens=agent_io_tokens or estimate_tokens(task),
        communication_model="structured_state_ref",
        wire_bytes=typed_envelope_stats.wire_bytes,
        agent_io_tokens=agent_io_tokens,
        agent_io_bytes=agent_io_bytes,
        per_msg_avg_tokens=(agent_io_tokens / len(messages) if messages else 0.0),
        protocol_total_bytes=typed_envelope_stats.wire_bytes + state_transfer_bytes,
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
        memory_hit_count=len(memory_hits),
        memory_query_hit_count=1 if memory_hits else 0,
        memory_reused_unit_count=len(memory_hits),
        memory_evidence_count=len(memory_hits),
        memory_evidence_bytes=memory_evidence_bytes,
        memory_avg_score=memory_quality["avg_score"],
        memory_avg_semantic_similarity=memory_quality["avg_semantic_similarity"],
        memory_avg_tag_overlap_score=memory_quality["avg_tag_overlap_score"],
        latency_ms=latency_ms,
        stage_latency_ms=stage_latency_ms,
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
        context_audits=[audit.model_dump() for audit in context_audits],
        context_original_chars=sum(audit.original_chars for audit in context_audits),
        context_retained_chars=sum(audit.retained_chars for audit in context_audits),
        context_safe_fallback_count=sum(
            1 for audit in context_audits if audit.safe_fallback
        ),
    )
    append_jsonl(paths.protocol_trace, {"trace_id": trace_id, "metrics": metrics.model_dump()})
    state_store.close()
    return ModeRunResult(mode="protocol", trace_id=trace_id, answer=summary_text, metrics=metrics)


def _memory_units_from_evidence(evidence: list[dict[str, Any]]) -> list[str]:
    memory_ids: list[str] = []
    for item in evidence:
        memory_id = item.get("memory_id")
        if isinstance(memory_id, str) and memory_id:
            memory_ids.append(memory_id)
    return list(dict.fromkeys(memory_ids))


def _memory_evidence_bytes(evidence: list[dict[str, Any]]) -> int:
    return sum(len(orjson.dumps(item)) for item in evidence if item.get("memory_id"))


def _memory_quality_from_log(*, paths: RuntimePaths, trace_id: str) -> dict[str, float]:
    hits: list[dict[str, Any]] = []
    for row in read_jsonl(paths.protocol_memory_hits):
        if str(row.get("trace_id", "")) != trace_id:
            continue
        row_hits = row.get("memory_hits")
        if isinstance(row_hits, list):
            hits.extend(item for item in row_hits if isinstance(item, dict))
    if not hits:
        return _empty_memory_quality()
    return {
        "avg_score": sum(float(item.get("score", 0.0)) for item in hits) / len(hits),
        "avg_semantic_similarity": (
            sum(float(item.get("semantic_similarity", 0.0)) for item in hits) / len(hits)
        ),
        "avg_tag_overlap_score": (
            sum(float(item.get("tag_overlap_score", 0.0)) for item in hits) / len(hits)
        ),
    }


def _memory_quality_from_evidence(evidence: list[dict[str, Any]]) -> dict[str, float]:
    memory_items = [item for item in evidence if item.get("memory_id")]
    if not memory_items:
        return _empty_memory_quality()
    return {
        "avg_score": _avg_float(memory_items, "memory_score"),
        "avg_semantic_similarity": _avg_float(memory_items, "semantic_similarity"),
        "avg_tag_overlap_score": _avg_float(memory_items, "tag_overlap_score"),
    }


def _empty_memory_quality() -> dict[str, float]:
    return {
        "avg_score": 0.0,
        "avg_semantic_similarity": 0.0,
        "avg_tag_overlap_score": 0.0,
    }


def _avg_float(items: list[dict[str, Any]], key: str) -> float:
    values: list[float] = []
    for item in items:
        try:
            values.append(float(item.get(key, 0.0)))
        except (TypeError, ValueError):
            continue
    if not values:
        return 0.0
    return sum(values) / len(values)


def _protocol_action_map() -> dict[str, str]:
    return {
        "planner": "plan.create",
        "retriever": "memory.semantic_search",
        "executor": "tool.run_python",
        "summarizer": "summary.create",
        "planner_refine": "plan.refine",
        "retriever_refine": "evidence.refine",
    }


def _planner_decision(
    message: AMPMessage,
    task: str,
    capability_to_agent: dict[str, str],
) -> PlannerDecision:
    try:
        return PlannerDecision.model_validate(message.result.get("decision")).normalized(
            capability_to_agent=capability_to_agent
        )
    except Exception:
        return PlannerDecision.from_task(task, capability_to_agent=capability_to_agent)


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


def _planner_direct_answer(message: AMPMessage, decision: PlannerDecision) -> str | None:
    if decision.need_retrieval or decision.need_tool_execution:
        return None
    llm_plan = _non_empty_text(message.result.get("llm_plan"))
    if llm_plan is None:
        return None
    match = re.search(
        r"(?:^|\n)#{0,6}\s*(?:Final Answer|最终回答|最终答案)\s*:?\s*(.+)",
        llm_plan,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    answer = re.split(r"\n#{1,6}\s+\S+", match.group(1).strip(), maxsplit=1)[0].strip()
    return answer or None


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


def _compact_evidence_items(
    evidence: list[dict[str, Any]],
    *,
    snippet_limit: int,
) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in evidence:
        compact_item: dict[str, Any] = {
            "title": str(item.get("title", "evidence")),
            "snippet": _compact_text(str(item.get("snippet", "")), snippet_limit),
        }
        memory_id = item.get("memory_id")
        if isinstance(memory_id, str) and memory_id:
            compact_item["memory_id"] = memory_id
        if "memory_score" in item:
            compact_item["memory_score"] = item.get("memory_score")
        compacted.append(compact_item)
    return compacted


def _compact_code_result_payload(payload: dict[str, Any], *, text_limit: int) -> dict[str, Any]:
    codeact = payload.get("codeact")
    codeact_summary: dict[str, Any] = {}
    if isinstance(codeact, dict):
        code = str(codeact.get("code", ""))
        codeact_summary = {
            "code": code,
            "code_preview": _compact_text(code, text_limit),
            "code_chars": len(code),
            "generated_by_llm": bool(codeact.get("generated_by_llm")),
            "stdout": _compact_text(str(codeact.get("stdout", "")), text_limit),
            "stderr": _compact_text(str(codeact.get("stderr", "")), max(120, text_limit // 2)),
            "exit_code": codeact.get("exit_code"),
        }
    generated_files = payload.get("generated_files")
    compact_files: list[dict[str, Any]] = []
    if isinstance(generated_files, list):
        for item in generated_files:
            if not isinstance(item, dict):
                continue
            compact_item: dict[str, Any] = {
                "path": str(item.get("path", "")),
                "relative_path": str(item.get("relative_path", "")),
                "written": bool(item.get("written", False)),
                "overwritten": bool(item.get("overwritten", False)),
                "bytes": int(item.get("bytes", 0)),
            }
            if item.get("error"):
                compact_item["error"] = str(item["error"])
            compact_files.append(compact_item)
    executor_result = payload.get("executor_result")
    compact_executor_result: dict[str, Any] = {}
    if isinstance(executor_result, dict):
        compact_executor_result = {
            "validated": bool(executor_result.get("validated", False)),
            "llm_generated_code": bool(executor_result.get("llm_generated_code", False)),
            "state_refs_consumed": list(executor_result.get("state_refs_consumed", [])),
        }
    return {
        "stdout": _compact_text(str(payload.get("stdout", "")), text_limit),
        "stderr": _compact_text(str(payload.get("stderr", "")), max(120, text_limit // 2)),
        "exit_code": payload.get("exit_code"),
        "latency_ms": payload.get("latency_ms"),
        "backend": payload.get("backend", ""),
        "executor_result": compact_executor_result,
        "generated_files": compact_files,
        "tool_feedback": payload.get("tool_feedback", {}),
        "codeact": codeact_summary,
    }


def _evidence_digest(
    evidence: list[dict[str, Any]],
    *,
    snippet_limit: int = 160,
    max_items: int = 3,
) -> str:
    snippets: list[str] = []
    has_memory_reuse = False
    for item in evidence[:max_items]:
        title = str(item.get("title", "evidence"))
        snippet = _compact_text(str(item.get("snippet", "")), snippet_limit)
        reuse_hint = str(item.get("reuse_hint", ""))
        if reuse_hint:
            has_memory_reuse = True
            snippets.append(f"{title} [memory reuse]: {snippet}")
        else:
            snippets.append(f"{title}: {snippet}")
    if has_memory_reuse:
        snippets.insert(
            0,
            "[MEMORY_REUSE_ACTIVE] The evidence below came from similar prior tasks. "
            "Use it as a starting reference; adapt rather than recompute.",
        )
    return "\n".join(snippets)


def _compact_text(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 3)]}..."


def _code_result_summary(payload: dict[str, Any]) -> str:
    codeact = payload.get("codeact")
    if not isinstance(codeact, dict):
        return "CodeAct execution result unavailable."
    stdout = str(codeact.get("stdout", "")).strip()
    stderr = str(codeact.get("stderr", "")).strip()
    exit_code = codeact.get("exit_code")
    generated_files = payload.get("generated_files")
    file_text = ""
    if isinstance(generated_files, list) and generated_files:
        paths = [
            str(item.get("path"))
            for item in generated_files
            if isinstance(item, dict) and item.get("path")
        ]
        if paths:
            file_text = f"; generated_files={', '.join(paths)}"
    return (
        f"sandbox exit {exit_code}; "
        f"stdout={stdout[:500] or '<empty>'}; "
        f"stderr={stderr[:300] or '<empty>'}"
        f"{file_text}"
    )


def _protocol_agent_io_stats(
    *,
    paths: RuntimePaths,
    trace_id: str,
) -> tuple[int, int]:
    """Count tokens and bytes from actual agent I/O messages only.

    Agents now receive their input via structured params (not by reading all
    state refs), so we count only the JSON payloads exchanged between agents.
    State ref payload sizes are tracked separately via state_transfer_bytes.
    """
    tokens = 0
    byte_count = 0
    for row in read_jsonl(paths.protocol_agent_io):
        if str(row.get("trace_id", "")) != trace_id:
            continue
        input_payload = row.get("input")
        output_payload = row.get("output")
        tokens += estimate_tokens(_json_text_without_state_refs(input_payload))
        tokens += estimate_tokens(_json_text_without_state_refs(output_payload))
        byte_count += _json_size_without_state_refs(input_payload)
        byte_count += _json_size_without_state_refs(output_payload)
    return tokens, byte_count


def _json_text_without_state_refs(value: object) -> str:
    cleaned = _without_state_refs(value)
    if cleaned in ({}, [], None):
        return ""
    return orjson.dumps(cleaned).decode("utf-8")


def _json_size_without_state_refs(value: object) -> int:
    cleaned = _without_state_refs(value)
    if cleaned in ({}, [], None):
        return 0
    return len(orjson.dumps(cleaned))


def _without_state_refs(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_state_refs(item)
            for key, item in value.items()
            if key not in {"state_refs", "state_refs_in", "state_refs_out"}
        }
    if isinstance(value, list):
        return [_without_state_refs(item) for item in value]
    return value




def _write_generated_files(*, root: Path, files: object) -> list[dict[str, object]]:
    if not isinstance(files, list):
        return []
    root_resolved = root.resolve()
    written: list[dict[str, object]] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        raw_path = item.get("path")
        content = item.get("content")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        if not isinstance(content, str):
            continue
        relative_path = Path(raw_path)
        if relative_path.is_absolute() or any(part == ".." for part in relative_path.parts):
            written.append(
                {
                    "path": raw_path,
                    "written": False,
                    "error": "generated file path must stay inside workspace",
                }
            )
            continue
        target = (root_resolved / relative_path).resolve()
        try:
            target.relative_to(root_resolved)
        except ValueError:
            written.append(
                {
                    "path": raw_path,
                    "written": False,
                    "error": "generated file path escaped workspace",
                }
            )
            continue
        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(
            {
                "path": str(target),
                "relative_path": str(target.relative_to(root_resolved)),
                "written": True,
                "overwritten": existed,
                "bytes": len(content.encode("utf-8")),
            }
        )
    return written


def _fallback_summary(
    *,
    evidence_count: int,
    route: list[str],
    code_result_summary: str,
) -> str:
    route_text = " -> ".join(route)
    base = (
        "Protocol Mode answer: structured collaboration completed with "
        f"{evidence_count} evidence item(s) and route {route_text}."
    )
    if not code_result_summary:
        return base
    return f"{base}\n\nCodeAct result: {code_result_summary}"


def _refinement_feedback_parts(
    tool_feedback: dict[str, Any],
) -> list[ContextPart]:
    priorities = [
        "evidence_gaps",
        "failed_claims",
        "recommended_next_actions",
        "validated_claims",
        "execution_status",
        "stdout_summary",
        "stderr_summary",
    ]
    return [
        ContextPart(
            name=name,
            text=orjson.dumps(tool_feedback.get(name)).decode("utf-8"),
        )
        for name in priorities
        if tool_feedback.get(name) not in (None, "", [], {})
    ]
