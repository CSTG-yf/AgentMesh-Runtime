import time
from pathlib import Path
from uuid import uuid4

from agentmesh.eval.metrics import ModeRunResult, RunMetrics, estimate_tokens
from agentmesh.eval.quality import deterministic_quality_score
from agentmesh.memory.policy import MemoryWritePolicy
from agentmesh.memory.schema import MemoryUnit
from agentmesh.memory.sqlite_store import SQLiteMemoryStore
from agentmesh.protocol.codec import encode_message
from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.schema import AMPMessage
from agentmesh.runtime.orchestrator import default_registry
from agentmesh.runtime.registry import RuntimeContext
from agentmesh.sandbox.runner import SandboxRunner
from agentmesh.state.embedding import HashEmbeddingEncoder
from agentmesh.state.store import StateStore
from agentmesh.storage.jsonl import append_jsonl
from agentmesh.storage.paths import RuntimePaths


def run_protocol_mode(task_path: Path, paths: RuntimePaths) -> ModeRunResult:
    paths.ensure()
    trace_id = f"trace-{uuid4().hex[:12]}"
    task = task_path.read_text(encoding="utf-8")
    start = time.perf_counter()
    encoder = HashEmbeddingEncoder()
    state_store = StateStore(paths)
    memory_store = SQLiteMemoryStore(paths=paths, state_store=state_store, encoder=encoder)
    registry = default_registry()
    context = RuntimeContext.from_paths(paths=paths, task_path=task_path, trace_id=trace_id)
    messages: list[AMPMessage] = []

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
        summary="; ".join(planner_result.result["plan"]),
        parent_state_refs=[task_ref, query_ref],
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

    memory_hits = memory_store.semantic_search(task)
    for hit in memory_hits:
        memory_store.increment_reuse(hit.memory_id)
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

    sandbox_result = SandboxRunner(paths.sandbox_dir).run_python("print('agentmesh validation ok')")
    executor_invoke = AMPMessage(
        trace_id=trace_id,
        source_agent="retriever",
        target_agent="executor",
        msg_type=MsgType.INVOKE,
        action="tool.run_python",
        state_refs=[evidence_ref],
    )
    messages.append(executor_invoke)
    executor_result = registry.get("executor").handle(executor_invoke, context)
    messages.append(executor_result)
    code_result_ref = state_store.put_code_result(
        trace_id=trace_id,
        producer="executor",
        result=sandbox_result.model_dump(),
        parent_state_refs=[evidence_ref],
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

    summary_text = (
        "Protocol Mode answer: structured collaboration completed with "
        f"{len(evidence)} evidence item(s) and sandbox exit {sandbox_result.exit_code}."
    )
    summarizer_invoke = AMPMessage(
        trace_id=trace_id,
        source_agent="executor",
        target_agent="summarizer",
        msg_type=MsgType.INVOKE,
        action="summary.create",
        state_refs=[task_ref, plan_ref, evidence_ref, code_result_ref],
    )
    messages.append(summarizer_invoke)
    messages.append(registry.get("summarizer").handle(summarizer_invoke, context))
    summary_ref = state_store.put_summary(
        trace_id=trace_id,
        producer="summarizer",
        summary=summary_text,
        parent_state_refs=[task_ref, plan_ref, evidence_ref, code_result_ref],
    )
    memory_embedding_ref = state_store.put_embedding(
        trace_id=trace_id,
        producer="summarizer",
        embedding=encoder.encode(summary_text),
        parent_state_refs=[summary_ref],
    )
    unit = MemoryUnit(
        source_agent="summarizer",
        task_topic=task[:80] or "untitled",
        summary=summary_text,
        tags=_derive_tags(task),
        evidence_refs=[evidence_ref],
        state_refs=[task_ref, plan_ref, code_result_ref, summary_ref],
        embedding_ref=memory_embedding_ref,
        confidence=0.8,
        validity_score=1.0 if sandbox_result.exit_code == 0 else 0.3,
        provenance_trace_id=trace_id,
    )
    if MemoryWritePolicy().should_write(unit):
        memory_store.put(unit)
        messages.append(
            AMPMessage(
                trace_id=trace_id,
                source_agent="summarizer",
                target_agent="runtime",
                msg_type=MsgType.MEMORY_PUT,
                action="memory.put",
                result={"memory_id": unit.memory_id},
                state_refs=unit.state_refs + unit.evidence_refs + [memory_embedding_ref],
            )
        )

    for message in messages:
        append_jsonl(paths.protocol_messages, message.model_dump(mode="json"))
    protocol_bytes = sum(len(encode_message(message)) for message in messages)
    state_records = state_store.list_by_trace(trace_id)
    latency_ms = int((time.perf_counter() - start) * 1000)
    metrics = RunMetrics(
        message_count=len(messages),
        text_chars=len(task),
        estimated_tokens=estimate_tokens(task),
        protocol_bytes=protocol_bytes,
        state_transfer_count=len(state_records),
        state_transfer_bytes=sum(record.size_bytes for record in state_records),
        memory_query_count=1,
        memory_hit_count=1 if memory_hits else 0,
        latency_ms=latency_ms,
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
