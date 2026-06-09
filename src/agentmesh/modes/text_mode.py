import time
from pathlib import Path
from typing import cast
from uuid import uuid4

from agentmesh.eval.metrics import ModeRunResult, RunMetrics, estimate_tokens
from agentmesh.eval.quality import deterministic_quality_score
from agentmesh.llm.client import ChatMessage, LLMClient, create_llm_client
from agentmesh.runtime.registry import RuntimeContext
from agentmesh.storage.agent_io import append_text_agent_io
from agentmesh.storage.jsonl import append_jsonl
from agentmesh.storage.paths import RuntimePaths


def run_text_mode(
    task_path: Path,
    paths: RuntimePaths,
    llm_client: LLMClient | None = None,
    load_configured_llm: bool = True,
) -> ModeRunResult:
    paths.ensure()
    trace_id = f"trace-{uuid4().hex[:12]}"
    task = task_path.read_text(encoding="utf-8")
    runtime_context = RuntimeContext.from_paths(
        paths=paths,
        task_path=task_path,
        trace_id=trace_id,
        llm_client=llm_client,
        load_configured_llm=load_configured_llm,
    )
    if llm_client is None and load_configured_llm:
        runtime_context = runtime_context.model_copy(
            update={"llm_client": _create_text_mode_llm_client(runtime_context)}
        )
    start = time.perf_counter()
    context = task
    final_response = ""
    text_wire_bytes = 0
    agents = ["planner", "retriever", "executor", "summarizer"]
    for index, agent in enumerate(agents):
        target = agents[index + 1] if index + 1 < len(agents) else "runtime"
        text_wire_bytes += len(context.encode("utf-8"))
        append_jsonl(
            paths.text_messages,
            {
                "trace_id": trace_id,
                "source_agent": agent,
                "target_agent": target,
                "content": context,
            },
        )
        response = _text_agent_response(
            agent=agent,
            full_context=context,
            runtime_context=runtime_context,
        )
        append_text_agent_io(
            paths=paths,
            trace_id=trace_id,
            step=index + 1,
            agent=agent,
            target_agent=target,
            input_content=context,
            output_content=response,
        )
        final_response = response
        context = f"{context}\n[{agent}] {response}"
    answer = final_response.strip()
    if not answer:
        answer = "Text Mode answer: " + context.splitlines()[-1]
    latency_ms = int((time.perf_counter() - start) * 1000)
    metrics = RunMetrics(
        message_count=len(agents),
        text_chars=sum(len(item) for item in [task, context]),
        estimated_tokens=estimate_tokens(task) + estimate_tokens(context),
        communication_model="plain_text",
        wire_bytes=text_wire_bytes,
        text_wire_bytes=text_wire_bytes,
        latency_ms=latency_ms,
        answer_quality_score=deterministic_quality_score(answer),
    )
    append_jsonl(paths.text_trace, {"trace_id": trace_id, "metrics": metrics.model_dump()})
    return ModeRunResult(mode="text", trace_id=trace_id, answer=answer, metrics=metrics)


def _text_agent_response(
    *,
    agent: str,
    full_context: str,
    runtime_context: RuntimeContext,
) -> str:
    if runtime_context.llm_client is None:
        return "processed task with full text context."
    try:
        response = runtime_context.llm_client.complete(
            agent_name=agent,
            messages=[
                ChatMessage(
                    role="system",
                    content=_text_mode_system_prompt(agent, runtime_context, full_context),
                ),
                ChatMessage(role="user", content=full_context),
            ],
            variables={"input": full_context},
        )
        return str(response)
    except Exception:
        return _fallback_text_agent_response(agent=agent, full_context=full_context)


def _create_text_mode_llm_client(runtime_context: RuntimeContext) -> LLMClient | None:
    text_timeout = runtime_context.config.llm.text_timeout_seconds
    if text_timeout is None:
        return cast(LLMClient | None, runtime_context.llm_client)
    config = runtime_context.config.model_copy(
        update={
            "llm": runtime_context.config.llm.model_copy(
                update={"timeout_seconds": text_timeout}
            )
        }
    )
    return create_llm_client(config)


def _fallback_text_agent_response(*, agent: str, full_context: str) -> str:
    if agent != "summarizer":
        return "processed task with full text context."
    latest_output = (
        _latest_agent_output(full_context, "executor")
        or _latest_agent_output(full_context, "retriever")
        or _latest_agent_output(full_context, "planner")
    )
    if not latest_output:
        return "processed task with full text context."
    return (
        "Text Mode answer: summarizer LLM unavailable; "
        "preserving latest upstream agent output.\n\n"
        f"{latest_output}"
    )


def _latest_agent_output(full_context: str, agent: str) -> str:
    marker = f"\n[{agent}] "
    start = full_context.rfind(marker)
    if start == -1:
        return ""
    start += len(marker)
    next_marker = full_context.find("\n[", start)
    if next_marker == -1:
        return full_context[start:].strip()
    return full_context[start:next_marker].strip()


def _text_mode_system_prompt(
    agent: str,
    runtime_context: RuntimeContext,
    full_context: str,
) -> str:
    rendered = runtime_context.prompts.render(
        agent,
        {
            "task": full_context,
            "query": full_context,
            "input": full_context,
        },
    )
    role_constraint = ""
    if agent == "executor":
        role_constraint = (
            " ExecutorAgent-specific rule: if the accumulated context asks for code, "
            "return only Python code or concise structured execution observations; do "
            "not write the final user-facing synthesis."
        )
    return (
        f"{rendered}\n\n"
        "Text Mode baseline rule: you receive the complete accumulated natural-language "
        "context from the previous agent. Do not use StateRefs, memory retrieval, "
        "embeddings, Rust helpers, structured protocol envelopes, sandbox tools, or "
        "hidden state. Respond only with natural language that can be appended to the "
        "full text context for the next agent."
        f"{role_constraint}"
    )
