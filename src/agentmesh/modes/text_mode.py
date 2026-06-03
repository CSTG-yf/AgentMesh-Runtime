import time
from pathlib import Path
from uuid import uuid4

from agentmesh.eval.metrics import ModeRunResult, RunMetrics, estimate_tokens
from agentmesh.eval.quality import deterministic_quality_score
from agentmesh.storage.jsonl import append_jsonl
from agentmesh.storage.paths import RuntimePaths


def run_text_mode(task_path: Path, paths: RuntimePaths) -> ModeRunResult:
    paths.ensure()
    trace_id = f"trace-{uuid4().hex[:12]}"
    task = task_path.read_text(encoding="utf-8")
    start = time.perf_counter()
    context = task
    agents = ["planner", "retriever", "executor", "summarizer"]
    for index, agent in enumerate(agents):
        target = agents[index + 1] if index + 1 < len(agents) else "runtime"
        append_jsonl(
            paths.text_messages,
            {
                "trace_id": trace_id,
                "source_agent": agent,
                "target_agent": target,
                "content": context,
            },
        )
        context = f"{context}\n[{agent}] processed task with full text context."
    answer = "Text Mode answer: " + context.splitlines()[-1]
    latency_ms = int((time.perf_counter() - start) * 1000)
    metrics = RunMetrics(
        message_count=len(agents),
        text_chars=sum(len(item) for item in [task, context]),
        estimated_tokens=estimate_tokens(task) + estimate_tokens(context),
        latency_ms=latency_ms,
        answer_quality_score=deterministic_quality_score(answer),
    )
    append_jsonl(paths.text_trace, {"trace_id": trace_id, "metrics": metrics.model_dump()})
    return ModeRunResult(mode="text", trace_id=trace_id, answer=answer, metrics=metrics)
