from uuid import uuid4

from pydantic import BaseModel

from agentmesh.eval.benchmark import _rate
from agentmesh.eval.metrics import ModeRunResult
from agentmesh.modes.protocol_mode import run_protocol_mode
from agentmesh.modes.text_mode import run_text_mode
from agentmesh.storage.paths import RuntimePaths


class CompareSummary(BaseModel):
    task_path: str
    text: ModeRunResult
    protocol: ModeRunResult
    token_saving_rate: float
    latency_reduction_rate: float
    wire_bytes_reduction_rate: float
    memory_hit_rate: float


def run_prompt_compare(
    prompt: str,
    paths: RuntimePaths,
    *,
    use_llm: bool = False,
) -> CompareSummary:
    paths.ensure()
    task_dir = paths.latest_run / "user_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    task_path = task_dir / f"prompt-{uuid4().hex[:8]}.txt"
    task_path.write_text(prompt, encoding="utf-8")

    text_result = run_text_mode(task_path=task_path, paths=paths)
    protocol_result = run_protocol_mode(
        task_path=task_path,
        paths=paths,
        load_configured_llm=use_llm,
    )
    return CompareSummary(
        task_path=str(task_path),
        text=text_result,
        protocol=protocol_result,
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
    )
