from pathlib import Path

from agentmesh.eval.compare import run_prompt_compare
from agentmesh.eval.metrics import ModeRunResult, RunMetrics
from agentmesh.storage.paths import RuntimePaths


def test_prompt_compare_runs_both_modes_from_user_input(tmp_path) -> None:
    summary = run_prompt_compare(
        "Analyze a long-context multi-agent runtime and compare communication overhead.",
        paths=RuntimePaths(root=tmp_path),
    )

    assert summary.text.mode == "text"
    assert summary.protocol.mode == "protocol"
    assert summary.text.metrics.estimated_tokens > summary.protocol.metrics.estimated_tokens
    assert summary.text.metrics.wire_bytes > 0
    assert summary.protocol.metrics.wire_bytes > 0
    assert summary.task_path.endswith(".txt")


def test_prompt_compare_forwards_llm_flag_to_both_modes(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_text_mode(
        task_path: Path,
        paths: RuntimePaths,
        llm_client=None,
        load_configured_llm: bool = True,
    ) -> ModeRunResult:
        del task_path, paths, llm_client
        calls.append(("text", load_configured_llm))
        return ModeRunResult(
            mode="text",
            trace_id="trace-text",
            answer="text",
            metrics=RunMetrics(estimated_tokens=10, wire_bytes=100),
        )

    def fake_protocol_mode(
        task_path: Path,
        paths: RuntimePaths,
        llm_client=None,
        load_configured_llm: bool = True,
    ) -> ModeRunResult:
        del task_path, paths, llm_client
        calls.append(("protocol", load_configured_llm))
        return ModeRunResult(
            mode="protocol",
            trace_id="trace-protocol",
            answer="protocol",
            metrics=RunMetrics(estimated_tokens=5, wire_bytes=50),
        )

    monkeypatch.setattr("agentmesh.eval.compare.run_text_mode", fake_text_mode)
    monkeypatch.setattr("agentmesh.eval.compare.run_protocol_mode", fake_protocol_mode)

    run_prompt_compare("hello", RuntimePaths(root=tmp_path), use_llm=True)

    assert calls == [("text", True), ("protocol", True)]
