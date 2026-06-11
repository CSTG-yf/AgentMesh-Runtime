from pathlib import Path

from agentmesh.eval.compare import CompareProgressEvent, run_prompt_compare
from agentmesh.eval.metrics import ModeRunResult, RunMetrics
from agentmesh.storage.agent_io import append_protocol_agent_io, append_text_agent_io
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


def test_prompt_compare_uses_llm_by_default_and_can_disable_it(
    tmp_path,
    monkeypatch,
) -> None:
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

    run_prompt_compare("hello", RuntimePaths(root=tmp_path))
    run_prompt_compare("hello", RuntimePaths(root=tmp_path), use_llm=False)

    assert calls == [
        ("text", True),
        ("protocol", True),
        ("text", False),
        ("protocol", False),
    ]


def test_prompt_compare_streams_and_collects_agent_outputs(
    tmp_path,
    monkeypatch,
) -> None:
    events: list[CompareProgressEvent] = []

    def fake_text_mode(
        task_path: Path,
        paths: RuntimePaths,
        llm_client=None,
        load_configured_llm: bool = True,
    ) -> ModeRunResult:
        del task_path, llm_client, load_configured_llm
        append_text_agent_io(
            paths=paths,
            trace_id="trace-text",
            step=1,
            agent="planner",
            target_agent="retriever",
            input_content="task",
            output_content="text planner output",
        )
        return ModeRunResult(
            mode="text",
            trace_id="trace-text",
            answer="text final answer",
            metrics=RunMetrics(estimated_tokens=10, wire_bytes=100),
        )

    def fake_protocol_mode(
        task_path: Path,
        paths: RuntimePaths,
        llm_client=None,
        load_configured_llm: bool = True,
    ) -> ModeRunResult:
        del task_path, llm_client, load_configured_llm
        append_protocol_agent_io(
            paths=paths,
            trace_id="trace-protocol",
            step=1,
            source_agent="runtime",
            agent="planner",
            action="plan.create",
            params={"task": "task"},
            result={"plan": ["protocol planner output"]},
            state_refs_in=["state://task"],
            state_refs_out=["state://plan"],
            result_msg_type="RESULT",
        )
        return ModeRunResult(
            mode="protocol",
            trace_id="trace-protocol",
            answer="protocol final answer",
            metrics=RunMetrics(estimated_tokens=5, wire_bytes=50),
        )

    monkeypatch.setattr("agentmesh.eval.compare.run_text_mode", fake_text_mode)
    monkeypatch.setattr("agentmesh.eval.compare.run_protocol_mode", fake_protocol_mode)

    summary = run_prompt_compare(
        "hello",
        RuntimePaths(root=tmp_path),
        progress_callback=events.append,
    )

    assert [item.output for item in summary.text_agent_outputs] == ["text planner output"]
    assert "protocol planner output" in summary.protocol_agent_outputs[0].output
    assert any(event.kind == "mode_start" and event.mode == "text" for event in events)
    assert any(event.kind == "mode_start" and event.mode == "protocol" for event in events)
    assert any(
        event.kind == "agent_output"
        and event.agent_output is not None
        and event.agent_output.agent == "planner"
        for event in events
    )
