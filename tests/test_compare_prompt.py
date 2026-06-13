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


def test_prompt_compare_summarizes_memory_metrics(
    tmp_path,
    monkeypatch,
) -> None:
    def fake_text_mode(
        task_path: Path,
        paths: RuntimePaths,
        llm_client=None,
        load_configured_llm: bool = True,
    ) -> ModeRunResult:
        del task_path, paths, llm_client, load_configured_llm
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
        del task_path, paths, llm_client, load_configured_llm
        return ModeRunResult(
            mode="protocol",
            trace_id="trace-protocol",
            answer="protocol",
            metrics=RunMetrics(
                estimated_tokens=5,
                wire_bytes=50,
                memory_query_count=2,
                memory_query_hit_count=1,
                memory_hit_count=3,
                memory_reused_unit_count=4,
                memory_avg_score=0.6,
                memory_avg_semantic_similarity=0.4,
                memory_avg_tag_overlap_score=0.25,
            ),
        )

    monkeypatch.setattr("agentmesh.eval.compare.run_text_mode", fake_text_mode)
    monkeypatch.setattr("agentmesh.eval.compare.run_protocol_mode", fake_protocol_mode)

    summary = run_prompt_compare("hello", RuntimePaths(root=tmp_path))

    assert summary.memory_hit_rate == 0.5
    assert summary.memory_query_count == 2
    assert summary.memory_query_hit_count == 1
    assert summary.memory_reused_unit_count == 4
    assert summary.memory_avg_reused_units_per_query == 2.0
    assert summary.memory_avg_score == 0.6
    assert summary.memory_avg_semantic_similarity == 0.4
    assert summary.memory_avg_tag_overlap_score == 0.25


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


def test_prompt_compare_progress_only_streams_current_trace(
    tmp_path,
    monkeypatch,
) -> None:
    events: list[CompareProgressEvent] = []

    def fake_text_mode(
        task_path: Path,
        paths: RuntimePaths,
        llm_client=None,
        load_configured_llm: bool = True,
        trace_id: str | None = None,
    ) -> ModeRunResult:
        del task_path, llm_client, load_configured_llm
        current_trace_id = trace_id or "trace-text-current"
        append_text_agent_io(
            paths=paths,
            trace_id="trace-text-foreign",
            step=1,
            agent="planner",
            target_agent="retriever",
            input_content="foreign task",
            output_content="foreign text output",
        )
        append_text_agent_io(
            paths=paths,
            trace_id=current_trace_id,
            step=1,
            agent="planner",
            target_agent="retriever",
            input_content="task",
            output_content="current text output",
        )
        return ModeRunResult(
            mode="text",
            trace_id=current_trace_id,
            answer="text final answer",
            metrics=RunMetrics(estimated_tokens=10, wire_bytes=100),
        )

    def fake_protocol_mode(
        task_path: Path,
        paths: RuntimePaths,
        llm_client=None,
        load_configured_llm: bool = True,
        trace_id: str | None = None,
    ) -> ModeRunResult:
        del task_path, llm_client, load_configured_llm
        current_trace_id = trace_id or "trace-protocol-current"
        append_protocol_agent_io(
            paths=paths,
            trace_id="trace-protocol-foreign",
            step=1,
            source_agent="runtime",
            agent="planner",
            action="plan.create",
            params={"task": "foreign task"},
            result={"plan": ["foreign protocol output"]},
            state_refs_in=["state://foreign-task"],
            state_refs_out=["state://foreign-plan"],
            result_msg_type="RESULT",
        )
        append_protocol_agent_io(
            paths=paths,
            trace_id=current_trace_id,
            step=1,
            source_agent="runtime",
            agent="planner",
            action="plan.create",
            params={"task": "task"},
            result={"plan": ["current protocol output"]},
            state_refs_in=["state://task"],
            state_refs_out=["state://plan"],
            result_msg_type="RESULT",
        )
        return ModeRunResult(
            mode="protocol",
            trace_id=current_trace_id,
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

    streamed_outputs = [
        event.agent_output.output
        for event in events
        if event.kind == "agent_output" and event.agent_output is not None
    ]
    assert streamed_outputs == [
        "current text output",
        '{\n  "plan": [\n    "current protocol output"\n  ]\n}',
    ]
    assert [item.output for item in summary.text_agent_outputs] == ["current text output"]
    assert "current protocol output" in summary.protocol_agent_outputs[0].output
