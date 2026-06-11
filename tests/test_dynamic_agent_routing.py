from pathlib import Path

import pytest

from agentmesh.errors import ProtocolError
from agentmesh.modes.protocol_mode import _tool_feedback, run_protocol_mode
from agentmesh.runtime.decision import PlannerDecision
from agentmesh.runtime.orchestrator import default_registry
from agentmesh.runtime.registry import RuntimeContext
from agentmesh.runtime.scheduler import ProtocolScheduler
from agentmesh.state.schema import StateType
from agentmesh.state.store import StateStore
from agentmesh.storage.jsonl import read_jsonl
from agentmesh.storage.paths import RuntimePaths


def test_planner_decision_classifies_analysis_without_tool_execution() -> None:
    decision = PlannerDecision.from_task("分析多 Agent 协作方式并给出架构改进方案")

    assert decision.intent == "analysis"
    assert decision.need_retrieval
    assert not decision.need_tool_execution
    assert decision.execution_route == ["planner", "retriever", "summarizer"]


def test_planner_decision_classifies_benchmark_with_tool_execution() -> None:
    decision = PlannerDecision.from_task("运行一次 benchmark，计算结构化协议的通信开销")

    assert decision.intent == "validation"
    assert decision.need_retrieval
    assert decision.need_tool_execution
    assert "tool.run_python" in decision.required_capabilities
    assert decision.execution_route == ["planner", "retriever", "executor", "summarizer"]


def test_planner_decision_classifies_chinese_sort_request_with_tool_execution() -> None:
    decision = PlannerDecision.from_task(
        "[1,3,11,231,4,55,66,1231,1,2] 写一个快速排序并输出排序后的结果"
    )

    assert decision.intent == "validation"
    assert decision.need_tool_execution
    assert "tool.run_python" in decision.required_capabilities
    assert decision.execution_route == ["planner", "retriever", "executor", "summarizer"]


def test_planner_decision_keeps_tool_execution_when_llm_json_under_routes() -> None:
    decision = PlannerDecision.from_llm_or_task(
        "[1,3,11,231,4,55,66,1231,1,2] 写一个快速排序并输出排序后的结果",
        """
        {
          "intent": "analysis",
          "required_capabilities": ["summary.create"],
          "execution_route": ["planner", "summarizer"],
          "need_retrieval": false,
          "need_tool_execution": false,
          "need_summary": true
        }
        """,
    )

    assert decision.need_tool_execution
    assert "executor" in decision.execution_route
    assert "tool.run_python" in decision.required_capabilities


def test_planner_decision_normalizes_route_from_capability_advertisements() -> None:
    capability_to_agent = {
        "plan.create": "planner",
        "memory.semantic_search": "retriever",
        "tool.run_python": "sandboxer",
        "summary.create": "summarizer",
    }

    decision = PlannerDecision.from_task(
        "run quicksort code",
        capability_to_agent=capability_to_agent,
    )

    assert decision.required_capabilities == [
        "plan.create",
        "memory.semantic_search",
        "tool.run_python",
        "summary.create",
    ]
    assert decision.execution_route == [
        "planner",
        "retriever",
        "sandboxer",
        "summarizer",
    ]


def test_planner_decision_normalizes_inconsistent_llm_json() -> None:
    decision = PlannerDecision.from_llm_or_task(
        "validate protocol benchmark",
        """
        {
          "intent": "analysis",
          "required_capabilities": ["tool.run_python"],
          "execution_route": ["planner", "summarizer"],
          "need_retrieval": false,
          "need_tool_execution": false,
          "need_summary": true
        }
        """,
    )

    assert decision.intent == "validation"
    assert decision.need_tool_execution
    assert decision.execution_route == ["planner", "executor", "summarizer"]
    assert "tool.run_python" in decision.required_capabilities


def test_protocol_scheduler_routes_by_capability(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    context = RuntimeContext.from_paths(paths=paths, trace_id="trace-scheduler")
    scheduler = ProtocolScheduler(
        registry=default_registry(),
        context=context,
        messages=[],
        protocol_map={"summarizer": "summary.create"},
    )

    result = scheduler.invoke(
        source_agent="runtime",
        action="summary.create",
        params={"code_result": "none"},
        state_refs=[],
    )

    assert result.source_agent == "summarizer"
    assert scheduler.selected_agents == ["summarizer"]


def test_protocol_scheduler_rejects_actions_outside_protocol_map(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    context = RuntimeContext.from_paths(paths=paths, trace_id="trace-scheduler")
    scheduler = ProtocolScheduler(
        registry=default_registry(),
        context=context,
        messages=[],
        protocol_map={"summarizer": "summary.create"},
    )

    with pytest.raises(ProtocolError):
        scheduler.invoke(source_agent="runtime", action="tool.run_python")


def test_protocol_mode_skips_executor_for_analysis_task(tmp_path: Path) -> None:
    task = tmp_path / "analysis.txt"
    task.write_text("分析多 Agent 协作方式并给出架构改进方案", encoding="utf-8")
    paths = RuntimePaths(root=tmp_path)

    result = run_protocol_mode(task, paths)

    assert result.metrics.dynamic_route == ["planner", "retriever", "summarizer"]
    assert "executor" in result.metrics.skipped_agents
    assert "executor" not in result.metrics.selected_agents
    assert "executor" not in result.metrics.stage_latency_ms
    assert "code_result_state" not in result.metrics.stage_latency_ms
    messages = read_jsonl(paths.protocol_messages)
    planner_state_targets = [
        item.get("target_agent")
        for item in messages
        if item.get("source_agent") == "planner" and item.get("action") == "state.put_summary"
    ]
    assert planner_state_targets == ["retriever"]


def test_protocol_mode_routes_summary_only_plan_state_to_summarizer(tmp_path: Path) -> None:
    task = tmp_path / "summary_only.txt"
    task.write_text("summary only: summarize existing implementation notes", encoding="utf-8")
    paths = RuntimePaths(root=tmp_path)

    result = run_protocol_mode(task, paths)

    assert result.metrics.dynamic_route == ["planner", "summarizer"]
    assert "retriever" in result.metrics.skipped_agents
    messages = read_jsonl(paths.protocol_messages)
    planner_state_targets = [
        item.get("target_agent")
        for item in messages
        if item.get("source_agent") == "planner" and item.get("action") == "state.put_summary"
    ]
    assert planner_state_targets == ["summarizer"]


def test_protocol_mode_invokes_executor_and_feedback_for_benchmark_task(
    tmp_path: Path,
) -> None:
    task = tmp_path / "benchmark.txt"
    task.write_text("运行一次 benchmark，计算结构化协议的通信开销并验证结果", encoding="utf-8")
    paths = RuntimePaths(root=tmp_path)

    result = run_protocol_mode(task, paths)

    assert result.metrics.dynamic_route == [
        "planner",
        "retriever",
        "executor",
        "planner",
        "retriever",
        "summarizer",
    ]
    assert "executor" in result.metrics.selected_agents
    assert result.metrics.feedback_round_count == 1
    assert result.metrics.planner_refine_count == 1
    assert result.metrics.retriever_refine_count == 1
    assert result.metrics.tool_feedback_count == 1
    assert "code_result_state" in result.metrics.stage_latency_ms
    messages = read_jsonl(paths.protocol_messages)
    assert any(item.get("action") == "plan.refine" for item in messages)
    assert any(item.get("action") == "evidence.refine" for item in messages)


def test_protocol_mode_executes_chinese_quicksort_request(tmp_path: Path) -> None:
    task = tmp_path / "sort.txt"
    task.write_text(
        "[1,3,11,231,4,55,66,1231,1,2] 写一个快速排序并输出排序后的结果",
        encoding="utf-8",
    )
    paths = RuntimePaths(root=tmp_path)

    result = run_protocol_mode(task, paths)

    assert "executor" in result.metrics.selected_agents
    assert "sandbox exit 0" in result.answer
    assert "[1, 1, 2, 3, 4, 11, 55, 66, 231, 1231]" in result.answer
    state_store = StateStore(paths)
    executor_payloads = [
        state_store.get(record.ref)[1]
        for record in state_store.list_by_trace(result.trace_id)
        if record.producer == "executor" and record.state_type == StateType.CODE_RESULT
    ]
    assert executor_payloads
    assert executor_payloads[0]["exit_code"] == 0
    assert "[1, 1, 2, 3, 4, 11, 55, 66, 231, 1231]" in executor_payloads[0]["stdout"]


def test_protocol_mode_executes_chinese_quick_sort_abbreviation_with_self_test(
    tmp_path: Path,
) -> None:
    task = tmp_path / "quick_sort_abbreviation.txt"
    task.write_text(
        "实现一个快排，并且自建用例 输入输出和快排函数要在结果中显现",
        encoding="utf-8",
    )
    paths = RuntimePaths(root=tmp_path)

    result = run_protocol_mode(task, paths)

    assert "executor" in result.metrics.selected_agents
    state_store = StateStore(paths)
    executor_payloads = [
        state_store.get(record.ref)[1]
        for record in state_store.list_by_trace(result.trace_id)
        if record.producer == "executor" and record.state_type == StateType.CODE_RESULT
    ]
    assert executor_payloads
    stdout = executor_payloads[0]["stdout"]
    assert '"algorithm": "quick_sort"' in stdout
    assert '"input":' in stdout
    assert '"sorted":' in stdout


def test_tool_feedback_parses_structured_executor_stdout() -> None:
    feedback = _tool_feedback(
        "validate benchmark",
        {
            "stdout": (
                '{"evidence_gaps":["missing baseline"],'
                '"validated_claims":["metric computed"],'
                '"failed_claims":["latency missing"],'
                '"recommended_next_actions":['
                '{"target_capability":"memory.semantic_search","reason":"need history"}]}'
            ),
            "stderr": "",
            "exit_code": 0,
        },
    )

    assert "missing baseline" in feedback["evidence_gaps"]
    assert "metric computed" in feedback["validated_claims"]
    assert "latency missing" in feedback["failed_claims"]
    assert any(
        item["target_capability"] == "memory.semantic_search"
        for item in feedback["recommended_next_actions"]
    )
