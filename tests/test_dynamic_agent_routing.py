from pathlib import Path

import pytest

from agentmesh.errors import ProtocolError
from agentmesh.eval.llm_experiment import LLMExperimentProfile
from agentmesh.llm.client import ChatMessage, LLMClient
from agentmesh.modes.protocol_mode import (
    _compact_code_result_payload,
    _tool_feedback,
    run_protocol_mode,
)
from agentmesh.runtime.decision import PlannerDecision
from agentmesh.runtime.orchestrator import default_registry
from agentmesh.runtime.registry import RuntimeContext
from agentmesh.runtime.scheduler import ProtocolScheduler
from agentmesh.state.schema import StateType
from agentmesh.state.store import StateStore
from agentmesh.storage.jsonl import read_jsonl
from agentmesh.storage.paths import RuntimePaths


def test_code_result_compaction_preserves_reconstructable_contract() -> None:
    payload = {
        "stdout": '{"status": "validated"}',
        "stderr": "",
        "exit_code": 0,
        "latency_ms": 3,
        "backend": "rust",
        "executor_result": {
            "validated": True,
            "llm_generated_code": True,
            "codeact_code": "print('ok')",
        },
        "generated_files": [
            {
                "path": "C:/workspace/generated_code.py",
                "relative_path": "generated_code.py",
                "written": True,
                "overwritten": False,
                "bytes": 11,
            }
        ],
        "tool_feedback": {"status": "success"},
        "codeact": {
            "code": "print('ok')",
            "generated_by_llm": True,
            "stdout": "ok",
            "stderr": "",
            "exit_code": 0,
        },
    }

    compacted = _compact_code_result_payload(payload, text_limit=800)

    assert compacted["codeact"]["code"] == "print('ok')"
    assert compacted["executor_result"]["llm_generated_code"] is True
    assert compacted["generated_files"][0]["written"] is True
    assert compacted["generated_files"][0]["relative_path"] == "generated_code.py"


def test_planner_decision_classifies_analysis_without_tool_execution() -> None:
    decision = PlannerDecision.from_task("Analyze multi-agent collaboration architecture.")

    assert decision.intent == "analysis"
    assert decision.need_retrieval
    assert not decision.need_tool_execution
    assert decision.execution_route == ["planner", "retriever", "summarizer"]


def test_planner_decision_routes_identity_question_without_retrieval() -> None:
    decision = PlannerDecision.from_task("Who are you?")

    assert decision.intent == "conversation"
    assert not decision.need_retrieval
    assert not decision.need_tool_execution
    assert decision.required_capabilities == ["plan.create", "summary.create"]
    assert decision.execution_route == ["planner", "summarizer"]


@pytest.mark.parametrize(
    ("task", "intent", "task_type", "need_retrieval", "need_tool", "route"),
    [
        (
            "Please fix this Python error.",
            "validation",
            "code_execution",
            True,
            True,
            ["planner", "retriever", "executor", "summarizer"],
        ),
        (
            "Remember that I prefer concise answers.",
            "memory",
            "memory_management",
            True,
            False,
            ["planner", "retriever", "summarizer"],
        ),
        (
            "Find the project configuration docs.",
            "retrieval",
            "knowledge_lookup",
            True,
            False,
            ["planner", "retriever", "summarizer"],
        ),
        (
            "Review this PR for security issues.",
            "review",
            "code_review",
            True,
            False,
            ["planner", "retriever", "summarizer"],
        ),
        (
            "summary only: summarize existing implementation notes",
            "summarization",
            "summary_only",
            False,
            False,
            ["planner", "summarizer"],
        ),
        (
            "Calculate 21 * 2.",
            "validation",
            "calculation",
            False,
            True,
            ["planner", "executor", "summarizer"],
        ),
    ],
)
def test_planner_decision_covers_common_business_intents(
    task: str,
    intent: str,
    task_type: str,
    need_retrieval: bool,
    need_tool: bool,
    route: list[str],
) -> None:
    decision = PlannerDecision.from_task(task)

    assert decision.intent == intent
    assert decision.task_type == task_type
    assert decision.need_retrieval is need_retrieval
    assert decision.need_tool_execution is need_tool
    assert decision.execution_route == route


def test_planner_decision_classifies_benchmark_with_tool_execution() -> None:
    decision = PlannerDecision.from_task("Run a benchmark and validate protocol overhead.")

    assert decision.intent == "validation"
    assert decision.need_retrieval
    assert decision.need_tool_execution
    assert "tool.run_python" in decision.required_capabilities
    assert decision.execution_route == ["planner", "retriever", "executor", "summarizer"]


def test_planner_decision_classifies_sort_request_with_tool_execution() -> None:
    decision = PlannerDecision.from_task("Sort [3, 1, 2] and print the result.")

    assert decision.intent == "validation"
    assert decision.need_tool_execution
    assert "tool.run_python" in decision.required_capabilities
    assert decision.execution_route == ["planner", "retriever", "executor", "summarizer"]


def test_planner_decision_keeps_tool_execution_when_llm_json_under_routes() -> None:
    decision = PlannerDecision.from_llm_or_task(
        "Sort [3, 1, 2] and print the result.",
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


def test_quality_safe_policy_skips_retrieval_for_explanation() -> None:
    decision = PlannerDecision.from_task(
        "Analyze and explain quicksort."
    ).for_policy("quality_safe_v1")

    assert decision.execution_route == ["planner", "summarizer"]
    assert not decision.need_retrieval
    assert "quality_safe_v1: retrieval not required" in decision.reason


def test_quality_safe_policy_keeps_executor_and_memory_routes() -> None:
    calculation = PlannerDecision.from_task("Calculate 2 + 2").for_policy(
        "quality_safe_v1"
    )
    memory = PlannerDecision.from_task("Remember my preference").for_policy(
        "quality_safe_v1"
    )

    assert calculation.execution_route == ["planner", "executor", "summarizer"]
    assert memory.need_retrieval
    assert "retriever" in memory.execution_route


def test_legacy_policy_is_unchanged() -> None:
    decision = PlannerDecision.from_task("Analyze this report")

    assert decision.for_policy("legacy") == decision


@pytest.mark.parametrize(
    "task",
    [
        "Compare linear search vs binary search.",
        "Explain when to use a binary search algorithm.",
        "Analyze the time complexity of linear search.",
        "分析二分查找和线性查找的复杂度。",
    ],
)
def test_quality_safe_v2_skips_retrieval_for_algorithmic_search(task: str) -> None:
    decision = PlannerDecision.from_task(task).for_policy(
        "quality_safe_v2",
        task=task,
    )

    assert decision.task_type == "analysis_or_report"
    assert decision.execution_route == ["planner", "summarizer"]
    assert not decision.need_retrieval
    assert "quality_safe_v2: algorithmic search analysis" in decision.reason


def test_quality_safe_v2_implements_search_algorithm_without_retrieval() -> None:
    task = "Implement a binary search algorithm."
    decision = PlannerDecision.from_llm_or_task(
        task,
        """
        {
          "intent": "analysis",
          "task_type": "analysis_or_report",
          "required_capabilities": ["summary.create"],
          "execution_route": ["planner", "summarizer"],
          "need_retrieval": false,
          "need_tool_execution": false,
          "need_summary": true,
          "reason": "untrusted model classification"
        }
        """,
    ).for_policy("quality_safe_v2", task=task)

    assert decision.need_tool_execution
    assert "executor" in decision.execution_route
    assert not decision.need_retrieval
    assert "retriever" not in decision.execution_route


@pytest.mark.parametrize(
    "task",
    [
        "Search memory for my notes about binary search.",
        "Find the latest docs for the binary search API.",
        "Search this repository for implementations of linear search.",
        "Find facts and sources about search algorithms.",
    ],
)
def test_quality_safe_v2_keeps_retrieval_for_explicit_external_search(
    task: str,
) -> None:
    decision = PlannerDecision.from_llm_or_task(
        task,
        """
        {
          "intent": "analysis",
          "task_type": "analysis_or_report",
          "required_capabilities": ["summary.create"],
          "execution_route": ["planner", "summarizer"],
          "need_retrieval": false,
          "need_tool_execution": false,
          "need_summary": true,
          "reason": "untrusted model classification"
        }
        """,
    ).for_policy("quality_safe_v2", task=task)

    assert decision.need_retrieval
    assert "retriever" in decision.execution_route
    assert "memory.semantic_search" in decision.required_capabilities


@pytest.mark.parametrize(
    "task",
    [
        "Search Confluence for binary search guidance.",
        "Find binary search facts in the engineering knowledge base.",
        "Research binary search using the company handbook.",
        "Look up binary search performance in the internal records.",
        "Consult the team wiki about linear search complexity.",
        "Use the architecture archive to compare linear and binary search.",
        "查阅团队知识库，分析二分查找的性能。",
    ],
)
def test_quality_safe_v2_unknown_sources_cannot_bypass_retrieval_floor(
    task: str,
) -> None:
    decision = PlannerDecision.from_llm_or_task(
        task,
        """
        {
          "intent": "analysis",
          "task_type": "analysis_or_report",
          "required_capabilities": ["summary.create"],
          "execution_route": ["planner", "summarizer"],
          "need_retrieval": false,
          "need_tool_execution": false,
          "need_summary": true,
          "reason": "untrusted model classification"
        }
        """,
    ).for_policy("quality_safe_v2", task=task)

    assert decision.need_retrieval
    assert "retriever" in decision.execution_route


def test_quality_safe_v1_keeps_algorithmic_search_legacy_behavior() -> None:
    task = "Compare linear search vs binary search."
    decision = PlannerDecision.from_task(task).for_policy(
        "quality_safe_v1",
        task=task,
    )

    assert decision.need_retrieval
    assert "retriever" in decision.execution_route


@pytest.mark.parametrize(
    "task",
    [
        "Remember my preferred editor for future sessions.",
        "Find the latest documentation for this API.",
        "Perform a security code review of this repository.",
    ],
)
def test_quality_safe_policy_does_not_trust_llm_analysis_misclassification(
    task: str,
) -> None:
    decision = PlannerDecision.from_llm_or_task(
        task,
        """
        {
          "intent": "analysis",
          "task_type": "analysis_or_report",
          "required_capabilities": ["summary.create"],
          "execution_route": ["planner", "summarizer"],
          "need_retrieval": false,
          "need_tool_execution": false,
          "need_summary": true,
          "reason": "untrusted model classification"
        }
        """,
    ).for_policy("quality_safe_v1", task=task)

    assert decision.need_retrieval
    assert "retriever" in decision.execution_route
    assert "memory.semantic_search" in decision.required_capabilities


def test_candidate_quality_safe_route_is_persisted_in_metrics(tmp_path: Path) -> None:
    task = tmp_path / "task.txt"
    task.write_text("Analyze and explain quicksort.", encoding="utf-8")

    result = run_protocol_mode(
        task,
        RuntimePaths(root=tmp_path / "candidate"),
        load_configured_llm=False,
        experiment_profile=LLMExperimentProfile(
            profile_id="candidate",
            artifact_label="candidate",
            route_policy_version="quality_safe_v1",
            optimization_enabled=True,
        ),
    )

    assert result.metrics.dynamic_route == ["planner", "summarizer"]
    assert result.metrics.route_policy_version == "quality_safe_v1"
    assert "retrieval not required" in result.metrics.route_reason


def test_quality_safe_v2_protocol_skips_algorithm_search_retriever_with_fake_llm(
    tmp_path: Path,
) -> None:
    class AnalysisLLM(LLMClient):
        def __init__(self) -> None:
            self.agents: list[str] = []

        def complete(
            self,
            *,
            agent_name: str,
            messages: list[ChatMessage],
            variables: dict[str, object] | None = None,
        ) -> str:
            del messages, variables
            self.agents.append(agent_name)
            if agent_name == "planner":
                return """
                {
                  "intent": "analysis",
                  "task_type": "analysis_or_report",
                  "required_capabilities": ["summary.create"],
                  "execution_route": ["planner", "summarizer"],
                  "need_retrieval": false,
                  "need_tool_execution": false,
                  "need_summary": true,
                  "reason": "algorithm comparison"
                }
                """
            return "concise answer"

    task = tmp_path / "task.txt"
    task.write_text("Compare linear search vs binary search.", encoding="utf-8")
    client = AnalysisLLM()
    result = run_protocol_mode(
        task,
        RuntimePaths(root=tmp_path / "candidate-v2"),
        llm_client=client,
        experiment_profile=LLMExperimentProfile(
            profile_id="candidate-v2",
            artifact_label="candidate-v2",
            route_policy_version="quality_safe_v2",
            optimization_enabled=True,
        ),
    )

    assert "retriever" not in client.agents
    assert result.metrics.dynamic_route == ["planner", "summarizer"]
    assert result.metrics.route_policy_version == "quality_safe_v2"


def test_planner_decision_normalizes_route_from_capability_advertisements() -> None:
    capability_to_agent = {
        "plan.create": "planner",
        "memory.semantic_search": "retriever",
        "tool.run_python": "sandboxer",
        "summary.create": "summarizer",
    }

    decision = PlannerDecision.from_task(
        "run python code",
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
    task.write_text("Analyze multi-agent collaboration architecture.", encoding="utf-8")
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


class PlannerFinalAnswerLLM(LLMClient):
    def complete(
        self,
        *,
        agent_name: str,
        messages: list[ChatMessage],
        variables: dict[str, object] | None = None,
    ) -> str:
        _ = messages, variables
        if agent_name == "planner":
            return (
                "### AgentMesh Protocol Workflow\n"
                "- identify user intent\n\n"
                "### Final Answer\n"
                "Hello, I am AgentMesh Runtime's interactive assistant."
            )
        if agent_name == "summarizer":
            raise RuntimeError("summarizer unavailable")
        return "memory tagger output"


def test_protocol_mode_uses_planner_final_answer_for_direct_question(
    tmp_path: Path,
) -> None:
    task = tmp_path / "identity.txt"
    task.write_text("Who are you?", encoding="utf-8")
    paths = RuntimePaths(root=tmp_path)

    result = run_protocol_mode(task, paths, llm_client=PlannerFinalAnswerLLM())

    assert result.answer == "Hello, I am AgentMesh Runtime's interactive assistant."
    assert result.metrics.dynamic_route == ["planner", "summarizer"]
    assert "retriever" in result.metrics.skipped_agents


def test_protocol_mode_invokes_executor_and_feedback_for_benchmark_task(
    tmp_path: Path,
) -> None:
    task = tmp_path / "benchmark.txt"
    task.write_text(
        "Run a benchmark, calculate protocol overhead, and validate results.",
        encoding="utf-8",
    )
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


def test_protocol_mode_routes_code_task_to_executor_without_task_specific_code(
    tmp_path: Path,
) -> None:
    task = tmp_path / "code_task.txt"
    task.write_text("Write Python code that validates a small task.", encoding="utf-8")
    paths = RuntimePaths(root=tmp_path)

    result = run_protocol_mode(task, paths)

    assert "executor" in result.metrics.selected_agents
    assert "sandbox exit 0" in result.answer
    state_store = StateStore(paths)
    executor_payloads = [
        state_store.get(record.ref)[1]
        for record in state_store.list_by_trace(result.trace_id)
        if record.producer == "executor" and record.state_type == StateType.CODE_RESULT
    ]
    assert executor_payloads
    assert executor_payloads[0]["exit_code"] == 0
    assert '"status": "validated"' in executor_payloads[0]["stdout"]
    assert "quick_sort" not in executor_payloads[0]["codeact"]["code"]


def test_protocol_mode_creates_requested_python_file_without_fixed_filename(
    tmp_path: Path,
) -> None:
    task = tmp_path / "create_python_file.txt"
    task.write_text("Create a Python file that validates the current task.", encoding="utf-8")
    paths = RuntimePaths(root=tmp_path)

    result = run_protocol_mode(task, paths)

    generated = tmp_path / "generated_code.py"
    assert generated.exists()
    content = generated.read_text(encoding="utf-8")
    assert "'status': 'validated'" in content
    assert "quick_sort" not in content
    assert "generated_code.py" in result.answer

    state_store = StateStore(paths)
    executor_payloads = [
        state_store.get(record.ref)[1]
        for record in state_store.list_by_trace(result.trace_id)
        if record.producer == "executor" and record.state_type == StateType.CODE_RESULT
    ]
    assert executor_payloads[0]["generated_files"][0]["written"] is True
    assert executor_payloads[0]["generated_files"][0]["relative_path"] == "generated_code.py"


def test_protocol_mode_handles_algorithm_request_without_baked_in_solution(
    tmp_path: Path,
) -> None:
    task = tmp_path / "algorithm_request.txt"
    task.write_text(
        "Implement an algorithm in Python and include a self-check.",
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
    assert '"status": "validated"' in stdout
    assert "quick_sort" not in executor_payloads[0]["codeact"]["code"]


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
