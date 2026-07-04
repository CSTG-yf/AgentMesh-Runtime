from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import orjson
from pydantic import BaseModel, Field

_CAPABILITY_BY_AGENT = {
    "planner": "plan.create",
    "retriever": "memory.semantic_search",
    "executor": "tool.run_python",
    "summarizer": "summary.create",
}


class PlannerDecision(BaseModel):
    intent: str = "analysis"
    task_type: str = "general"
    required_capabilities: list[str] = Field(default_factory=list)
    execution_route: list[str] = Field(default_factory=list)
    need_retrieval: bool = True
    need_tool_execution: bool = False
    need_summary: bool = True
    reason: str = "rule-based fallback"

    @classmethod
    def from_task(
        cls,
        task: str,
        capability_to_agent: Mapping[str, str] | None = None,
    ) -> PlannerDecision:
        return _rule_based_policy(task).normalized(capability_to_agent=capability_to_agent)

    @classmethod
    def from_llm_or_task(
        cls,
        task: str,
        llm_text: str | None,
        capability_to_agent: Mapping[str, str] | None = None,
    ) -> PlannerDecision:
        fallback = cls.from_task(task, capability_to_agent=capability_to_agent)
        if not llm_text:
            return fallback
        try:
            loaded: Any = orjson.loads(_json_object_text(llm_text))
            decision = cls.model_validate(loaded).normalized(
                capability_to_agent=capability_to_agent
            )
        except Exception:
            return fallback
        if not decision.execution_route or not decision.required_capabilities:
            return fallback
        if fallback.need_tool_execution and not decision.need_tool_execution:
            decision = decision.model_copy(
                update={
                    "intent": "validation",
                    "need_tool_execution": True,
                    "reason": f"{decision.reason}; task keywords require tool execution",
                }
            ).normalized(capability_to_agent=capability_to_agent)
        return decision

    def normalized(
        self,
        capability_to_agent: Mapping[str, str] | None = None,
    ) -> PlannerDecision:
        route = [agent for agent in self.execution_route if isinstance(agent, str) and agent]
        capabilities = [
            capability
            for capability in self.required_capabilities
            if isinstance(capability, str) and capability
        ]
        need_retrieval = (
            self.need_retrieval
            or "retriever" in route
            or "memory.semantic_search" in capabilities
            or "evidence.refine" in capabilities
        )
        need_tool = (
            self.need_tool_execution
            or "executor" in route
            or "tool.run_python" in capabilities
            or "codeact.generate" in capabilities
        )
        need_summary = True if self.need_summary else "summarizer" in route
        normalized_route = [_agent_for_capability("plan.create", capability_to_agent)]
        if need_retrieval:
            normalized_route.append(
                _agent_for_capability("memory.semantic_search", capability_to_agent)
            )
        if need_tool:
            normalized_route.append(_agent_for_capability("tool.run_python", capability_to_agent))
        if need_summary:
            normalized_route.append(_agent_for_capability("summary.create", capability_to_agent))
        normalized_capabilities = _capabilities_for_route(normalized_route, capability_to_agent)
        intent = self.intent or ("validation" if need_tool else "analysis")
        if need_tool and intent == "analysis":
            intent = "validation"
        return self.model_copy(
            update={
                "intent": intent,
                "required_capabilities": list(dict.fromkeys(normalized_capabilities)),
                "execution_route": list(dict.fromkeys(normalized_route)),
                "need_retrieval": need_retrieval,
                "need_tool_execution": need_tool,
                "need_summary": need_summary,
            }
        )

    def for_policy(
        self,
        policy_version: str,
        capability_to_agent: Mapping[str, str] | None = None,
        *,
        task: str | None = None,
        trusted_decision: PlannerDecision | None = None,
    ) -> PlannerDecision:
        if policy_version not in {"quality_safe_v1", "quality_safe_v2"}:
            return self
        trusted = trusted_decision
        if trusted is None and task is not None:
            trusted = PlannerDecision.from_task(
                task,
                capability_to_agent=capability_to_agent,
            )
        if (
            policy_version == "quality_safe_v2"
            and task is not None
            and _is_algorithmic_search_task(task)
            and not _has_explicit_retrieval_intent(task)
            and (
                trusted is None
                or trusted.task_type
                in {
                    "knowledge_lookup",
                    "analysis_or_report",
                    "general",
                    "code_execution",
                    "calculation",
                    "benchmark",
                }
            )
        ):
            retriever = _agent_for_capability(
                "memory.semantic_search", capability_to_agent
            )
            need_tool_execution = self.need_tool_execution or bool(
                trusted is not None and trusted.need_tool_execution
            )
            return self.model_copy(
                update={
                    "intent": (
                        "validation" if need_tool_execution else "analysis"
                    ),
                    "task_type": (
                        trusted.task_type
                        if need_tool_execution and trusted is not None
                        else "analysis_or_report"
                    ),
                    "required_capabilities": [
                        capability
                        for capability in self.required_capabilities
                        if capability
                        not in {"memory.semantic_search", "evidence.refine"}
                    ],
                    "execution_route": [
                        agent
                        for agent in self.execution_route
                        if agent not in {"retriever", retriever}
                    ],
                    "need_retrieval": False,
                    "need_tool_execution": need_tool_execution,
                    "reason": (
                        f"{self.reason}; "
                        "quality_safe_v2: algorithmic search analysis"
                    ),
                }
            ).normalized(capability_to_agent=capability_to_agent)
        safe_task_types = {"chat", "summary_only", "analysis_or_report"}
        if trusted is not None and (
            trusted.need_tool_execution
            or trusted.task_type not in safe_task_types
        ):
            return self.model_copy(
                update={
                    "need_retrieval": self.need_retrieval
                    or trusted.need_retrieval,
                    "need_tool_execution": self.need_tool_execution
                    or trusted.need_tool_execution,
                    "reason": (
                        f"{self.reason}; {policy_version}: trusted route safety floor"
                    ),
                }
            ).normalized(capability_to_agent=capability_to_agent)
        if self.need_tool_execution or self.task_type not in safe_task_types:
            return self
        retriever = _agent_for_capability(
            "memory.semantic_search", capability_to_agent
        )
        decision = self.model_copy(
            update={
                "required_capabilities": [
                    capability
                    for capability in self.required_capabilities
                    if capability
                    not in {"memory.semantic_search", "evidence.refine"}
                ],
                "execution_route": [
                    agent
                    for agent in self.execution_route
                    if agent not in {"retriever", retriever}
                ],
                "need_retrieval": False,
                "reason": (
                    f"{self.reason}; {policy_version}: retrieval not required"
                ),
            }
        )
        return decision.normalized(capability_to_agent=capability_to_agent)


def _rule_based_policy(task: str) -> PlannerDecision:
    text = re.sub(r"\s+", " ", task.lower()).strip()
    if not text:
        return PlannerDecision(
            intent="clarification",
            task_type="empty",
            need_retrieval=False,
            need_tool_execution=False,
            reason="empty input; ask for the user's goal",
        )
    if _is_direct_conversation(text):
        return PlannerDecision(
            intent="conversation",
            task_type="chat",
            need_retrieval=False,
            need_tool_execution=False,
            reason="direct conversational response",
        )
    if _is_memory_management(text):
        return PlannerDecision(
            intent="memory",
            task_type="memory_management",
            need_retrieval=True,
            need_tool_execution=False,
            reason="memory request needs scoped memory context",
        )
    if _is_benchmark_or_validation(text):
        return PlannerDecision(
            intent="validation",
            task_type="benchmark",
            need_retrieval=True,
            need_tool_execution=True,
            reason="benchmark or validation needs evidence and executor",
        )
    if _is_code_review(text):
        return PlannerDecision(
            intent="review",
            task_type="code_review",
            need_retrieval=True,
            need_tool_execution=False,
            reason="code review needs repository context before findings",
        )
    if _is_code_execution(text):
        return PlannerDecision(
            intent="validation",
            task_type="code_execution",
            need_retrieval=True,
            need_tool_execution=True,
            reason="code/run/test/debug request needs executor",
        )
    if _is_calculation_or_data_task(text):
        return PlannerDecision(
            intent="validation",
            task_type="calculation",
            need_retrieval=False,
            need_tool_execution=True,
            reason="deterministic calculation should use executor without memory",
        )
    if _is_summary_only(text):
        return PlannerDecision(
            intent="summarization",
            task_type="summary_only",
            need_retrieval=False,
            need_tool_execution=False,
            reason="user asked to summarize provided context only",
        )
    if _is_retrieval_task(text):
        return PlannerDecision(
            intent="retrieval",
            task_type="knowledge_lookup",
            need_retrieval=True,
            need_tool_execution=False,
            reason="factual or historical query needs retriever grounding",
        )
    if _is_report_or_analysis(text):
        return PlannerDecision(
            intent="analysis",
            task_type="analysis_or_report",
            need_retrieval=True,
            need_tool_execution=False,
            reason="analysis/report should gather evidence before summary",
        )
    return PlannerDecision(
        intent="analysis",
        task_type="general",
        need_retrieval=True,
        need_tool_execution=False,
        reason="general task gets lightweight evidence before summary",
    )


def _is_direct_conversation(text: str) -> bool:
    direct = _contains_any(
        text,
        [
            "\u4f60\u662f\u8c01",
            "\u4f60\u662f\u4ec0\u4e48",
            "\u4ecb\u7ecd\u4e00\u4e0b\u4f60",
            "\u4f60\u80fd\u505a\u4ec0\u4e48",
            "\u4f60\u597d",
            "\u8c22\u8c22",
            "who are you",
            "what are you",
            "introduce yourself",
            "what can you do",
            "hello",
            "hi",
            "thanks",
        ],
    )
    work_intent = _contains_any(
        text,
        [
            "\u5e2e\u6211",
            "\u8bf7\u4f60",
            "\u5b9e\u73b0",
            "\u4fee\u590d",
            "\u8c03\u8bd5",
            "\u6d4b\u8bd5",
            "\u5206\u6790",
            "\u68c0\u7d22",
            "\u62a5\u544a",
            "\u4ee3\u7801",
            "help me",
            "please",
            "implement",
            "fix",
            "debug",
            "test",
            "analyze",
            "search",
            "report",
            "code",
        ],
    )
    return direct and not work_intent


def _is_memory_management(text: str) -> bool:
    return _contains_any(
        text,
        [
            "\u8bb0\u4f4f",
            "\u8bb0\u5fc6",
            "\u5fd8\u8bb0",
            "\u5220\u9664\u8bb0\u5fc6",
            "\u6e05\u9664\u8bb0\u5fc6",
            "\u4f60\u8fd8\u8bb0\u5f97",
            "\u5386\u53f2",
            "\u504f\u597d",
            "remember",
            "memory",
            "forget",
            "delete memory",
            "clear memory",
            "preference",
        ],
    )


def _is_code_review(text: str) -> bool:
    return _contains_any(
        text,
        [
            "review",
            "code review",
            "pr review",
            "\u5ba1\u67e5",
            "\u4ee3\u7801\u5ba1\u67e5",
            "\u5b89\u5168\u5ba1\u8ba1",
            "\u6f0f\u6d1e",
            "\u6027\u80fd\u95ee\u9898",
            "security audit",
            "vulnerability",
            "performance issue",
        ],
    )


def _is_benchmark_or_validation(text: str) -> bool:
    return _contains_any(
        text,
        [
            "benchmark",
            "bench",
            "\u8bc4\u6d4b",
            "\u57fa\u51c6",
            "\u9a8c\u8bc1",
            "validate",
            "validation",
            "verify",
        ],
    )


def _is_code_execution(text: str) -> bool:
    return _contains_any(
        text,
        [
            "\u4ee3\u7801",
            "\u811a\u672c",
            "\u51fd\u6570",
            "\u5b9e\u73b0",
            "\u4fee\u590d",
            "\u8c03\u8bd5",
            "\u6d4b\u8bd5",
            "\u8fd0\u884c",
            "\u6267\u884c",
            "\u62a5\u9519",
            "\u4ed3\u5e93",
            "\u6587\u4ef6",
            "\u6392\u5e8f",
            "\u8f93\u5165\u8f93\u51fa",
            "\u6253\u5370\u7ed3\u679c",
            "code",
            "script",
            "function",
            "implement",
            "fix",
            "debug",
            "test",
            "run",
            "execute",
            "error",
            "repo",
            "repository",
            "file",
            ".py",
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            ".rs",
            ".go",
            ".java",
            "pytest",
            "npm",
            "cargo",
            "uv run",
            "traceback",
            "exception",
            "sort",
            "print result",
        ],
    )


def _is_calculation_or_data_task(text: str) -> bool:
    return _contains_any(
        text,
        [
            "\u8ba1\u7b97",
            "\u7b97\u4e00\u4e0b",
            "\u7edf\u8ba1",
            "\u6392\u5e8f",
            "\u8868\u683c",
            "calculate",
            "compute",
            "count",
            "statistics",
            "sort",
        ],
    )


def _is_retrieval_task(text: str) -> bool:
    return _contains_any(
        text,
        [
            "\u68c0\u7d22",
            "\u67e5\u627e",
            "\u641c\u7d22",
            "\u8d44\u6599",
            "\u6587\u6863",
            "\u8bc1\u636e",
            "\u6765\u6e90",
            "\u5f15\u7528",
            "\u6700\u65b0",
            "retrieve",
            "search",
            "find",
            "docs",
            "document",
            "evidence",
            "source",
            "citation",
            "latest",
        ],
    )


_ALGORITHM_SEARCH_PATTERN = re.compile(
    r"\b(?:binary|linear|sequential|interpolation|exponential|"
    r"ternary|jump|depth[- ]first|breadth[- ]first)\s+search\b"
    r"|\bsearch\s+algorithms?\b"
)


def _is_algorithmic_search_task(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    has_algorithm_context = bool(
        _ALGORITHM_SEARCH_PATTERN.search(normalized)
        or _contains_any(
            normalized,
            ["二分查找", "线性查找", "顺序查找", "搜索算法", "查找算法"],
        )
    )
    has_direct_algorithm_intent = bool(
        re.search(
            r"^(?:please\s+)?(?:compare|analy[sz]e|explain|implement|"
            r"describe|evaluate|contrast)\b",
            normalized,
        )
        or re.search(
            r"^(?:请)?(?:比较|对比|分析|解释|实现|描述|评估)",
            normalized,
        )
    )
    return has_algorithm_context and has_direct_algorithm_intent


def _has_explicit_retrieval_intent(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    without_algorithm_terms = _ALGORITHM_SEARCH_PATTERN.sub(
        " algorithm ",
        normalized,
    )
    for algorithm_term in [
        "二分查找",
        "线性查找",
        "顺序查找",
        "搜索算法",
        "查找算法",
    ]:
        without_algorithm_terms = without_algorithm_terms.replace(
            algorithm_term,
            " 算法 ",
        )
    has_retrieval_action = bool(
        re.search(
            r"\b(?:search|find|research|look\s+up|consult|"
            r"retrieve|query|browse)\b",
            without_algorithm_terms,
        )
        or re.search(r"\buse\b.+\bto\b", without_algorithm_terms)
        or _contains_any(
            without_algorithm_terms,
            ["检索", "查找", "搜索", "查询", "浏览", "查阅", "参考", "使用"],
        )
    )
    has_retrieval_object = bool(
        re.search(
            r"\b(?:memory|knowledge|facts?|sources?|evidence|citations?)\b",
            without_algorithm_terms,
        )
        or _contains_any(
            without_algorithm_terms,
            ["记忆", "知识", "事实", "来源", "证据", "引用"],
        )
    )
    has_source_attribution = bool(
        re.search(
            r"\b(?:according\s+to|based\s+on|per|using|from|"
            r"supplied\s+by|provided\s+by)\b",
            normalized,
        )
        or re.search(
            r"\bwith\b.{0,80}\b(?:data|records?|guidance|sources?)\b",
            normalized,
        )
        or re.search(
            r"\b(?:data|records?|guidance|sources?)\b.{0,80}"
            r"\b(?:from|supplied\s+by|provided\s+by)\b",
            normalized,
        )
        or _contains_any(
            normalized,
            ["根据", "基于", "来自", "引用", "按照", "依照"],
        )
        or bool(
            re.search(
                r"使用.{0,40}(?:资料|数据|记录|指引|来源)",
                normalized,
            )
        )
        or bool(
            re.search(
                r"(?:资料|数据|记录|指引|来源).{0,40}"
                r"(?:来自|由.{0,20}提供)",
                normalized,
            )
        )
    )
    return has_retrieval_action or has_retrieval_object or has_source_attribution


def _is_report_or_analysis(text: str) -> bool:
    return _contains_any(
        text,
        [
            "\u5206\u6790",
            "\u65b9\u6848",
            "\u62a5\u544a",
            "\u603b\u7ed3",
            "\u5bf9\u6bd4",
            "\u8bc4\u4f30",
            "\u8bbe\u8ba1",
            "\u67b6\u6784",
            "benchmark",
            "analyze",
            "analysis",
            "report",
            "compare",
            "evaluate",
            "design",
            "architecture",
        ],
    )


def _is_summary_only(text: str) -> bool:
    return _contains_any(
        text,
        [
            "\u53ea\u603b\u7ed3",
            "\u603b\u7ed3\u5df2\u6709",
            "\u4e0d\u9700\u8981\u68c0\u7d22",
            "summary only",
            "summarize existing",
            "no retrieval",
        ],
    )


def _agent_for_capability(
    capability: str,
    capability_to_agent: Mapping[str, str] | None,
) -> str:
    if capability_to_agent and capability in capability_to_agent:
        return capability_to_agent[capability]
    for agent, agent_capability in _CAPABILITY_BY_AGENT.items():
        if agent_capability == capability:
            return agent
    return capability


def _capabilities_for_route(
    route: list[str],
    capability_to_agent: Mapping[str, str] | None,
) -> list[str]:
    capabilities: list[str] = []
    for agent in route:
        capability = _CAPABILITY_BY_AGENT.get(agent)
        if capability is None and capability_to_agent:
            capability = next(
                (
                    candidate_capability
                    for candidate_capability, candidate_agent in capability_to_agent.items()
                    if candidate_agent == agent
                ),
                None,
            )
        if capability is not None:
            capabilities.append(capability)
    return capabilities


def _json_object_text(content: str) -> str:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        return content
    return content[start : end + 1]


def _contains_any(text: str, words: list[str]) -> bool:
    for word in words:
        if word.isascii() and re.fullmatch(r"[a-z0-9_-]+", word):
            if re.search(rf"\b{re.escape(word)}\b", text):
                return True
            continue
        if word in text:
            return True
    return False
