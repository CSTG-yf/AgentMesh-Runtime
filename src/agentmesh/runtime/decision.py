from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import orjson
from pydantic import BaseModel, Field

_ROUTE_ORDER = ["planner", "retriever", "executor", "summarizer"]
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
        lowered = task.lower()
        tool_words = [
            "运行",
            "执行",
            "测试",
            "验证",
            "计算",
            "评测",
            "跑一次",
            "写代码",
            "写一个",
            "脚本",
            "代码",
            "输出结果",
            "打印",
            "排序",
            "快速排序",
            "快排",
            "函数",
            "用例",
            "输入输出",
            "benchmark",
            "bench",
            "codeact",
            "run",
            "execute",
            "test",
            "validate",
            "calculate",
            "compute",
            "code",
            "script",
            "sort",
            "quicksort",
            "quick sort",
        ]
        retrieval_words = [
            "检索",
            "查找",
            "历史",
            "记忆",
            "证据",
            "复用",
            "memory",
            "evidence",
            "retrieve",
            "search",
        ]
        summary_only_words = [
            "总结已有",
            "只总结",
            "不需要检索",
            "summarize existing",
            "summary only",
        ]
        need_tool = _contains_any(lowered, tool_words)
        need_retrieval = _contains_any(lowered, retrieval_words) or not _contains_any(
            lowered,
            summary_only_words,
        )
        intent = "validation" if need_tool else "analysis"
        task_type = "benchmark" if "benchmark" in lowered or "评测" in lowered else "general"
        return cls(
            intent=intent,
            task_type=task_type,
            need_retrieval=need_retrieval,
            need_tool_execution=need_tool,
            need_summary=True,
            reason="matched tool/retrieval keywords",
        ).normalized(capability_to_agent=capability_to_agent)

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
    if capability_to_agent:
        by_agent = {agent: capability for capability, agent in capability_to_agent.items()}
        return [by_agent[agent] for agent in route if agent in by_agent]
    return [capability for agent in route if (capability := _CAPABILITY_BY_AGENT.get(agent))]


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
