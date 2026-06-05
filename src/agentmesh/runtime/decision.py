from __future__ import annotations

import re
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
    def from_task(cls, task: str) -> PlannerDecision:
        lowered = task.lower()
        tool_words = [
            "运行",
            "执行",
            "测试",
            "验证",
            "计算",
            "评测",
            "跑一次",
            "benchmark",
            "bench",
            "codeact",
            "run",
            "execute",
            "test",
            "validate",
            "calculate",
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
        ).normalized()

    @classmethod
    def from_llm_or_task(cls, task: str, llm_text: str | None) -> PlannerDecision:
        fallback = cls.from_task(task)
        if not llm_text:
            return fallback
        try:
            loaded: Any = orjson.loads(_json_object_text(llm_text))
            decision = cls.model_validate(loaded).normalized()
        except Exception:
            return fallback
        if not decision.execution_route or not decision.required_capabilities:
            return fallback
        return decision

    def normalized(self) -> PlannerDecision:
        route = [agent for agent in self.execution_route if agent in _ROUTE_ORDER]
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
        normalized_route = ["planner"]
        if need_retrieval:
            normalized_route.append("retriever")
        if need_tool:
            normalized_route.append("executor")
        if need_summary:
            normalized_route.append("summarizer")
        normalized_capabilities = [
            _CAPABILITY_BY_AGENT[agent]
            for agent in normalized_route
            if agent in _CAPABILITY_BY_AGENT
        ]
        intent = self.intent or ("validation" if need_tool else "analysis")
        if need_tool and intent == "analysis":
            intent = "validation"
        return self.model_copy(
            update={
                "intent": intent,
                "required_capabilities": list(dict.fromkeys(normalized_capabilities)),
                "execution_route": normalized_route,
                "need_retrieval": need_retrieval,
                "need_tool_execution": need_tool,
                "need_summary": need_summary,
            }
        )


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
