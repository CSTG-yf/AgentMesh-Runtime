from __future__ import annotations

import re
from typing import Any

import orjson
from pydantic import BaseModel, Field


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
            "memory",
            "evidence",
            "retrieve",
            "search",
        ]
        summary_only_words = ["总结已有", "只总结", "summarize existing", "summary only"]
        need_tool = _contains_any(lowered, tool_words)
        need_retrieval = _contains_any(lowered, retrieval_words) or not _contains_any(
            lowered,
            summary_only_words,
        )
        intent = "validation" if need_tool else "analysis"
        task_type = "benchmark" if "benchmark" in lowered or "评测" in lowered else "general"
        capabilities = ["plan.create"]
        route = ["planner"]
        if need_retrieval:
            capabilities.append("memory.semantic_search")
            route.append("retriever")
        if need_tool:
            capabilities.append("tool.run_python")
            route.append("executor")
        capabilities.append("summary.create")
        route.append("summarizer")
        return cls(
            intent=intent,
            task_type=task_type,
            required_capabilities=capabilities,
            execution_route=route,
            need_retrieval=need_retrieval,
            need_tool_execution=need_tool,
            need_summary=True,
            reason="matched tool/retrieval keywords",
        )

    @classmethod
    def from_llm_or_task(cls, task: str, llm_text: str | None) -> PlannerDecision:
        fallback = cls.from_task(task)
        if not llm_text:
            return fallback
        try:
            loaded: Any = orjson.loads(_json_object_text(llm_text))
            decision = cls.model_validate(loaded)
        except Exception:
            return fallback
        if not decision.execution_route or not decision.required_capabilities:
            return fallback
        return decision


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
