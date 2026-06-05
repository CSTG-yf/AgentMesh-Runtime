from __future__ import annotations

import re
from typing import Any

import orjson
from pydantic import BaseModel, Field

from agentmesh.llm.client import ChatMessage, LLMClient


class MemoryClassification(BaseModel):
    topic: str
    tags: list[str] = Field(default_factory=list)
    memory_type: str = "task_summary"
    domain: str = "general"
    importance_score: float = 0.5


class RuleBasedMemoryTagger:
    def classify(self, *, task: str, summary: str, evidence_count: int = 0) -> MemoryClassification:
        text = f"{task}\n{summary}".lower()
        tags: list[str] = []
        candidates = {
            "protocol": ["protocol", "amp", "structured"],
            "state": ["state", "stateref", "embedding", "vector"],
            "memory": ["memory", "reuse", "archive", "retrieval"],
            "benchmark": ["benchmark", "metric", "evaluate", "评测", "指标"],
            "agent": ["agent", "multi-agent", "planner", "retriever"],
            "runtime": ["runtime", "rust", "ipc", "sandbox", "codeact"],
            "chinese": ["中文", "汉语", "chinese"],
        }
        for tag, needles in candidates.items():
            if any(needle in text for needle in needles):
                tags.append(tag)
        if not tags:
            tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text)
            tags = list(dict.fromkeys(tokens[:4])) or ["general"]
        memory_type = "experiment_result" if "benchmark" in tags else "task_summary"
        domain = "agent-runtime" if {"agent", "runtime", "protocol"} & set(tags) else "general"
        importance = 0.35 + min(0.25, evidence_count * 0.05)
        if "memory" in tags or "benchmark" in tags:
            importance += 0.15
        if "protocol" in tags or "state" in tags:
            importance += 0.10
        return MemoryClassification(
            topic=(task.strip()[:80] or "untitled"),
            tags=tags,
            memory_type=memory_type,
            domain=domain,
            importance_score=min(1.0, importance),
        )


class LLMMemoryTagger:
    def __init__(
        self,
        llm_client: LLMClient | None,
        fallback: RuleBasedMemoryTagger | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.fallback = fallback or RuleBasedMemoryTagger()

    def classify(self, *, task: str, summary: str, evidence_count: int = 0) -> MemoryClassification:
        fallback = self.fallback.classify(
            task=task,
            summary=summary,
            evidence_count=evidence_count,
        )
        if self.llm_client is None:
            return fallback
        prompt = (
            "Classify whether this multi-agent task memory is useful for long-term reuse. "
            "Return only compact JSON with keys topic, tags, memory_type, domain, "
            "importance_score. importance_score must be 0..1.\n\n"
            f"Task:\n{task[:2000]}\n\nSummary:\n{summary[:1000]}"
        )
        try:
            content = self.llm_client.complete(
                agent_name="memory_tagger",
                messages=[ChatMessage(role="user", content=prompt)],
                variables={"task": task, "summary": summary},
            )
            return _classification_from_json(content, fallback)
        except Exception:
            return fallback


def _classification_from_json(content: str, fallback: MemoryClassification) -> MemoryClassification:
    try:
        loaded: Any = orjson.loads(_json_object_text(content))
    except Exception:
        return fallback
    if not isinstance(loaded, dict):
        return fallback
    merged = fallback.model_dump()
    merged.update({key: value for key, value in loaded.items() if value is not None})
    if not isinstance(merged.get("tags"), list):
        merged["tags"] = fallback.tags
    merged["tags"] = [str(tag)[:40] for tag in merged["tags"]][:8]
    try:
        merged["importance_score"] = max(0.0, min(1.0, float(merged["importance_score"])))
    except Exception:
        merged["importance_score"] = fallback.importance_score
    return MemoryClassification.model_validate(merged)


def _json_object_text(content: str) -> str:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        return content
    return content[start : end + 1]
