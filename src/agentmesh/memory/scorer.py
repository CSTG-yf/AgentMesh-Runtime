import math
from datetime import UTC, datetime

from pydantic import BaseModel

from agentmesh.memory.schema import MemoryUnit


class MemoryScorer(BaseModel):
    recency_decay_days: float = 14.0

    def score(
        self,
        semantic_similarity: float,
        tag_match_score: float,
        confidence: float,
        evidence_coverage: float,
        time_decay: float,
    ) -> float:
        return (
            0.45 * semantic_similarity
            + 0.20 * tag_match_score
            + 0.15 * confidence
            + 0.15 * evidence_coverage
            - 0.05 * time_decay
        )

    def rank_score(
        self,
        *,
        memory: MemoryUnit,
        semantic_similarity: float,
        query_tags: list[str] | None = None,
        max_reuse_count: int = 0,
    ) -> tuple[float, dict[str, float]]:
        reuse_score = _safe_ratio(
            math.log1p(memory.reuse_count),
            math.log1p(max_reuse_count),
        )
        recency_score = self.recency_score(memory)
        tag_overlap_score = self.tag_overlap_score(query_tags or [], memory.tags)
        score = (
            0.50 * semantic_similarity
            + 0.15 * memory.validity_score
            + 0.10 * memory.confidence
            + 0.10 * reuse_score
            + 0.10 * recency_score
            + 0.05 * tag_overlap_score
        )
        return score, {
            "semantic_similarity": semantic_similarity,
            "validity_score": memory.validity_score,
            "confidence_score": memory.confidence,
            "reuse_score": reuse_score,
            "recency_score": recency_score,
            "tag_overlap_score": tag_overlap_score,
        }

    def recency_score(self, memory: MemoryUnit) -> float:
        timestamp = memory.last_used_at or memory.created_at
        age_days = max(0.0, (datetime.now(UTC) - timestamp).total_seconds() / 86400)
        return math.exp(-age_days / self.recency_decay_days)

    def tag_overlap_score(self, query_tags: list[str], memory_tags: list[str]) -> float:
        query = set(query_tags)
        memory = set(memory_tags)
        if not query or not memory:
            return 0.0
        return len(query & memory) / len(query | memory)


def _safe_ratio(value: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return min(1.0, max(0.0, value / denominator))
