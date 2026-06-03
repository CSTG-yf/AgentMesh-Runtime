from pydantic import BaseModel


class MemoryScorer(BaseModel):
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
