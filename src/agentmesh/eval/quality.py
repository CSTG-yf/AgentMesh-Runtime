def deterministic_quality_score(answer: str) -> float:
    if not answer.strip():
        return 0.0
    return min(1.0, 0.6 + min(len(answer), 400) / 1000)
