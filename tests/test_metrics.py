from agentmesh.eval.metrics import TOKEN_ESTIMATOR, RunMetrics, estimate_tokens


def test_mixed_cjk_token_estimator_counts_chinese_more_accurately() -> None:
    assert TOKEN_ESTIMATOR == "mixed_cjk"
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd" * 4) == 4
    assert estimate_tokens("\u5feb\u901f\u6392\u5e8f") == 2


def test_run_metrics_records_token_estimator() -> None:
    metrics = RunMetrics()

    assert metrics.token_estimator == TOKEN_ESTIMATOR
    assert metrics.answer_quality_score is None
    assert metrics.quality_rule_id == "unscored"
    assert metrics.quality_passed is None
