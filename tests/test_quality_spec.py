from agentmesh.eval.quality_spec import QualitySpec, evaluate_quality


def test_contains_all_quality_is_case_insensitive() -> None:
    spec = QualitySpec(
        rule_id="sum-output-v1",
        kind="contains_all",
        expected=["sum: 77", "even count: 5"],
    )

    result = evaluate_quality("Sum: 77\nEven count: 5", spec)

    assert result.scored
    assert result.passed
    assert result.score == 1.0
    assert result.matched == 2
    assert result.total == 2


def test_contains_all_quality_reports_partial_credit() -> None:
    spec = QualitySpec(
        rule_id="two-facts-v1",
        kind="contains_all",
        expected=["alpha", "beta"],
    )

    result = evaluate_quality("Alpha only", spec)

    assert result.scored
    assert not result.passed
    assert result.score == 0.5


def test_regex_and_non_empty_rules() -> None:
    regex = evaluate_quality(
        "result=42",
        QualitySpec(rule_id="number-v1", kind="regex", pattern=r"result\s*=\s*42"),
    )
    non_empty = evaluate_quality(
        "  ",
        QualitySpec(rule_id="answer-v1", kind="non_empty"),
    )

    assert regex.passed and regex.score == 1.0
    assert not non_empty.passed and non_empty.score == 0.0


def test_missing_quality_rule_is_unscored() -> None:
    result = evaluate_quality("arbitrary answer", None)

    assert not result.scored
    assert result.score is None
    assert result.reason == "quality rule not configured"
