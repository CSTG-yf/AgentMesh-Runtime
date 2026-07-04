import csv
from pathlib import Path

import orjson
import pytest

from agentmesh.eval.experiment import BenchmarkTrack, ExperimentManifest
from agentmesh.eval.llm_comparison import (
    LLMExperimentResult,
    compare_llm_experiments,
    load_llm_experiment_result,
)
from agentmesh.storage.paths import RuntimePaths


def _experiment(**overrides: object) -> LLMExperimentResult:
    values: dict[str, object] = {
        "profile_id": "p3-profile",
        "suite_sha256": "a" * 64,
        "model_fingerprint": "model-a",
        "quality_rules_sha256": "b" * 64,
        "repeat_count": 3,
        "text_quality_mean": 0.9,
        "text_quality_pass_rate": 0.8,
        "protocol_quality_mean": 0.9,
        "protocol_quality_pass_rate": 0.8,
        "protocol_token_mean": 100.0,
        "protocol_latency_p50": 200.0,
        "protocol_latency_p95": 400.0,
    }
    values.update(overrides)
    return LLMExperimentResult.model_validate(values)


def test_comparison_rejects_incompatible_model_fingerprints() -> None:
    with pytest.raises(ValueError, match="model fingerprint"):
        compare_llm_experiments(
            baseline=_experiment(model_fingerprint="model-a"),
            candidate=_experiment(model_fingerprint="model-b"),
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("suite_sha256", "suite fingerprint"),
        ("quality_rules_sha256", "quality rules fingerprint"),
        ("repeat_count", "repeat count"),
    ],
)
def test_comparison_rejects_incompatible_experiments(
    field: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        compare_llm_experiments(
            baseline=_experiment(),
            candidate=_experiment(**{field: "different" if field != "repeat_count" else 2}),
        )


def test_quality_guard_blocks_efficiency_claim_when_quality_drops() -> None:
    result = compare_llm_experiments(
        baseline=_experiment(protocol_quality_mean=0.9),
        candidate=_experiment(
            protocol_quality_mean=0.8,
            protocol_token_mean=50,
            protocol_latency_p50=100,
            protocol_latency_p95=200,
        ),
    )

    assert not result.quality_gate_passed
    assert not result.efficiency_claim_allowed
    assert "quality mean regressed" in result.reasons


def test_quality_guard_enforces_candidate_text_tolerances() -> None:
    result = compare_llm_experiments(
        baseline=_experiment(protocol_quality_mean=0.8, protocol_quality_pass_rate=0.7),
        candidate=_experiment(
            text_quality_mean=0.9,
            text_quality_pass_rate=0.9,
            protocol_quality_mean=0.84,
            protocol_quality_pass_rate=0.84,
            protocol_token_mean=70,
            protocol_latency_p50=150,
        ),
    )

    assert not result.quality_gate_passed
    assert "candidate Protocol quality mean is outside Text tolerance" in result.reasons
    assert "candidate Protocol pass rate is outside Text tolerance" in result.reasons


def test_quality_guard_accepts_lower_tokens_and_latency() -> None:
    result = compare_llm_experiments(
        baseline=_experiment(),
        candidate=_experiment(
            protocol_token_mean=70,
            protocol_latency_p50=150,
            protocol_latency_p95=390,
        ),
    )

    assert result.quality_gate_passed
    assert result.efficiency_gate_passed
    assert result.efficiency_claim_allowed
    assert result.protocol_token_reduction_rate == pytest.approx(0.3)


def test_efficiency_guard_requires_lower_tokens_and_safe_latency() -> None:
    no_token_gain = compare_llm_experiments(
        baseline=_experiment(),
        candidate=_experiment(protocol_token_mean=100, protocol_latency_p50=150),
    )
    excessive_p95_regression = compare_llm_experiments(
        baseline=_experiment(),
        candidate=_experiment(
            protocol_token_mean=90,
            protocol_latency_p50=150,
            protocol_latency_p95=441,
        ),
    )

    assert not no_token_gain.efficiency_gate_passed
    assert "Protocol token mean did not decrease" in no_token_gain.reasons
    assert not excessive_p95_regression.efficiency_gate_passed
    assert "Protocol latency p95 regressed by more than 10%" in (
        excessive_p95_regression.reasons
    )


def test_loader_rejects_incomplete_manifest(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    artifact_dir = paths.benchmark_suite_dir(
        "continuous_tasks", track="llm", variant="candidate"
    )
    artifact_dir.mkdir(parents=True)
    manifest = ExperimentManifest(
        experiment_id="exp-test",
        suite_name="continuous_tasks",
        suite_sha256="a" * 64,
        track=BenchmarkTrack.LLM,
        repeat_count=3,
        seed=42,
        python_version="3.11",
        platform="test",
        environment={},
        profile_id="candidate",
        model_fingerprint="model-a",
        quality_rules_sha256="b" * 64,
        status="in_progress",
        completed_pairs=17,
        expected_pairs=18,
    )
    (artifact_dir / "manifest.json").write_bytes(
        orjson.dumps(manifest.model_dump(mode="json"))
    )

    with pytest.raises(ValueError, match="not complete"):
        load_llm_experiment_result(paths, "continuous_tasks", "candidate")


def test_loader_reads_complete_typed_artifacts(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    artifact_dir = paths.benchmark_suite_dir(
        "continuous_tasks", track="llm", variant="candidate"
    )
    artifact_dir.mkdir(parents=True)
    manifest = ExperimentManifest(
        experiment_id="exp-test",
        suite_name="continuous_tasks",
        suite_sha256="a" * 64,
        track=BenchmarkTrack.LLM,
        repeat_count=3,
        seed=42,
        python_version="3.11",
        platform="test",
        environment={},
        profile_id="candidate",
        model_fingerprint="model-a",
        quality_rules_sha256="b" * 64,
        status="complete",
        completed_pairs=18,
        expected_pairs=18,
    )
    (artifact_dir / "manifest.json").write_bytes(
        orjson.dumps(manifest.model_dump(mode="json"))
    )
    with (artifact_dir / "benchmark_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "experiment_id",
                "repeat_count",
                "quality_rules_sha256",
                "text_quality_mean",
                "text_quality_pass_rate",
                "protocol_quality_mean",
                "protocol_quality_pass_rate",
                "protocol_token_stats",
                "protocol_latency_stats",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "experiment_id": "exp-test",
                "repeat_count": "3",
                "quality_rules_sha256": "b" * 64,
                "text_quality_mean": "0.9",
                "text_quality_pass_rate": "0.8",
                "protocol_quality_mean": "0.9",
                "protocol_quality_pass_rate": "0.8",
                "protocol_token_stats": '{"mean":70}',
                "protocol_latency_stats": '{"p50":150,"p95":390}',
            }
        )

    result = load_llm_experiment_result(paths, "continuous_tasks", "candidate")

    assert result.profile_id == "candidate"
    assert result.protocol_token_mean == 70
    assert result.protocol_latency_p95 == 390
