import csv
from pathlib import Path

import orjson
import pytest
from pydantic import ValidationError

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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "field",
    [
        "text_quality_mean",
        "text_quality_pass_rate",
        "protocol_quality_mean",
        "protocol_quality_pass_rate",
        "protocol_token_mean",
        "protocol_latency_p50",
        "protocol_latency_p95",
    ],
)
def test_typed_experiment_rejects_non_finite_metrics(
    field: str, value: float
) -> None:
    with pytest.raises(ValidationError, match=field):
        _experiment(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("text_quality_mean", -0.01),
        ("text_quality_pass_rate", 1.01),
        ("protocol_quality_mean", 1.01),
        ("protocol_quality_pass_rate", -0.01),
        ("protocol_token_mean", -1),
        ("protocol_latency_p50", -1),
        ("protocol_latency_p95", -1),
    ],
)
def test_typed_experiment_rejects_out_of_range_metrics(
    field: str, value: float
) -> None:
    with pytest.raises(ValidationError, match=field):
        _experiment(**{field: value})


def test_typed_experiment_allows_zero_token_and_latency_metrics() -> None:
    result = _experiment(
        protocol_token_mean=0,
        protocol_latency_p50=0,
        protocol_latency_p95=0,
    )

    assert result.protocol_token_mean == 0
    assert result.protocol_latency_p50 == 0


def test_typed_experiment_rejects_p50_above_p95() -> None:
    with pytest.raises(ValidationError, match="p50.*p95"):
        _experiment(protocol_latency_p50=401, protocol_latency_p95=400)


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize(
    "field",
    [
        "text_quality_mean",
        "text_quality_pass_rate",
        "protocol_quality_mean",
        "protocol_quality_pass_rate",
        "protocol_token_mean",
        "protocol_latency_p50",
        "protocol_latency_p95",
    ],
)
def test_typed_experiment_rejects_boolean_metrics(
    field: str, value: bool
) -> None:
    with pytest.raises(ValidationError, match="boolean"):
        _experiment(**{field: value})


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("protocol_quality_mean", "nan"),
        ("text_quality_pass_rate", "inf"),
        ("protocol_token_stats", '{"mean":-Infinity}'),
        ("protocol_latency_stats", '{"p50":-1,"p95":390}'),
    ],
)
def test_loader_rejects_invalid_metrics(
    tmp_path: Path, field: str, invalid: str
) -> None:
    paths = RuntimePaths(root=tmp_path)
    artifact_dir = _write_complete_artifacts(paths)
    summary_path = artifact_dir / "benchmark_summary.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as file:
        row = list(csv.DictReader(file))[0]
    row[field] = invalid
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises((ValueError, ValidationError), match="finite|greater|invalid"):
        load_llm_experiment_result(paths, "continuous_tasks", "candidate")


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("protocol_quality_mean", "True"),
        ("text_quality_pass_rate", "False"),
        ("protocol_token_stats", '{"mean":true}'),
        ("protocol_latency_stats", '{"p50":false,"p95":390}'),
        ("protocol_latency_stats", '{"p50":150,"p95":true}'),
    ],
)
def test_loader_rejects_boolean_metrics(
    tmp_path: Path, field: str, invalid: str
) -> None:
    paths = RuntimePaths(root=tmp_path)
    artifact_dir = _write_complete_artifacts(paths)
    summary_path = artifact_dir / "benchmark_summary.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as file:
        row = list(csv.DictReader(file))[0]
    row[field] = invalid
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises((ValueError, ValidationError), match="boolean"):
        load_llm_experiment_result(paths, "continuous_tasks", "candidate")


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


def test_loader_rejects_manifest_suite_name_mismatch(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    artifact_dir = _write_complete_artifacts(paths)
    manifest_path = artifact_dir / "manifest.json"
    manifest = orjson.loads(manifest_path.read_bytes())
    manifest["suite_name"] = "other_suite"
    manifest_path.write_bytes(orjson.dumps(manifest))

    with pytest.raises(ValueError, match="suite name mismatch"):
        load_llm_experiment_result(paths, "continuous_tasks", "candidate")


@pytest.mark.parametrize(
    ("header_transform", "message"),
    [
        (lambda fields: fields[1:], "missing required columns"),
        (lambda fields: [*fields, fields[0]], "duplicate columns"),
    ],
)
def test_loader_rejects_invalid_csv_headers(
    tmp_path: Path, header_transform, message: str
) -> None:
    paths = RuntimePaths(root=tmp_path)
    artifact_dir = _write_complete_artifacts(paths)
    summary_path = artifact_dir / "benchmark_summary.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as file:
        row = list(csv.DictReader(file))[0]
    fieldnames = header_transform(list(row))
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(ValueError, match=message):
        load_llm_experiment_result(paths, "continuous_tasks", "candidate")


def test_loader_allows_additional_csv_columns(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    artifact_dir = _write_complete_artifacts(paths)
    summary_path = artifact_dir / "benchmark_summary.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as file:
        row = list(csv.DictReader(file))[0]
    row["future_metric"] = "supported"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    result = load_llm_experiment_result(paths, "continuous_tasks", "candidate")

    assert result.profile_id == "candidate"


def test_loader_reads_complete_typed_artifacts(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    _write_complete_artifacts(paths)

    result = load_llm_experiment_result(paths, "continuous_tasks", "candidate")

    assert result.profile_id == "candidate"
    assert result.protocol_token_mean == 70
    assert result.protocol_latency_p95 == 390


def _write_complete_artifacts(paths: RuntimePaths) -> Path:
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

    return artifact_dir
