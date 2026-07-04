from __future__ import annotations

import csv
from pathlib import Path

import orjson
from pydantic import BaseModel, ConfigDict

from agentmesh.eval.experiment import BenchmarkTrack, ExperimentManifest
from agentmesh.storage.paths import RuntimePaths


class LLMExperimentResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile_id: str
    suite_sha256: str
    model_fingerprint: str
    quality_rules_sha256: str
    repeat_count: int
    text_quality_mean: float
    text_quality_pass_rate: float
    protocol_quality_mean: float
    protocol_quality_pass_rate: float
    protocol_token_mean: float
    protocol_latency_p50: float
    protocol_latency_p95: float


class LLMComparisonResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    compatible: bool
    quality_gate_passed: bool
    efficiency_gate_passed: bool
    efficiency_claim_allowed: bool
    protocol_quality_mean_delta: float
    protocol_quality_pass_rate_delta: float
    protocol_token_reduction_rate: float | None
    protocol_latency_p50_reduction_rate: float | None
    protocol_latency_p95_reduction_rate: float | None
    reasons: list[str]


def compare_llm_experiments(
    *,
    baseline: LLMExperimentResult,
    candidate: LLMExperimentResult,
) -> LLMComparisonResult:
    _validate_compatibility(baseline, candidate)
    reasons: list[str] = []

    if candidate.protocol_quality_mean < baseline.protocol_quality_mean:
        reasons.append("quality mean regressed")
    if candidate.protocol_quality_pass_rate < baseline.protocol_quality_pass_rate:
        reasons.append("quality pass rate regressed")
    if candidate.protocol_quality_mean < candidate.text_quality_mean - 0.05:
        reasons.append("candidate Protocol quality mean is outside Text tolerance")
    if (
        candidate.protocol_quality_pass_rate
        < candidate.text_quality_pass_rate - (1 / 18)
    ):
        reasons.append("candidate Protocol pass rate is outside Text tolerance")
    quality_ok = not reasons

    token_rate = _reduction_rate(
        baseline.protocol_token_mean, candidate.protocol_token_mean
    )
    p50_rate = _reduction_rate(
        baseline.protocol_latency_p50, candidate.protocol_latency_p50
    )
    p95_rate = _reduction_rate(
        baseline.protocol_latency_p95, candidate.protocol_latency_p95
    )
    efficiency_reasons: list[str] = []
    if candidate.protocol_token_mean >= baseline.protocol_token_mean:
        efficiency_reasons.append("Protocol token mean did not decrease")
    p50_improved = candidate.protocol_latency_p50 < baseline.protocol_latency_p50
    p95_improved = candidate.protocol_latency_p95 < baseline.protocol_latency_p95
    if not (p50_improved or p95_improved):
        efficiency_reasons.append("neither Protocol latency percentile improved")
    if candidate.protocol_latency_p50 > baseline.protocol_latency_p50 * 1.1:
        efficiency_reasons.append("Protocol latency p50 regressed by more than 10%")
    if candidate.protocol_latency_p95 > baseline.protocol_latency_p95 * 1.1:
        efficiency_reasons.append("Protocol latency p95 regressed by more than 10%")
    reasons.extend(efficiency_reasons)
    efficiency_ok = not efficiency_reasons
    if quality_ok and efficiency_ok:
        reasons.append("quality and efficiency gates passed")

    return LLMComparisonResult(
        compatible=True,
        quality_gate_passed=quality_ok,
        efficiency_gate_passed=efficiency_ok,
        efficiency_claim_allowed=quality_ok and efficiency_ok,
        protocol_quality_mean_delta=(
            candidate.protocol_quality_mean - baseline.protocol_quality_mean
        ),
        protocol_quality_pass_rate_delta=(
            candidate.protocol_quality_pass_rate
            - baseline.protocol_quality_pass_rate
        ),
        protocol_token_reduction_rate=token_rate,
        protocol_latency_p50_reduction_rate=p50_rate,
        protocol_latency_p95_reduction_rate=p95_rate,
        reasons=reasons,
    )


def load_llm_experiment_result(
    paths: RuntimePaths,
    suite_name: str,
    variant: str,
) -> LLMExperimentResult:
    manifest_path = paths.benchmark_suite_manifest(
        suite_name, BenchmarkTrack.LLM.value, variant
    )
    summary_path = paths.benchmark_suite_summary(
        suite_name, track=BenchmarkTrack.LLM.value, variant=variant
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"experiment manifest not found: {manifest_path}")
    manifest = ExperimentManifest.model_validate(orjson.loads(manifest_path.read_bytes()))
    if (
        manifest.status != "complete"
        or manifest.expected_pairs <= 0
        or manifest.completed_pairs != manifest.expected_pairs
    ):
        raise ValueError(
            f"experiment {variant!r} is not complete "
            f"(status={manifest.status}, "
            f"pairs={manifest.completed_pairs}/{manifest.expected_pairs})"
        )
    if not summary_path.exists():
        raise FileNotFoundError(f"experiment summary not found: {summary_path}")
    if manifest.track is not BenchmarkTrack.LLM:
        raise ValueError(f"experiment {variant!r} is not an LLM benchmark")
    summary = _read_single_summary(summary_path)
    _require_summary_identity(summary, manifest, summary_path)
    token_stats = _json_mapping(summary, "protocol_token_stats")
    latency_stats = _json_mapping(summary, "protocol_latency_stats")
    return LLMExperimentResult(
        profile_id=manifest.profile_id,
        suite_sha256=manifest.suite_sha256,
        model_fingerprint=manifest.model_fingerprint,
        quality_rules_sha256=manifest.quality_rules_sha256,
        repeat_count=manifest.repeat_count,
        text_quality_mean=_required_float(summary, "text_quality_mean"),
        text_quality_pass_rate=_required_float(summary, "text_quality_pass_rate"),
        protocol_quality_mean=_required_float(summary, "protocol_quality_mean"),
        protocol_quality_pass_rate=_required_float(
            summary, "protocol_quality_pass_rate"
        ),
        protocol_token_mean=_required_number(token_stats, "mean"),
        protocol_latency_p50=_required_number(latency_stats, "p50"),
        protocol_latency_p95=_required_number(latency_stats, "p95"),
    )


def compare_llm_artifacts(
    paths: RuntimePaths,
    *,
    suite_name: str,
    baseline_variant: str,
    candidate_variant: str,
) -> LLMComparisonResult:
    return compare_llm_experiments(
        baseline=load_llm_experiment_result(paths, suite_name, baseline_variant),
        candidate=load_llm_experiment_result(paths, suite_name, candidate_variant),
    )


def _validate_compatibility(
    baseline: LLMExperimentResult, candidate: LLMExperimentResult
) -> None:
    checks = [
        ("suite_sha256", "suite fingerprint"),
        ("model_fingerprint", "model fingerprint"),
        ("quality_rules_sha256", "quality rules fingerprint"),
        ("repeat_count", "repeat count"),
    ]
    for field, label in checks:
        if getattr(baseline, field) != getattr(candidate, field):
            raise ValueError(f"incompatible {label}")


def _reduction_rate(baseline: float, candidate: float) -> float | None:
    return (baseline - candidate) / baseline if baseline > 0 else None


def _read_single_summary(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1:
        raise ValueError(f"expected one summary row in {path}, found {len(rows)}")
    return rows[0]


def _require_summary_identity(
    summary: dict[str, str], manifest: ExperimentManifest, path: Path
) -> None:
    expected = {
        "experiment_id": manifest.experiment_id,
        "repeat_count": str(manifest.repeat_count),
        "quality_rules_sha256": manifest.quality_rules_sha256,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"{field} mismatch between manifest and {path}")


def _json_mapping(summary: dict[str, str], field: str) -> dict[str, object]:
    raw = summary.get(field, "")
    try:
        value = orjson.loads(raw)
    except orjson.JSONDecodeError as exc:
        raise ValueError(f"invalid {field} in benchmark summary") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {field} in benchmark summary")
    return value


def _required_float(summary: dict[str, str], field: str) -> float:
    raw = summary.get(field)
    if raw is None or raw == "":
        raise ValueError(f"missing {field} in benchmark summary")
    return float(raw)


def _required_number(values: dict[str, object], field: str) -> float:
    value = values.get(field)
    if not isinstance(value, int | float):
        raise ValueError(f"missing numeric {field} in benchmark statistics")
    return float(value)
