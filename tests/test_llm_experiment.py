import os
from pathlib import Path

import orjson
import pytest
from pydantic import ValidationError

from agentmesh.errors import BenchmarkConfigError
from agentmesh.eval.benchmark import run_benchmark
from agentmesh.eval.llm_experiment import (
    LLMExperimentProfile,
    ProtocolOptimizationProfile,
    load_llm_experiment_profile,
)
from agentmesh.eval.metrics import ModeRunResult, RunMetrics
from agentmesh.llm.client import ChatMessage, InstrumentedLLMClient
from agentmesh.modes.protocol_mode import run_protocol_mode
from agentmesh.storage.jsonl import read_jsonl
from agentmesh.storage.paths import RuntimePaths


def test_load_llm_experiment_profile_is_reproducible(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        """
profile_id: baseline
artifact_label: baseline
repeat: 3
prompt_version: p2
route_policy_version: legacy
optimization_enabled: false
protocol: {}
""",
        encoding="utf-8",
    )
    profile = load_llm_experiment_profile(profile_path)
    assert profile.protocol.planner_max_chars == 1200
    assert profile.sha256 == load_llm_experiment_profile(profile_path).sha256
    assert len(profile.sha256) == 64


@pytest.mark.parametrize("artifact_label", ["a/b", "", " leading", "two words", "a\\b"])
def test_profile_rejects_noncanonical_artifact_label(artifact_label: str) -> None:
    with pytest.raises(ValidationError):
        LLMExperimentProfile(profile_id="profile", artifact_label=artifact_label)


@pytest.mark.parametrize(
    ("field", "value"),
    [("planner_max_chars", 0), ("evidence_min_score", 1.1)],
)
def test_protocol_profile_rejects_invalid_budget(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        ProtocolOptimizationProfile(**{field: value})


def test_benchmark_variant_path_is_isolated_and_optional(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    assert paths.benchmark_suite_dir("suite", "llm") == (
        tmp_path / "runs/latest/benchmarks/suite/llm"
    )
    assert paths.benchmark_suite_dir("suite", "llm", "candidate") == (
        tmp_path / "runs/latest/benchmarks/suite/llm/candidate"
    )


def test_instrumented_llm_client_counts_failure() -> None:
    class FailingClient:
        def complete(self, **_: object) -> str:
            raise RuntimeError("provider down")

    client = InstrumentedLLMClient(FailingClient())
    with pytest.raises(RuntimeError, match="provider down"):
        client.complete(agent_name="planner", messages=[ChatMessage(role="user", content="x")])
    assert client.stats.call_count == 1
    assert client.stats.error_count == 1


def test_instrumented_llm_client_counts_success() -> None:
    class SuccessfulClient:
        def complete(self, **_: object) -> str:
            return "ok"

    client = InstrumentedLLMClient(SuccessfulClient())
    assert client.complete(agent_name="planner", messages=[]) == "ok"
    assert client.stats.call_count == 1
    assert client.stats.error_count == 0


@pytest.mark.parametrize("original", [None, "keep-me"])
def test_benchmark_restores_global_memory_env_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, original: str | None
) -> None:
    suite = _write_suite(tmp_path)
    if original is None:
        monkeypatch.delenv("AGENTMESH_DISABLE_GLOBAL_MEMORY", raising=False)
    else:
        monkeypatch.setenv("AGENTMESH_DISABLE_GLOBAL_MEMORY", original)
    monkeypatch.setattr(
        "agentmesh.eval.benchmark.run_text_mode",
        lambda **_: (_ for _ in ()).throw(BenchmarkConfigError("mode failed")),
    )
    with pytest.raises(BenchmarkConfigError, match="mode failed"):
        run_benchmark(suite, RuntimePaths(root=tmp_path), use_llm=False)
    assert os.environ.get("AGENTMESH_DISABLE_GLOBAL_MEMORY") == original


def test_profile_repeat_forwarding_summary_and_failure_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _write_suite(tmp_path)
    _write_llm_env(tmp_path)
    profile = LLMExperimentProfile(
        profile_id="profile",
        artifact_label="variant",
        repeat=2,
    )
    protocol_profiles: list[LLMExperimentProfile | None] = []
    calls = 0

    def fake_text(
        *, task_path: Path, paths: RuntimePaths, load_configured_llm: bool
    ) -> ModeRunResult:
        del task_path, paths, load_configured_llm
        return ModeRunResult(
            mode="text", trace_id="text", answer="ok", metrics=RunMetrics(llm_call_count=1)
        )

    def fake_protocol(
        *, experiment_profile: LLMExperimentProfile | None = None, **_: object
    ) -> ModeRunResult:
        nonlocal calls
        calls += 1
        protocol_profiles.append(experiment_profile)
        return ModeRunResult(
            mode="protocol",
            trace_id="protocol",
            answer="ok",
            metrics=RunMetrics(
                llm_call_count=1,
                llm_error_count=int(calls == 2),
            ),
        )

    monkeypatch.setattr("agentmesh.eval.benchmark.run_text_mode", fake_text)
    monkeypatch.setattr("agentmesh.eval.benchmark.run_protocol_mode", fake_protocol)
    with pytest.raises(BenchmarkConfigError, match="provider failure"):
        run_benchmark(suite, RuntimePaths(root=tmp_path), profile=profile)
    detail = RuntimePaths(root=tmp_path).benchmark_suite_detail(
        "profile_suite", "llm", "variant"
    )
    assert len(read_jsonl(detail)) == 2
    assert protocol_profiles == [profile, profile]
    manifest = orjson.loads(
        RuntimePaths(root=tmp_path)
        .benchmark_suite_manifest("profile_suite", "llm", "variant")
        .read_bytes()
    )
    assert manifest["repeat_count"] == 2
    assert manifest["status"] == "failed"
    assert manifest["completed_pairs"] == 1
    assert manifest["expected_pairs"] == 2
    assert manifest["failure_reason"] == "llm_provider_error"


def test_profile_success_summary_has_variant_and_quality_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _write_suite(tmp_path)
    _write_llm_env(tmp_path)
    profile = LLMExperimentProfile(
        profile_id="profile", artifact_label="variant", repeat=1
    )

    def result(mode: str) -> ModeRunResult:
        return ModeRunResult(
            mode=mode,
            trace_id=mode,
            answer="ok",
            metrics=RunMetrics(llm_call_count=1),
        )

    monkeypatch.setattr(
        "agentmesh.eval.benchmark.run_text_mode", lambda **_: result("text")
    )
    monkeypatch.setattr(
        "agentmesh.eval.benchmark.run_protocol_mode",
        lambda **_: result("protocol"),
    )
    summary = run_benchmark(
        suite, RuntimePaths(root=tmp_path), profile=profile
    )
    assert summary.repeat_count == 1
    assert summary.artifact_variant == "variant"
    assert len(summary.quality_rules_sha256) == 64
    manifest = orjson.loads(
        RuntimePaths(root=tmp_path)
        .benchmark_suite_manifest("profile_suite", "llm", "variant")
        .read_bytes()
    )
    assert manifest["status"] == "complete"
    assert manifest["completed_pairs"] == 1
    assert manifest["expected_pairs"] == 1
    assert manifest["failure_reason"] == ""


def test_profile_requires_complete_llm_config_before_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _write_suite(tmp_path)
    called = False

    def fake_mode(**_: object) -> ModeRunResult:
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr("agentmesh.eval.benchmark.run_text_mode", fake_mode)
    with pytest.raises(BenchmarkConfigError, match="requires base_url"):
        run_benchmark(
            suite,
            RuntimePaths(root=tmp_path),
            profile=LLMExperimentProfile(profile_id="p", artifact_label="v"),
        )
    assert called is False


def test_task1_profile_does_not_apply_candidate_optimization(tmp_path: Path) -> None:
    task = tmp_path / "task.txt"
    task.write_text("Summarize this task.", encoding="utf-8")
    baseline = run_protocol_mode(
        task,
        RuntimePaths(root=tmp_path / "baseline"),
        load_configured_llm=False,
    )
    candidate = run_protocol_mode(
        task,
        RuntimePaths(root=tmp_path / "candidate"),
        load_configured_llm=False,
        experiment_profile=LLMExperimentProfile(
            profile_id="candidate",
            artifact_label="candidate",
            optimization_enabled=True,
            protocol=ProtocolOptimizationProfile(
                planner_max_chars=1,
                retriever_max_chars=1,
                executor_max_chars=1,
                summarizer_max_chars=1,
                evidence_max_items=1,
                evidence_min_score=1,
            ),
        ),
    )
    assert candidate.answer == baseline.answer
    assert candidate.metrics.agent_io_tokens == baseline.metrics.agent_io_tokens


def _write_suite(root: Path) -> Path:
    (root / "task.txt").write_text("task", encoding="utf-8")
    suite = root / "suite.yaml"
    suite.write_text(
        """
name: profile_suite
repeat: 9
tasks:
  - id: T1
    input_file: task.txt
    quality:
      rule_id: ok
      kind: contains_all
      expected: [ok]
""".strip(),
        encoding="utf-8",
    )
    return suite


def _write_llm_env(root: Path) -> None:
    (root / ".env").write_text(
        "\n".join(
            [
                "AGENTMESH_LLM_BASE_URL=https://example.test/v1",
                "AGENTMESH_LLM_API_KEY=secret",
                "AGENTMESH_LLM_MODEL=model",
            ]
        ),
        encoding="utf-8",
    )
