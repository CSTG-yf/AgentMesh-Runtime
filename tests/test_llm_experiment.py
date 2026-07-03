from pathlib import Path

import pytest
from pydantic import ValidationError

from agentmesh.eval.llm_experiment import (
    ProtocolOptimizationProfile,
    load_llm_experiment_profile,
)
from agentmesh.llm.client import ChatMessage, InstrumentedLLMClient
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
