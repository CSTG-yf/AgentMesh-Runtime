from pathlib import Path

import orjson
from typer.testing import CliRunner

from agentmesh.cli import app
from agentmesh.eval.llm_comparison import LLMComparisonResult


def test_config_masks_api_key(tmp_path: Path) -> None:
    secret = "never-print-this-secret"
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "AGENTMESH_LLM_BASE_URL=https://example.test/v1",
                f"AGENTMESH_LLM_API_KEY={secret}",
                "AGENTMESH_LLM_MODEL=test-model",
            ]
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["config", "--root", str(tmp_path)])
    assert result.exit_code == 0
    assert secret not in result.stdout
    assert orjson.loads(result.stdout) == {
        "llm_configured": True,
        "llm_base_url": "https://example.test/v1",
        "llm_model": "test-model",
        "llm_api_key": "configured",
    }


def test_config_uses_empty_strings_and_missing_api_key(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["config", "--root", str(tmp_path)])

    assert result.exit_code == 0
    assert orjson.loads(result.stdout) == {
        "llm_configured": False,
        "llm_base_url": "",
        "llm_model": "",
        "llm_api_key": "missing",
    }


def test_profile_cannot_be_combined_with_no_llm(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--suite",
            "standard",
            "--root",
            str(tmp_path),
            "--profile",
            "profile.yaml",
            "--no-llm",
        ],
    )
    assert result.exit_code == 1
    assert "--profile cannot be combined with --no-llm" in result.stdout


def test_benchmark_compare_prints_machine_readable_gate_result(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "agentmesh.cli.compare_llm_artifacts",
        lambda *_args, **_kwargs: LLMComparisonResult(
            compatible=True,
            quality_gate_passed=False,
            efficiency_gate_passed=True,
            efficiency_claim_allowed=False,
            protocol_quality_mean_delta=-0.1,
            protocol_quality_pass_rate_delta=0,
            protocol_token_reduction_rate=0.3,
            protocol_latency_p50_reduction_rate=0.2,
            protocol_latency_p95_reduction_rate=0.1,
            reasons=["quality mean regressed"],
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "benchmark-compare",
            "--suite",
            "continuous_tasks",
            "--baseline",
            "p3-baseline",
            "--candidate",
            "p3-candidate",
            "--root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    payload = orjson.loads(result.stdout)
    assert payload["efficiency_claim_allowed"] is False
    assert payload["reasons"] == ["quality mean regressed"]


def test_benchmark_compare_input_error_exits_one(
    tmp_path: Path, monkeypatch
) -> None:
    def fail(*_args, **_kwargs):
        raise ValueError("incompatible model fingerprint")

    monkeypatch.setattr("agentmesh.cli.compare_llm_artifacts", fail)

    result = CliRunner().invoke(
        app,
        [
            "benchmark-compare",
            "--suite",
            "continuous_tasks",
            "--baseline",
            "p3-baseline",
            "--candidate",
            "p3-candidate",
            "--root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "incompatible model fingerprint" in result.stdout
