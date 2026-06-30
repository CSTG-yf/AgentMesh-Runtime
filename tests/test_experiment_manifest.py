from pathlib import Path

from agentmesh.eval.experiment import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkTrack,
    build_experiment_manifest,
    paired_mode_order,
)


def test_manifest_is_stable_and_does_not_contain_secrets(tmp_path: Path) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: stable\nrepeat: 2\nseed: 17\ntasks: []\n",
        encoding="utf-8",
    )

    first = build_experiment_manifest(
        suite_path=suite,
        suite_name="stable",
        track=BenchmarkTrack.DETERMINISTIC,
        repeat_count=2,
        seed=17,
        environment={"model": "demo", "api_key": "secret"},
    )
    second = build_experiment_manifest(
        suite_path=suite,
        suite_name="stable",
        track=BenchmarkTrack.DETERMINISTIC,
        repeat_count=2,
        seed=17,
        environment={"model": "demo", "api_key": "different"},
    )

    assert first.schema_version == BENCHMARK_SCHEMA_VERSION
    assert first.experiment_id == second.experiment_id
    assert first.environment == {"model": "demo"}
    assert "secret" not in first.model_dump_json()


def test_paired_mode_order_is_deterministic_and_balanced() -> None:
    orders = [paired_mode_order(seed=17, pair_index=index) for index in range(10)]

    assert orders == [paired_mode_order(seed=17, pair_index=index) for index in range(10)]
    assert orders.count(("text", "protocol")) == 5
    assert orders.count(("protocol", "text")) == 5
