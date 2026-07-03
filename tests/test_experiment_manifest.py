from pathlib import Path

from agentmesh.eval.experiment import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkTrack,
    build_experiment_manifest,
    model_fingerprint,
    paired_mode_order,
    prompt_tree_sha256,
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


def test_profile_fields_are_part_of_manifest_identity(tmp_path: Path) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text("tasks: []", encoding="utf-8")
    common = dict(
        suite_path=suite,
        suite_name="suite",
        track=BenchmarkTrack.LLM,
        repeat_count=3,
        seed=1,
        profile_id="p",
        profile_sha256="a" * 64,
        prompt_version="p2",
        route_policy_version="legacy",
        model_fingerprint=model_fingerprint("https://example.test/v1", "model"),
        prompt_tree_sha256="b" * 64,
        quality_rules_sha256="c" * 64,
    )
    first = build_experiment_manifest(**common)
    second = build_experiment_manifest(**(common | {"prompt_version": "p3"}))
    assert first.experiment_id != second.experiment_id
    assert "example.test" not in first.model_dump_json()


def test_prompt_tree_hash_is_stable(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("one", encoding="utf-8")
    first = prompt_tree_sha256(tmp_path)
    (tmp_path / "b.md").write_text("two", encoding="utf-8")
    assert first != prompt_tree_sha256(tmp_path)
