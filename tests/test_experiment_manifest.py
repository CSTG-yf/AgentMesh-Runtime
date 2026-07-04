import hashlib
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
        environment={
            "model": "demo",
            "api_key": "secret",
            "LLM_API_KEY": "secret",
            "access_token": "secret",
            "client_secret_value": "secret",
            "database_password_file": "secret",
            "Authorization_Header": "secret",
            "ordinary_key": "visible",
        },
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
    assert first.environment == {"model": "demo", "ordinary_key": "visible"}
    assert "secret" not in first.model_dump_json()


def test_model_fingerprint_uses_canonical_url_and_model() -> None:
    base_url = "https://example.test/v1"
    model = "test-model"
    expected = hashlib.sha256(f"{base_url}|{model}".encode()).hexdigest()[:16]

    fingerprint = model_fingerprint(f"{base_url}/", model)

    assert fingerprint == expected
    assert fingerprint == model_fingerprint(base_url, model)
    assert len(fingerprint) == 16


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


def test_prompt_tree_hash_uses_sorted_markdown_names_and_content(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_bytes(b"two")
    (tmp_path / "a.md").write_bytes(b"one")
    expected = hashlib.sha256(b"a.md\0one\0b.md\0two\0").hexdigest()

    assert prompt_tree_sha256(tmp_path) == expected

    original = prompt_tree_sha256(tmp_path)
    (tmp_path / "notes.txt").write_text("not a prompt", encoding="utf-8")
    assert prompt_tree_sha256(tmp_path) == original

    (tmp_path / "a.md").write_text("changed", encoding="utf-8")
    assert prompt_tree_sha256(tmp_path) != original
