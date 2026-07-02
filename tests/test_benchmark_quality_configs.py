from pathlib import Path

import yaml

from agentmesh.eval.quality_spec import QualitySpec


def test_all_contest_benchmark_tasks_have_valid_quality_rules() -> None:
    for path in Path("examples/benchmarks").glob("*.yaml"):
        suite = yaml.safe_load(path.read_text(encoding="utf-8"))
        for task in suite["tasks"]:
            spec = QualitySpec.model_validate(task["quality"])
            assert spec.rule_id.startswith(f"{task['id'].lower()}-")
