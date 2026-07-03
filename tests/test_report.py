from pathlib import Path

from agentmesh.eval.report import generate_report
from agentmesh.storage.paths import RuntimePaths


def test_report_uses_variant_directory(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    summary = paths.benchmark_suite_summary("suite", "llm", "baseline")
    summary.parent.mkdir(parents=True)
    summary.write_text(
        "schema_version,track,experiment_id\n2.0,llm,exp-test\n",
        encoding="utf-8",
    )
    report = generate_report(
        paths, suite_name="suite", track="llm", variant="baseline"
    )
    assert report.parent == summary.parent
    assert report.exists()
