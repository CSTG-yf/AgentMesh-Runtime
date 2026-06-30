from pathlib import Path

from typer.testing import CliRunner


def _write_suite(root: Path, name: str) -> Path:
    suite = root / "runs" / "latest" / "benchmarks" / name
    suite.mkdir(parents=True)
    return suite


def test_load_dashboard_data_keeps_configured_and_generated_suites(tmp_path: Path) -> None:
    from agentmesh.eval.dashboard import load_dashboard_data

    configs = tmp_path / "examples" / "benchmarks"
    configs.mkdir(parents=True)
    (configs / "one.yaml").write_text(
        "name: configured_suite\nrepeat: 1\ntasks:\n"
        "  - id: C1\n    group: A\n    topic: configured\n",
        encoding="utf-8",
    )
    suite = _write_suite(tmp_path, "generated_suite")
    (suite / "benchmark_summary.csv").write_text(
        "suite_name,total_runs,token_saving_rate,missing_metric\n"
        "generated_suite,2,0.25,\n",
        encoding="utf-8",
    )
    (suite / "benchmark_detail.jsonl").write_text(
        '{"task_id":"A1","mode":"text","metrics":{"latency_ms":12}}\n'
        "not-json\n",
        encoding="utf-8",
    )

    data = load_dashboard_data(tmp_path)

    assert [item["name"] for item in data["suites"]] == [
        "configured_suite",
        "generated_suite",
    ]
    configured = data["suites"][0]
    assert configured["status"] == "not_generated"
    assert configured["summary"] == {}
    assert configured["configured_task_count"] == 1
    generated = data["suites"][1]
    assert generated["status"] == "partial"
    assert generated["summary"]["token_saving_rate"] == 0.25
    assert generated["summary"]["missing_metric"] is None
    assert generated["details"][0]["metrics"]["latency_ms"] == 12
    assert generated["diagnostics"][0]["kind"] == "malformed_jsonl"


def test_dashboard_keeps_deterministic_and_llm_tracks_separate(tmp_path: Path) -> None:
    from agentmesh.eval.dashboard import load_dashboard_data

    suite = _write_suite(tmp_path, "same_suite")
    for track in ["deterministic", "llm"]:
        track_dir = suite / track
        track_dir.mkdir()
        (track_dir / "benchmark_summary.csv").write_text(
            "schema_version,suite_name,track,total_runs\n"
            f"2.0,same_suite,{track},1\n",
            encoding="utf-8",
        )
        (track_dir / "benchmark_detail.jsonl").write_text(
            f'{{"schema_version":"2.0","track":"{track}","task_id":"T1"}}\n',
            encoding="utf-8",
        )
        (track_dir / "experiment_report.md").write_text(
            f"# {track} report\n",
            encoding="utf-8",
        )

    data = load_dashboard_data(tmp_path)
    identities = {
        (str(item["name"]), str(item["track"]))
        for item in data["suites"]
        if item["status"] == "complete"
    }

    assert identities == {
        ("same_suite", "deterministic"),
        ("same_suite", "llm"),
    }


def test_generate_dashboard_is_self_contained_complete_and_safe(tmp_path: Path) -> None:
    from agentmesh.eval.dashboard import generate_dashboard

    suite = _write_suite(tmp_path, "demo")
    (suite / "benchmark_summary.csv").write_text(
        "suite_name,total_runs,token_saving_rate,custom_metric\n"
        "demo,1,0.5,7\n",
        encoding="utf-8",
    )
    (suite / "benchmark_detail.jsonl").write_text(
        '{"task_id":"</script><b>x</b>","group":"A","mode":"protocol",'
        '"trace_id":"trace-1","metrics":{"latency_ms":10}}\n',
        encoding="utf-8",
    )
    (suite / "experiment_report.md").write_text(
        "# Evidence\nProtocol report body.", encoding="utf-8"
    )

    output = generate_dashboard(tmp_path)
    html = output.read_text(encoding="utf-8")

    assert output.name == "benchmark_dashboard.html"
    assert "AgentMesh Benchmark" in html
    assert "custom_metric" in html
    assert "No benchmark data" in html
    assert "</script><b>x</b>" not in html
    assert "<script src=" not in html
    assert "<link rel=" not in html
    assert "Task results" in html
    assert "All metrics" in html
    assert "Evidence &amp; diagnostics" in html


def test_generate_dashboard_without_artifacts_shows_safe_empty_state(tmp_path: Path) -> None:
    from agentmesh.eval.dashboard import generate_dashboard

    output = generate_dashboard(tmp_path)
    html = output.read_text(encoding="utf-8")

    assert output.exists()
    assert "No suites discovered" in html
    assert "N/A" in html
    assert "Not recorded" in html


def test_dashboard_command_generates_html(tmp_path: Path) -> None:
    from agentmesh.cli import app

    result = CliRunner().invoke(app, ["dashboard", "--root", str(tmp_path)])

    assert result.exit_code == 0
    assert (tmp_path / "runs/latest/benchmarks/benchmark_dashboard.html").exists()
