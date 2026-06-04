from agentmesh.eval.compare import run_prompt_compare
from agentmesh.storage.paths import RuntimePaths


def test_prompt_compare_runs_both_modes_from_user_input(tmp_path) -> None:
    summary = run_prompt_compare(
        "Analyze a long-context multi-agent runtime and compare communication overhead.",
        paths=RuntimePaths(root=tmp_path),
    )

    assert summary.text.mode == "text"
    assert summary.protocol.mode == "protocol"
    assert summary.text.metrics.estimated_tokens > summary.protocol.metrics.estimated_tokens
    assert summary.text.metrics.wire_bytes > 0
    assert summary.protocol.metrics.wire_bytes > 0
    assert summary.task_path.endswith(".txt")
