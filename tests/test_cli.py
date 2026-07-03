from pathlib import Path

from typer.testing import CliRunner

from agentmesh.cli import app


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
    assert '"api_key": "configured"' in result.stdout
