from pathlib import Path

from agentmesh.config import AgentMeshConfig
from agentmesh.modes.protocol_mode import run_protocol_mode
from agentmesh.storage.paths import RuntimePaths


def test_config_loads_llm_settings_from_dotenv_without_exporting_secret(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "AGENTMESH_LLM_BASE_URL=https://llm.example/v1",
                "AGENTMESH_LLM_API_KEY=sk-test-secret",
                "AGENTMESH_LLM_MODEL=agentmesh-test-model",
                "AGENTMESH_LLM_TIMEOUT_SECONDS=15",
            ]
        ),
        encoding="utf-8",
    )

    config = AgentMeshConfig.from_env_file(env_file)

    assert config.llm.base_url == "https://llm.example/v1"
    assert config.llm.api_key.get_secret_value() == "sk-test-secret"
    assert config.llm.model == "agentmesh-test-model"
    assert config.llm.timeout_seconds == 15.0
    assert config.model_dump(mode="json")["llm"]["api_key"] == "**********"


def test_protocol_mode_receives_optional_llm_config_from_env(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "AGENTMESH_LLM_BASE_URL=https://llm.example/v1",
                "AGENTMESH_LLM_API_KEY=sk-test-secret",
                "AGENTMESH_LLM_MODEL=agentmesh-test-model",
            ]
        ),
        encoding="utf-8",
    )
    task = tmp_path / "task.txt"
    task.write_text("Use optional LLM config without requiring network access.", encoding="utf-8")

    result = run_protocol_mode(task_path=task, paths=RuntimePaths(root=tmp_path))

    assert result.mode == "protocol"
    assert result.metrics.message_count > 0
