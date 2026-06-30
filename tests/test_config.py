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
                "AGENTMESH_TEXT_LLM_TIMEOUT_SECONDS=90",
            ]
        ),
        encoding="utf-8",
    )

    config = AgentMeshConfig.from_env_file(env_file)

    assert config.llm.base_url == "https://llm.example/v1"
    assert config.llm.api_key.get_secret_value() == "sk-test-secret"
    assert config.llm.model == "agentmesh-test-model"
    assert config.llm.timeout_seconds == 15.0
    assert config.llm.text_timeout_seconds == 90.0
    assert config.model_dump(mode="json")["llm"]["api_key"] == "**********"


def test_config_loads_tei_embedding_settings(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "AGENTMESH_EMBEDDING_PROVIDER=tei",
                "AGENTMESH_EMBEDDING_BASE_URL=http://127.0.0.1:8080",
                "AGENTMESH_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5",
                "AGENTMESH_EMBEDDING_DIMENSIONS=512",
                "AGENTMESH_EMBEDDING_TIMEOUT_SECONDS=9",
            ]
        ),
        encoding="utf-8",
    )

    config = AgentMeshConfig.from_env_file(env_file)

    assert config.embedding.provider == "tei"
    assert config.embedding.base_url == "http://127.0.0.1:8080"
    assert config.embedding.model == "BAAI/bge-small-zh-v1.5"
    assert config.embedding.dimensions == 512
    assert config.embedding.timeout_seconds == 9.0
    assert config.embedding.tei_configured


def test_config_loads_memory_maintenance_settings(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "AGENTMESH_MEMORY_MAINTENANCE_ENABLED=false",
                "AGENTMESH_MEMORY_MAINTENANCE_INTERVAL_SECONDS=12",
                "AGENTMESH_MEMORY_MAX_BACKGROUND_ITEMS=7",
            ]
        ),
        encoding="utf-8",
    )

    config = AgentMeshConfig.from_env_file(env_file)

    assert not config.memory.maintenance_enabled
    assert config.memory.maintenance_interval_seconds == 12.0
    assert config.memory.maintenance_max_items == 7


def test_config_loads_state_payload_backend_settings(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "AGENTMESH_STATE_PAYLOAD_BACKEND=shm",
                "AGENTMESH_STATE_SHM_THRESHOLD_BYTES=128",
            ]
        ),
        encoding="utf-8",
    )

    config = AgentMeshConfig.from_env_file(env_file)

    assert config.state.payload_backend == "shm"
    assert config.state.shm_threshold_bytes == 128
    assert config.state.shm_enabled


def test_config_loads_protocol_budget_settings(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "AGENTMESH_STATE_SUMMARY_MAX_CHARS=640",
                "AGENTMESH_EVIDENCE_SNIPPET_MAX_CHARS=120",
                "AGENTMESH_AGENT_LOG_OUTPUT_MAX_CHARS=900",
                "AGENTMESH_AGENT_LOG_PARAM_MAX_CHARS=320",
                "AGENTMESH_MEMORY_REUSE_DEFAULT_LIMIT=2",
            ]
        ),
        encoding="utf-8",
    )

    config = AgentMeshConfig.from_env_file(env_file)

    assert config.protocol.state_summary_max_chars == 640
    assert config.protocol.evidence_snippet_max_chars == 120
    assert config.protocol.agent_log_output_max_chars == 900
    assert config.protocol.agent_log_param_max_chars == 320
    assert config.protocol.memory_reuse_default_limit == 2


def test_protocol_handshake_is_skipped_for_inproc_by_default() -> None:
    config = AgentMeshConfig.from_mapping({})

    assert config.protocol.skip_handshake_for_inproc is True


def test_protocol_handshake_can_be_enabled_explicitly() -> None:
    config = AgentMeshConfig.from_mapping(
        {"AGENTMESH_PROTOCOL_SKIP_HANDSHAKE_FOR_INPROC": "false"}
    )

    assert config.protocol.skip_handshake_for_inproc is False


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
