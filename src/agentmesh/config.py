from os import environ
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field, SecretStr, field_serializer


class LLMConfig(BaseModel):
    base_url: str | None = None
    api_key: SecretStr | None = None
    model: str | None = None
    timeout_seconds: float = 30.0
    text_timeout_seconds: float | None = None

    @field_serializer("api_key", when_used="json")
    def serialize_api_key(self, value: SecretStr | None) -> str | None:
        if value is None:
            return None
        return "**********"

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)


class EmbeddingConfig(BaseModel):
    provider: str = "hash"
    base_url: str | None = None
    model: str = "BAAI/bge-small-zh-v1.5"
    dimensions: int = 512
    timeout_seconds: float = 15.0

    @property
    def tei_configured(self) -> bool:
        return self.provider == "tei" and bool(self.base_url)


class MemoryConfig(BaseModel):
    maintenance_enabled: bool = True
    maintenance_interval_seconds: float = 60.0
    maintenance_max_items: int = 20


class StateConfig(BaseModel):
    payload_backend: Literal["file", "shm"] = "file"
    shm_threshold_bytes: int = 4096

    @property
    def shm_enabled(self) -> bool:
        return self.payload_backend == "shm"


class ProtocolConfig(BaseModel):
    skip_handshake_for_inproc: bool = True
    state_summary_max_chars: int = 800
    evidence_snippet_max_chars: int = 160
    agent_log_output_max_chars: int = 1200
    agent_log_param_max_chars: int = 500
    memory_reuse_default_limit: int = 1


class AgentMeshConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)
    prompt_dir: Path | None = None

    @classmethod
    def from_env_file(cls, path: Path) -> "AgentMeshConfig":
        values = dict(environ)
        values.update(_read_dotenv(path))
        return cls.from_mapping(values)

    @classmethod
    def from_project_root(cls, root: Path) -> "AgentMeshConfig":
        return cls.from_env_file(root / ".env")

    @classmethod
    def from_mapping(cls, values: dict[str, str]) -> "AgentMeshConfig":
        api_key = _first_present(values, "AGENTMESH_LLM_API_KEY", "OPENAI_API_KEY")
        timeout_raw = _first_present(values, "AGENTMESH_LLM_TIMEOUT_SECONDS")
        text_timeout_raw = _first_present(values, "AGENTMESH_TEXT_LLM_TIMEOUT_SECONDS")
        embedding_provider = _first_present(values, "AGENTMESH_EMBEDDING_PROVIDER") or "hash"
        embedding_dimensions_raw = _first_present(values, "AGENTMESH_EMBEDDING_DIMENSIONS")
        embedding_timeout_raw = _first_present(values, "AGENTMESH_EMBEDDING_TIMEOUT_SECONDS")
        maintenance_enabled_raw = _first_present(
            values,
            "AGENTMESH_MEMORY_MAINTENANCE_ENABLED",
        )
        maintenance_interval_raw = _first_present(
            values,
            "AGENTMESH_MEMORY_MAINTENANCE_INTERVAL_SECONDS",
        )
        maintenance_max_items_raw = _first_present(
            values,
            "AGENTMESH_MEMORY_MAX_BACKGROUND_ITEMS",
        )
        state_payload_backend = cast(
            Literal["file", "shm"],
            _first_present(values, "AGENTMESH_STATE_PAYLOAD_BACKEND") or "file",
        )
        state_shm_threshold_raw = _first_present(values, "AGENTMESH_STATE_SHM_THRESHOLD_BYTES")
        skip_handshake_raw = _first_present(
            values,
            "AGENTMESH_PROTOCOL_SKIP_HANDSHAKE_FOR_INPROC",
        )
        state_summary_max_raw = _first_present(values, "AGENTMESH_STATE_SUMMARY_MAX_CHARS")
        evidence_snippet_max_raw = _first_present(
            values,
            "AGENTMESH_EVIDENCE_SNIPPET_MAX_CHARS",
        )
        agent_log_output_max_raw = _first_present(
            values,
            "AGENTMESH_AGENT_LOG_OUTPUT_MAX_CHARS",
        )
        agent_log_param_max_raw = _first_present(
            values,
            "AGENTMESH_AGENT_LOG_PARAM_MAX_CHARS",
        )
        memory_reuse_default_limit_raw = _first_present(
            values,
            "AGENTMESH_MEMORY_REUSE_DEFAULT_LIMIT",
        )
        prompt_dir_raw = _first_present(values, "AGENTMESH_PROMPT_DIR")
        timeout = 30.0
        if timeout_raw:
            timeout = float(timeout_raw)
        text_timeout = float(text_timeout_raw) if text_timeout_raw else None
        embedding_dimensions = 512
        if embedding_dimensions_raw:
            embedding_dimensions = int(embedding_dimensions_raw)
        embedding_timeout = 15.0
        if embedding_timeout_raw:
            embedding_timeout = float(embedding_timeout_raw)
        maintenance_interval = 60.0
        if maintenance_interval_raw:
            maintenance_interval = float(maintenance_interval_raw)
        maintenance_max_items = 20
        if maintenance_max_items_raw:
            maintenance_max_items = int(maintenance_max_items_raw)
        state_shm_threshold = 4096
        if state_shm_threshold_raw:
            state_shm_threshold = int(state_shm_threshold_raw)
        state_summary_max = _positive_int(state_summary_max_raw, 800)
        evidence_snippet_max = _positive_int(evidence_snippet_max_raw, 160)
        agent_log_output_max = _positive_int(agent_log_output_max_raw, 1200)
        agent_log_param_max = _positive_int(agent_log_param_max_raw, 500)
        memory_reuse_default_limit = _positive_int(memory_reuse_default_limit_raw, 1)
        return cls(
            llm=LLMConfig(
                base_url=_first_present(values, "AGENTMESH_LLM_BASE_URL", "OPENAI_BASE_URL"),
                api_key=SecretStr(api_key) if api_key else None,
                model=_first_present(values, "AGENTMESH_LLM_MODEL", "OPENAI_MODEL"),
                timeout_seconds=timeout,
                text_timeout_seconds=text_timeout,
            ),
            embedding=EmbeddingConfig(
                provider=embedding_provider,
                base_url=_first_present(values, "AGENTMESH_EMBEDDING_BASE_URL"),
                model=_first_present(values, "AGENTMESH_EMBEDDING_MODEL")
                or "BAAI/bge-small-zh-v1.5",
                dimensions=embedding_dimensions,
                timeout_seconds=embedding_timeout,
            ),
            memory=MemoryConfig(
                maintenance_enabled=_parse_bool(maintenance_enabled_raw, default=True),
                maintenance_interval_seconds=maintenance_interval,
                maintenance_max_items=maintenance_max_items,
            ),
            state=StateConfig(
                payload_backend=state_payload_backend,
                shm_threshold_bytes=state_shm_threshold,
            ),
            protocol=ProtocolConfig(
                skip_handshake_for_inproc=_parse_bool(skip_handshake_raw, default=True),
                state_summary_max_chars=state_summary_max,
                evidence_snippet_max_chars=evidence_snippet_max,
                agent_log_output_max_chars=agent_log_output_max,
                agent_log_param_max_chars=agent_log_param_max,
                memory_reuse_default_limit=memory_reuse_default_limit,
            ),
            prompt_dir=Path(prompt_dir_raw) if prompt_dir_raw else None,
        )


def _first_present(values: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = values.get(key)
        if value:
            return value
    return None


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            parsed[key] = value
    return parsed


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _positive_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    parsed = int(value)
    return parsed if parsed > 0 else default
