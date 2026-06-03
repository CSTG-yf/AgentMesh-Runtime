from os import environ
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, field_serializer


class LLMConfig(BaseModel):
    base_url: str | None = None
    api_key: SecretStr | None = None
    model: str | None = None
    timeout_seconds: float = 30.0

    @field_serializer("api_key", when_used="json")
    def serialize_api_key(self, value: SecretStr | None) -> str | None:
        if value is None:
            return None
        return "**********"

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)


class AgentMeshConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
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
        prompt_dir_raw = _first_present(values, "AGENTMESH_PROMPT_DIR")
        timeout = 30.0
        if timeout_raw:
            timeout = float(timeout_raw)
        return cls(
            llm=LLMConfig(
                base_url=_first_present(values, "AGENTMESH_LLM_BASE_URL", "OPENAI_BASE_URL"),
                api_key=SecretStr(api_key) if api_key else None,
                model=_first_present(values, "AGENTMESH_LLM_MODEL", "OPENAI_MODEL"),
                timeout_seconds=timeout,
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
