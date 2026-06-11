from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentmesh.config import AgentMeshConfig
from agentmesh.errors import CapabilityNotFoundError
from agentmesh.llm.client import LLMClient, create_llm_client
from agentmesh.prompts.store import PromptTemplateStore
from agentmesh.runtime.agent import BaseAgent
from agentmesh.storage.paths import RuntimePaths


class RuntimeContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    paths: RuntimePaths
    task_path: Path | None = None
    trace_id: str
    config: AgentMeshConfig
    prompts: PromptTemplateStore
    llm_client: Any = None
    capability_to_agent: dict[str, str] = Field(default_factory=dict)
    state_store: Any = None
    embedding_encoder: Any = None
    memory_store: Any = None

    @classmethod
    def from_paths(
        cls,
        *,
        paths: RuntimePaths,
        trace_id: str,
        task_path: Path | None = None,
        llm_client: LLMClient | None = None,
        load_configured_llm: bool = True,
    ) -> "RuntimeContext":
        config = AgentMeshConfig.from_project_root(paths.root)
        prompts = PromptTemplateStore.from_project_root(paths.root, config.prompt_dir)
        configured_client = create_llm_client(config) if load_configured_llm else None
        return cls(
            paths=paths,
            task_path=task_path,
            trace_id=trace_id,
            config=config,
            prompts=prompts,
            llm_client=llm_client if llm_client is not None else configured_client,
        )


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.name] = agent

    def get(self, name: str) -> BaseAgent:
        return self._agents[name]

    def all(self) -> list[BaseAgent]:
        return list(self._agents.values())

    def require_capability(self, capability: str) -> BaseAgent:
        for agent in self._agents.values():
            if capability in agent.capabilities:
                return agent
        raise CapabilityNotFoundError(f"No registered agent provides {capability}")

    def capability_owner_map(self) -> dict[str, str]:
        owners: dict[str, str] = {}
        for agent in self._agents.values():
            for capability in agent.capabilities:
                owners.setdefault(capability, agent.name)
        return owners
