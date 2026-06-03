from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agentmesh.config import AgentMeshConfig
from agentmesh.errors import CapabilityNotFoundError
from agentmesh.runtime.agent import BaseAgent
from agentmesh.storage.paths import RuntimePaths


class RuntimeContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    paths: RuntimePaths
    task_path: Path | None = None
    trace_id: str
    config: AgentMeshConfig


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
