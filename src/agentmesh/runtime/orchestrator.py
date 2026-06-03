from agentmesh.agents import ExecutorAgent, PlannerAgent, RetrieverAgent, SummarizerAgent
from agentmesh.runtime.registry import AgentRegistry


def default_registry() -> AgentRegistry:
    registry = AgentRegistry()
    for agent in [PlannerAgent(), RetrieverAgent(), ExecutorAgent(), SummarizerAgent()]:
        registry.register(agent)
    return registry
