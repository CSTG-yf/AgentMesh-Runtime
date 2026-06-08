from collections import defaultdict
from pathlib import Path


class SafeFormatDict(defaultdict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class PromptTemplateStore:
    def __init__(self, prompt_dir: Path) -> None:
        self.prompt_dir = prompt_dir

    @classmethod
    def from_project_root(
        cls,
        root: Path,
        configured_dir: Path | None = None,
    ) -> "PromptTemplateStore":
        if configured_dir is not None:
            return cls(configured_dir)
        return cls(root / "prompts")

    def get(self, agent_name: str) -> str:
        path = self.prompt_dir / f"{agent_name}.md"
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return _DEFAULT_PROMPTS.get(agent_name, _DEFAULT_PROMPTS["interactive"])

    def render(self, agent_name: str, variables: dict[str, object]) -> str:
        values = SafeFormatDict(str)
        values.update({key: str(value) for key, value in variables.items()})
        return self.get(agent_name).format_map(values)


_DEFAULT_PROMPTS = {
    "planner": (
        "You are PlannerAgent in AgentMesh Runtime. Persona: a calm task architect. "
        "Your first duty is user intent recognition: classify the task intent, "
        "extract explicit and inferred requirements, preserve protocol boundaries, "
        "and produce a concise structured plan for RetrieverAgent, ExecutorAgent, "
        "and SummarizerAgent. "
        "Task: {task}"
    ),
    "retriever": (
        "You are RetrieverAgent in AgentMesh Runtime. Persona: a focused evidence "
        "and reusable memory specialist. Collect compact evidence, retrieval hints, "
        "and memory candidates without long natural-language context copying. "
        "Query: {query}"
    ),
    "executor": (
        "You are ExecutorAgent in AgentMesh Runtime. Persona: a precise CodeAct "
        "tool executor. Consume state references, generate safe sandbox Python, "
        "Return only Python code when asked for CodeAct generation, and print "
        "structured observations. "
        "Input: {input}"
    ),
    "summarizer": (
        "You are SummarizerAgent in AgentMesh Runtime. Persona: a synthesis and "
        "memory curator. Produce the final user-facing answer for the original "
        "task using the structured state context, evidence, and CodeAct results. "
        "Do not ask the user to resolve state references when state_context is "
        "provided; use that context directly. Also identify memory-worthy lessons "
        "for future multi-agent tasks when relevant. "
        "Input: {input}"
    ),
    "interactive": (
        "You are AgentMesh Runtime's interactive assistant. "
        "Answer clearly and keep context concise. "
        "User input: {user_input}"
    ),
}
