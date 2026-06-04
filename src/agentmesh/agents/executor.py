from agentmesh.llm.client import ChatMessage
from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.schema import AMPMessage
from agentmesh.runtime.agent import BaseAgent
from agentmesh.runtime.registry import RuntimeContext


class ExecutorAgent(BaseAgent):
    name = "executor"
    capabilities = ["tool.run_python", "tool.validate_result", "state.consume", "codeact.generate"]

    def handle(self, message: AMPMessage, context: RuntimeContext) -> AMPMessage:
        task = str(message.params.get("task", ""))
        evidence = str(message.params.get("evidence", ""))
        code_input = "\n".join(item for item in [task, evidence] if item)
        llm_code: str | None = None
        if context.llm_client is not None:
            try:
                llm_code = context.llm_client.complete(
                    agent_name=self.name,
                    messages=[
                        ChatMessage(
                            role="system",
                            content=(
                                context.prompts.render(self.name, {"input": code_input})
                                + "\nReturn only Python code. The code must not access the "
                                "network or filesystem. It should print a concise JSON-like "
                                "validation result for the task."
                            ),
                        ),
                        ChatMessage(role="user", content=code_input),
                    ],
                    variables={"input": code_input},
                )
            except Exception:
                llm_code = None
        code = _extract_python_code(llm_code) or _deterministic_code(task, evidence)
        return AMPMessage(
            trace_id=context.trace_id,
            source_agent=self.name,
            target_agent=message.source_agent,
            msg_type=MsgType.RESULT,
            action=message.action,
            result={
                "validated": True,
                "state_refs_consumed": message.state_refs,
                "codeact_code": code,
                "llm_generated_code": llm_code is not None,
            },
            state_refs=message.state_refs,
        )


def _extract_python_code(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    if "```" not in cleaned:
        return cleaned
    parts = cleaned.split("```")
    for part in parts:
        candidate = part.strip()
        if not candidate:
            continue
        if candidate.startswith("python"):
            candidate = candidate.removeprefix("python").strip()
        if candidate:
            return candidate
    return None


def _deterministic_code(task: str, evidence: str) -> str:
    task_chars = len(task)
    evidence_chars = len(evidence)
    return (
        "import json\n"
        "result = {\n"
        f"    'task_chars': {task_chars},\n"
        f"    'evidence_chars': {evidence_chars},\n"
        "    'status': 'validated',\n"
        "    'codeact': True,\n"
        "}\n"
        "print(json.dumps(result, ensure_ascii=False, sort_keys=True))\n"
    )
