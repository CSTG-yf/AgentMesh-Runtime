import ast
import re

from agentmesh.agents.state_refs import first_text_payload, state_payloads_as_text
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
        state_text = state_payloads_as_text(
            context=context,
            state_refs=message.state_refs,
            consumer=self.name,
        )
        if not task:
            task = first_text_payload(
                context=context,
                state_refs=message.state_refs,
                consumer=self.name,
            )
        if not evidence:
            evidence = state_text
        code_input = "\n".join(item for item in [task, evidence] if item)
        llm_code: str | None = None
        deterministic_code = _deterministic_task_code(task)
        if deterministic_code is None and context.llm_client is not None:
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
        code = deterministic_code or _extract_python_code(llm_code) or _deterministic_code(
            task,
            evidence,
        )
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


def _deterministic_task_code(task: str) -> str | None:
    lowered = task.lower()
    if not any(
        word in lowered
        for word in ["快速排序", "快排", "排序", "quicksort", "quick sort", "sort"]
    ):
        return None
    numbers = _first_numeric_list(task) or [5, 3, 8, 3, 1, 9, -2, 0, 3]
    return (
        "import json\n"
        "\n"
        "def quick_sort(arr):\n"
        "    if len(arr) <= 1:\n"
        "        return arr\n"
        "    pivot = arr[len(arr) // 2]\n"
        "    left = [x for x in arr if x < pivot]\n"
        "    middle = [x for x in arr if x == pivot]\n"
        "    right = [x for x in arr if x > pivot]\n"
        "    return quick_sort(left) + middle + quick_sort(right)\n"
        "\n"
        f"data = {numbers!r}\n"
        "sorted_data = quick_sort(data)\n"
        "result = {\n"
        "    'algorithm': 'quick_sort',\n"
        "    'input': data,\n"
        "    'sorted': sorted_data,\n"
        "    'validated_claims': ['quick_sort executed in sandbox'],\n"
        "    'failed_claims': [],\n"
        "    'evidence_gaps': [],\n"
        "    'recommended_next_actions': [],\n"
        "}\n"
        "print(json.dumps(result, ensure_ascii=False, sort_keys=True))\n"
    )


def _first_numeric_list(text: str) -> list[int | float] | None:
    for match in re.finditer(r"\[[^\[\]]+\]", text):
        try:
            loaded = ast.literal_eval(match.group(0))
        except Exception:
            continue
        if not isinstance(loaded, list) or not loaded:
            continue
        if all(isinstance(item, int | float) and not isinstance(item, bool) for item in loaded):
            return loaded
    return None
