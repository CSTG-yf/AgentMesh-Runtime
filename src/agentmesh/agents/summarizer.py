import orjson

from agentmesh.agents.state_refs import first_code_result_text, state_payloads_as_text
from agentmesh.llm.client import ChatMessage
from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.schema import AMPMessage
from agentmesh.runtime.agent import BaseAgent
from agentmesh.runtime.registry import RuntimeContext


class SummarizerAgent(BaseAgent):
    name = "summarizer"
    capabilities = ["summary.create", "memory.put", "report.fragment"]

    def handle(self, message: AMPMessage, context: RuntimeContext) -> AMPMessage:
        llm_summary: str | None = None
        code_result = str(message.params.get("code_result", ""))
        if not code_result:
            code_result = first_code_result_text(
                context=context,
                state_refs=message.state_refs,
                consumer=self.name,
            )
        state_context = message.params.get("state_context")
        input_text = _state_context_text(state_context) or state_payloads_as_text(
            context=context,
            state_refs=message.state_refs,
            consumer=self.name,
        )
        if code_result:
            input_text = f"{input_text}\n\nCodeAct result:\n{code_result}"
        summary = (
            "AgentMesh Runtime completed a deterministic collaboration step. "
            f"{code_result}".strip()
        )
        if context.llm_client is not None:
            try:
                llm_summary = context.llm_client.complete(
                    agent_name=self.name,
                    messages=[
                        ChatMessage(
                            role="system",
                            content=context.prompts.render(
                                self.name,
                                {"input": input_text},
                            ),
                        ),
                        ChatMessage(role="user", content=input_text),
                    ],
                    variables={"input": input_text},
                )
                summary = llm_summary
            except Exception:
                pass
        return AMPMessage(
            trace_id=context.trace_id,
            source_agent=self.name,
            target_agent=message.source_agent,
            msg_type=MsgType.RESULT,
            action=message.action,
            result={"summary": summary, "llm_summary": llm_summary},
            state_refs=message.state_refs,
        )


def _state_context_text(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    return orjson.dumps(value, option=orjson.OPT_INDENT_2).decode("utf-8")
