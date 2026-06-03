from agentmesh.llm.client import ChatMessage
from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.schema import AMPMessage
from agentmesh.runtime.agent import BaseAgent
from agentmesh.runtime.registry import RuntimeContext


class SummarizerAgent(BaseAgent):
    name = "summarizer"
    capabilities = ["summary.create", "memory.put", "report.fragment"]

    def handle(self, message: AMPMessage, context: RuntimeContext) -> AMPMessage:
        summary = "AgentMesh Runtime completed a deterministic collaboration step."
        if context.llm_client is not None:
            try:
                summary = context.llm_client.complete(
                    agent_name=self.name,
                    messages=[
                        ChatMessage(
                            role="system",
                            content=context.prompts.render(
                                self.name,
                                {"input": ", ".join(message.state_refs)},
                            ),
                        ),
                        ChatMessage(role="user", content=", ".join(message.state_refs)),
                    ],
                    variables={"input": ", ".join(message.state_refs)},
                )
            except Exception:
                pass
        return AMPMessage(
            trace_id=context.trace_id,
            source_agent=self.name,
            target_agent=message.source_agent,
            msg_type=MsgType.RESULT,
            action=message.action,
            result={"summary": summary},
            state_refs=message.state_refs,
        )
