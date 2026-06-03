from agentmesh.llm.client import ChatMessage
from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.schema import AMPMessage
from agentmesh.runtime.agent import BaseAgent
from agentmesh.runtime.registry import RuntimeContext


class ExecutorAgent(BaseAgent):
    name = "executor"
    capabilities = ["tool.run_python", "tool.validate_result", "state.consume"]

    def handle(self, message: AMPMessage, context: RuntimeContext) -> AMPMessage:
        llm_validation: str | None = None
        if context.llm_client is not None:
            try:
                llm_validation = context.llm_client.complete(
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
                llm_validation = None
        return AMPMessage(
            trace_id=context.trace_id,
            source_agent=self.name,
            target_agent=message.source_agent,
            msg_type=MsgType.RESULT,
            action=message.action,
            result={
                "validated": True,
                "state_refs_consumed": message.state_refs,
                "llm_validation": llm_validation,
            },
            state_refs=message.state_refs,
        )
