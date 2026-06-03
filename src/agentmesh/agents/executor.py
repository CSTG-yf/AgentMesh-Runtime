from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.schema import AMPMessage
from agentmesh.runtime.agent import BaseAgent
from agentmesh.runtime.registry import RuntimeContext


class ExecutorAgent(BaseAgent):
    name = "executor"
    capabilities = ["tool.run_python", "tool.validate_result", "state.consume"]

    def handle(self, message: AMPMessage, context: RuntimeContext) -> AMPMessage:
        return AMPMessage(
            trace_id=context.trace_id,
            source_agent=self.name,
            target_agent=message.source_agent,
            msg_type=MsgType.RESULT,
            action=message.action,
            result={"validated": True, "state_refs_consumed": message.state_refs},
            state_refs=message.state_refs,
        )
