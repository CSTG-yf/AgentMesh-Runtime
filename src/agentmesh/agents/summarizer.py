from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.schema import AMPMessage
from agentmesh.runtime.agent import BaseAgent
from agentmesh.runtime.registry import RuntimeContext


class SummarizerAgent(BaseAgent):
    name = "summarizer"
    capabilities = ["summary.create", "memory.put", "report.fragment"]

    def handle(self, message: AMPMessage, context: RuntimeContext) -> AMPMessage:
        return AMPMessage(
            trace_id=context.trace_id,
            source_agent=self.name,
            target_agent=message.source_agent,
            msg_type=MsgType.RESULT,
            action=message.action,
            result={"summary": "AgentMesh Runtime completed a deterministic collaboration step."},
            state_refs=message.state_refs,
        )
