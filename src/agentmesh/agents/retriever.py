from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.schema import AMPMessage
from agentmesh.runtime.agent import BaseAgent
from agentmesh.runtime.registry import RuntimeContext


class RetrieverAgent(BaseAgent):
    name = "retriever"
    capabilities = [
        "memory.keyword_search",
        "memory.tag_search",
        "memory.semantic_search",
        "evidence.collect",
    ]

    def handle(self, message: AMPMessage, context: RuntimeContext) -> AMPMessage:
        query = str(message.params.get("query", ""))
        return AMPMessage(
            trace_id=context.trace_id,
            source_agent=self.name,
            target_agent=message.source_agent,
            msg_type=MsgType.RESULT,
            action=message.action,
            result={
                "evidence": [
                    {
                        "title": "deterministic-local-evidence",
                        "snippet": f"Evidence for {query[:80]}",
                    }
                ]
            },
            state_refs=message.state_refs,
        )
