from agentmesh.llm.client import ChatMessage
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
        llm_evidence: str | None = None
        if context.llm_client is not None:
            try:
                llm_evidence = context.llm_client.complete(
                    agent_name=self.name,
                    messages=[
                        ChatMessage(
                            role="system",
                            content=context.prompts.render(self.name, {"query": query}),
                        ),
                        ChatMessage(role="user", content=query),
                    ],
                    variables={"query": query},
                )
            except Exception:
                llm_evidence = None
        evidence = [
            {
                "title": "deterministic-local-evidence",
                "snippet": f"Evidence for {query[:80]}",
            }
        ]
        if llm_evidence:
            evidence.insert(0, {"title": "llm-evidence", "snippet": llm_evidence})
        return AMPMessage(
            trace_id=context.trace_id,
            source_agent=self.name,
            target_agent=message.source_agent,
            msg_type=MsgType.RESULT,
            action=message.action,
            result={"evidence": evidence},
            state_refs=message.state_refs,
        )
