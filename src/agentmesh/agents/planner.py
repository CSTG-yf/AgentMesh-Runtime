from agentmesh.llm.client import ChatMessage
from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.schema import AMPMessage
from agentmesh.runtime.agent import BaseAgent
from agentmesh.runtime.registry import RuntimeContext


class PlannerAgent(BaseAgent):
    name = "planner"
    capabilities = ["plan.create", "protocol.map", "memory.query"]

    def handle(self, message: AMPMessage, context: RuntimeContext) -> AMPMessage:
        task = str(message.params.get("task", ""))
        llm_plan: str | None = None
        if context.llm_client is not None:
            try:
                llm_plan = context.llm_client.complete(
                    agent_name=self.name,
                    messages=[
                        ChatMessage(
                            role="system",
                            content=context.prompts.render(self.name, {"task": task}),
                        ),
                        ChatMessage(role="user", content=task),
                    ],
                    variables={"task": task},
                )
            except Exception:
                llm_plan = None
        steps = [
            "identify task topic",
            "retrieve reusable memory",
            "execute deterministic tool step",
            "summarize and persist memory",
        ]
        return AMPMessage(
            trace_id=context.trace_id,
            source_agent=self.name,
            target_agent=message.source_agent,
            msg_type=MsgType.RESULT,
            action=message.action,
            result={
                "plan": steps,
                "topic": task[:80],
                "llm_configured": context.config.llm.configured,
                "llm_plan": llm_plan,
            },
            state_refs=message.state_refs,
        )
