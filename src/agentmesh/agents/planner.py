from agentmesh.agents.state_refs import first_text_payload
from agentmesh.llm.client import ChatMessage
from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.schema import AMPMessage
from agentmesh.runtime.agent import BaseAgent
from agentmesh.runtime.decision import PlannerDecision
from agentmesh.runtime.registry import RuntimeContext


class PlannerAgent(BaseAgent):
    name = "planner"
    capabilities = ["plan.create", "plan.refine", "protocol.map", "memory.query"]

    def handle(self, message: AMPMessage, context: RuntimeContext) -> AMPMessage:
        task = str(message.params.get("task", ""))
        if not task:
            task = first_text_payload(
                context=context,
                state_refs=message.state_refs,
                consumer=self.name,
            )
        if message.action == "plan.refine":
            feedback = message.params.get("tool_feedback", {})
            refined_plan = [
                "review tool feedback",
                "address evidence gaps",
                "keep protocol state handoff compact",
                "send refined context to summarizer",
            ]
            return AMPMessage(
                trace_id=context.trace_id,
                source_agent=self.name,
                target_agent=message.source_agent,
                msg_type=MsgType.RESULT,
                action=message.action,
                result={
                    "refined_plan": refined_plan,
                    "changed": True,
                    "reason": "tool feedback requested plan.refine",
                    "tool_feedback": feedback,
                },
                state_refs=message.state_refs,
            )
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
        decision = PlannerDecision.from_llm_or_task(
            task,
            llm_plan,
            capability_to_agent=context.capability_to_agent,
        )
        steps = [
            "identify user intent",
            "select dynamic agent route",
            "retrieve reusable memory" if decision.need_retrieval else "skip retrieval",
            "execute tool step" if decision.need_tool_execution else "skip tool execution",
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
                "decision": decision.model_dump(),
            },
            state_refs=message.state_refs,
        )
