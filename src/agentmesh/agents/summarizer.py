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

        # Use structured params from orchestrator first — avoids reading all 6+ state refs
        task = str(message.params.get("task", "") or "")
        evidence_digest = str(message.params.get("evidence_digest", "") or "")
        code_result = str(message.params.get("code_result", "") or "")
        disable_state_fallback = message.params.get("_disable_state_fallback") is True

        if not code_result and not disable_state_fallback:
            code_result = first_code_result_text(
                context=context,
                state_refs=message.state_refs,
                consumer=self.name,
            )

        # Assemble compact input from structured params; fall back to state_payloads
        # only when no param was provided by the orchestrator.
        if task or evidence_digest or code_result:
            input_parts = []
            if task:
                input_parts.append(f"Task:\n{task}")
            if evidence_digest:
                input_parts.append(f"Evidence:\n{evidence_digest}")
            if code_result:
                input_parts.append(f"CodeAct result:\n{code_result}")
            input_text = "\n\n".join(input_parts)
        elif not disable_state_fallback:
            input_text = state_payloads_as_text(
                context=context,
                state_refs=message.state_refs,
                consumer=self.name,
            )
            if code_result:
                input_text = f"{input_text}\n\nCodeAct result:\n{code_result}"
        else:
            input_text = ""
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
