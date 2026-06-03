from pydantic import BaseModel

from agentmesh.llm.client import ChatMessage
from agentmesh.runtime.registry import RuntimeContext


class ChatTurn(BaseModel):
    user: str
    assistant: str


def run_chat_turn(
    user_input: str,
    *,
    context: RuntimeContext,
    history: list[ChatTurn],
) -> ChatTurn:
    system_prompt = context.prompts.render("interactive", {"user_input": user_input})
    messages = [ChatMessage(role="system", content=system_prompt)]
    for turn in history:
        messages.extend(
            [
                ChatMessage(role="user", content=turn.user),
                ChatMessage(role="assistant", content=turn.assistant),
            ]
        )
    messages.append(ChatMessage(role="user", content=user_input))
    if context.llm_client is None:
        answer = (
            "LLM is not configured. Add AGENTMESH_LLM_BASE_URL, "
            "AGENTMESH_LLM_API_KEY, and AGENTMESH_LLM_MODEL to .env."
        )
    else:
        answer = context.llm_client.complete(
            agent_name="interactive",
            messages=messages,
            variables={"user_input": user_input},
        )
    return ChatTurn(user=user_input, assistant=answer)
