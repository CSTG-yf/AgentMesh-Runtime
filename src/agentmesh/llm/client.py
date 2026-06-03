import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol

import orjson
from pydantic import BaseModel

from agentmesh.config import AgentMeshConfig


class ChatMessage(BaseModel):
    role: str
    content: str


class LLMClient(Protocol):
    def complete(
        self,
        *,
        agent_name: str,
        messages: list[ChatMessage],
        variables: dict[str, object] | None = None,
    ) -> str:
        raise NotImplementedError


Transport = Callable[[str, dict[str, str], dict[str, object], float], dict[str, object]]


class OpenAICompatibleClient:
    def __init__(self, config: AgentMeshConfig, transport: Transport | None = None) -> None:
        if not config.llm.configured:
            raise ValueError("LLM config requires base_url, api_key, and model")
        self._config = config
        self._transport = transport or _default_transport

    def complete(
        self,
        *,
        agent_name: str,
        messages: list[ChatMessage],
        variables: dict[str, object] | None = None,
    ) -> str:
        del agent_name, variables
        api_key = self._config.llm.api_key
        model = self._config.llm.model
        if api_key is None or model is None:
            raise ValueError("LLM config is incomplete")
        payload: dict[str, object] = {
            "model": model,
            "messages": [message.model_dump() for message in messages],
        }
        headers = {
            "Authorization": f"Bearer {api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        response = self._transport(
            _chat_completions_url(str(self._config.llm.base_url)),
            headers,
            payload,
            self._config.llm.timeout_seconds,
        )
        return _extract_content(response)


def create_llm_client(config: AgentMeshConfig) -> LLMClient | None:
    if not config.llm.configured:
        return None
    return OpenAICompatibleClient(config)


def _chat_completions_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"


def _default_transport(
    url: str,
    headers: dict[str, str],
    payload: dict[str, object],
    timeout_seconds: float,
) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=orjson.dumps(payload),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            data = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc
    parsed: Any = orjson.loads(data)
    if not isinstance(parsed, dict):
        raise RuntimeError("LLM response must be a JSON object")
    return parsed


def _extract_content(response: dict[str, object]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("LLM choice must be an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("LLM choice missing message")
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("LLM message missing content")
    return content
