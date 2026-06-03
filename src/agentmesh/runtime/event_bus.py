from collections.abc import Callable

from agentmesh.protocol.schema import AMPMessage


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[Callable[[AMPMessage], None]] = []

    def subscribe(self, handler: Callable[[AMPMessage], None]) -> None:
        self._subscribers.append(handler)

    def publish(self, message: AMPMessage) -> None:
        for handler in self._subscribers:
            handler(message)
