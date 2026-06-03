from agentmesh.protocol.schema import AMPMessage


class SyncScheduler:
    def run(self, messages: list[AMPMessage]) -> list[AMPMessage]:
        return messages
