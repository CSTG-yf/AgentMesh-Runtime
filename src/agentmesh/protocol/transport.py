from __future__ import annotations

import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from agentmesh.protocol.codec import encode_message
from agentmesh.protocol.schema import AMPMessage
from agentmesh.runtime.registry import AgentRegistry, RuntimeContext


@dataclass
class TransportMetrics:
    transport_type: str
    send_count: int = 0
    total_bytes: int = 0
    total_latency_ms: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return self.total_latency_ms / len(self.latencies_ms)

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        index = min(len(ordered) - 1, int(len(ordered) * 0.99))
        return ordered[index]

    def record(self, *, byte_count: int, latency_ms: float) -> None:
        self.send_count += 1
        self.total_bytes += byte_count
        self.total_latency_ms += latency_ms
        self.latencies_ms.append(latency_ms)

    def model_dump(self) -> dict[str, float | int | str]:
        return {
            "transport_type": self.transport_type,
            "send_count": self.send_count,
            "total_bytes": self.total_bytes,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
        }


class AgentTransport(ABC):
    @property
    @abstractmethod
    def transport_type(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def send(self, message: AMPMessage, timeout: float | None = None) -> AMPMessage:
        raise NotImplementedError

    @abstractmethod
    def metrics(self) -> TransportMetrics:
        raise NotImplementedError


class InProcTransport(AgentTransport):
    def __init__(self, registry: AgentRegistry, context: RuntimeContext) -> None:
        self._registry = registry
        self._context = context
        self._metrics = TransportMetrics(transport_type="inproc")

    @property
    def transport_type(self) -> str:
        return "inproc"

    def send(self, message: AMPMessage, timeout: float | None = None) -> AMPMessage:
        _ = timeout
        start = time.perf_counter()
        result = self._registry.get(message.target_agent).handle(message, self._context)
        latency_ms = (time.perf_counter() - start) * 1000
        self._metrics.record(byte_count=len(encode_message(message)), latency_ms=latency_ms)
        return result

    def metrics(self) -> TransportMetrics:
        return self._metrics


class SocketFrameTransport(AgentTransport):
    """Length-prefixed AMPMessage frame transport for IPC/socket experiments."""

    def __init__(self, sock: socket.socket, *, transport_type: str = "socket") -> None:
        self._sock = sock
        self._metrics = TransportMetrics(transport_type=transport_type)

    @property
    def transport_type(self) -> str:
        return str(self._metrics.transport_type)

    def send_frame(self, message: AMPMessage) -> None:
        payload = encode_message(message)
        header = len(payload).to_bytes(4, byteorder="big", signed=False)
        start = time.perf_counter()
        self._sock.sendall(header + payload)
        latency_ms = (time.perf_counter() - start) * 1000
        self._metrics.record(byte_count=len(header) + len(payload), latency_ms=latency_ms)

    def receive_frame(self) -> AMPMessage:
        header = _recv_exact(self._sock, 4)
        payload_size = int.from_bytes(header, byteorder="big", signed=False)
        payload = _recv_exact(self._sock, payload_size)
        return AMPMessage.model_validate_json(payload)

    def send(self, message: AMPMessage, timeout: float | None = None) -> AMPMessage:
        previous_timeout = self._sock.gettimeout()
        if timeout is not None:
            self._sock.settimeout(timeout)
        try:
            self.send_frame(message)
            return self.receive_frame()
        finally:
            if timeout is not None:
                self._sock.settimeout(previous_timeout)

    def metrics(self) -> TransportMetrics:
        return self._metrics


class TransportFactory:
    @staticmethod
    def inproc(registry: AgentRegistry, context: RuntimeContext) -> AgentTransport:
        return InProcTransport(registry=registry, context=context)

    @staticmethod
    def socket_frame(sock: socket.socket, *, transport_type: str = "socket") -> AgentTransport:
        return SocketFrameTransport(sock, transport_type=transport_type)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError("socket closed while reading AMP frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
