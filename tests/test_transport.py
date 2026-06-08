import socket
import threading
from pathlib import Path

from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.schema import AMPMessage
from agentmesh.protocol.transport import SocketFrameTransport, TransportFactory
from agentmesh.runtime.orchestrator import default_registry
from agentmesh.runtime.registry import RuntimeContext
from agentmesh.runtime.scheduler import ProtocolScheduler
from agentmesh.storage.paths import RuntimePaths


def test_socket_frame_transport_round_trips_amp_message() -> None:
    left, right = socket.socketpair()
    try:
        sender = SocketFrameTransport(left, transport_type="socketpair")
        receiver = SocketFrameTransport(right, transport_type="socketpair")
        message = AMPMessage(
            trace_id="trace-transport",
            source_agent="planner",
            target_agent="summarizer",
            msg_type=MsgType.INVOKE,
            action="summary.create",
            params={"code_result": "none"},
        )

        sender.send_frame(message)
        received = receiver.receive_frame()

        assert received == message
        assert sender.metrics().send_count == 1
        assert sender.metrics().total_bytes > 4
    finally:
        left.close()
        right.close()


def test_socket_frame_transport_send_waits_for_response() -> None:
    left, right = socket.socketpair()
    try:
        client = SocketFrameTransport(left, transport_type="socketpair")
        server = SocketFrameTransport(right, transport_type="socketpair")
        request = AMPMessage(
            trace_id="trace-transport",
            source_agent="planner",
            target_agent="summarizer",
            msg_type=MsgType.INVOKE,
            action="summary.create",
            params={"code_result": "none"},
        )

        def respond() -> None:
            incoming = server.receive_frame()
            server.send_frame(
                incoming.model_copy(
                    update={
                        "source_agent": incoming.target_agent,
                        "target_agent": incoming.source_agent,
                        "msg_type": MsgType.RESULT,
                        "result": {"ok": True},
                    }
                )
            )

        thread = threading.Thread(target=respond)
        thread.start()
        response = client.send(request, timeout=2)
        thread.join(timeout=2)

        assert response.msg_type == MsgType.RESULT
        assert response.result == {"ok": True}
        assert client.metrics().send_count == 1
    finally:
        left.close()
        right.close()


def test_transport_factory_creates_inproc_and_socket_transports(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    context = RuntimeContext.from_paths(paths=paths, trace_id="trace-transport")
    inproc = TransportFactory.inproc(default_registry(), context)
    left, right = socket.socketpair()
    try:
        socket_transport = TransportFactory.socket_frame(left, transport_type="socketpair")

        assert inproc.transport_type == "inproc"
        assert socket_transport.transport_type == "socketpair"
    finally:
        left.close()
        right.close()


def test_scheduler_records_inproc_transport_metrics(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    context = RuntimeContext.from_paths(paths=paths, trace_id="trace-transport")
    scheduler = ProtocolScheduler(
        registry=default_registry(),
        context=context,
        messages=[],
        protocol_map={"summarizer": "summary.create"},
    )

    scheduler.invoke(
        source_agent="runtime",
        action="summary.create",
        params={"code_result": "none"},
    )
    metrics = scheduler.transport_metrics()

    assert metrics.transport_type == "inproc"
    assert metrics.send_count == 1
    assert metrics.total_bytes > 0
    assert metrics.avg_latency_ms >= 0
