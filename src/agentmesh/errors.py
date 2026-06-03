class AgentMeshError(Exception):
    """Base exception for AgentMesh Runtime."""


class ProtocolError(AgentMeshError):
    """Raised when an AMP message violates protocol rules."""


class StateNotFoundError(AgentMeshError):
    """Raised when a state reference cannot be resolved."""


class MemoryNotFoundError(AgentMeshError):
    """Raised when a memory unit cannot be resolved."""


class CapabilityNotFoundError(AgentMeshError):
    """Raised when no registered agent provides a requested capability."""


class SandboxTimeoutError(AgentMeshError):
    """Raised when sandboxed execution exceeds its timeout."""


class BenchmarkConfigError(AgentMeshError):
    """Raised when a benchmark suite is invalid."""
