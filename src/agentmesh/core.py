import importlib
from types import ModuleType
from typing import Any

try:
    _rust_core: ModuleType | None = importlib.import_module("agentmesh_core")
except Exception:
    _rust_core = None


def rust_available() -> bool:
    return _rust_core is not None


def rust_core() -> Any:
    if _rust_core is None:
        raise RuntimeError("agentmesh_core Rust extension is not installed")
    return _rust_core
