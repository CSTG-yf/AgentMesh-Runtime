import importlib
from types import ModuleType
from typing import Any

try:
    _rust_core: ModuleType | None = importlib.import_module("agentmesh_core")
except Exception:
    _rust_core = None

_REQUIRED_FUNCTIONS = {
    "version",
    "hash_embedding",
    "cosine_similarity_f32",
    "encode_json_bytes",
    "decode_json_text",
    "encode_msgpack_bytes",
    "decode_msgpack_json_text",
    "encode_typed_envelope_bytes",
    "decode_typed_envelope_json_text",
    "run_python_subprocess",
    "parse_state_ref_parts",
    "top_k_cosine",
    "memory_rank_top_k",
}


def rust_available() -> bool:
    return _rust_core is not None and all(
        hasattr(_rust_core, function_name) for function_name in _REQUIRED_FUNCTIONS
    )


def rust_core() -> Any:
    if _rust_core is None:
        raise RuntimeError("agentmesh_core Rust extension is not installed")
    return _rust_core
