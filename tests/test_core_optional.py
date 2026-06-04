from agentmesh.core import rust_available


def test_rust_core_is_optional() -> None:
    assert isinstance(rust_available(), bool)
