import json
from io import StringIO

from rich.console import Console

from agentmesh.encoding import configure_utf8_environment


class ReconfigurableStream:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def reconfigure(self, **kwargs: str) -> None:
        self.calls.append(kwargs)


def test_configure_utf8_environment_sets_python_utf8_flags(monkeypatch) -> None:
    stdout = ReconfigurableStream()
    stderr = ReconfigurableStream()
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    configure_utf8_environment()

    assert stdout.calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert stderr.calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert "utf-8" == __import__("os").environ["PYTHONIOENCODING"]
    assert "1" == __import__("os").environ["PYTHONUTF8"]


def test_json_console_output_keeps_chinese_text() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False)

    console.print(
        json.dumps({"summary": "最长公共子序列 中文验证"}, ensure_ascii=False),
        markup=False,
    )

    rendered = output.getvalue()
    assert "最长公共子序列 中文验证" in rendered
    assert "\\u6700" not in rendered
