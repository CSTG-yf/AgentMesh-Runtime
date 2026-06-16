from __future__ import annotations

import os
import sys

if sys.platform == "win32":
    import ctypes


def configure_utf8_environment() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    if sys.platform == "win32":
        _configure_windows_console()
    _reconfigure_stream(sys.stdout)
    _reconfigure_stream(sys.stderr)


def _reconfigure_stream(stream: object) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def _configure_windows_console() -> None:
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleCP(65001)
    kernel32.SetConsoleOutputCP(65001)
