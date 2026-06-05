from __future__ import annotations

import shlex
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedShellCommand:
    name: str
    args: list[str]
    raw_text: str


def parse_shell_line(line: str) -> ParsedShellCommand | None:
    stripped = line.strip()
    if not stripped:
        return None
    if not stripped.startswith("/"):
        return ParsedShellCommand(name="compare", args=[stripped], raw_text=stripped)

    parts = shlex.split(stripped[1:])
    if not parts:
        return None
    return ParsedShellCommand(name=parts[0].lower(), args=parts[1:], raw_text=stripped)


def join_prompt(args: list[str]) -> str:
    return " ".join(args).strip()

