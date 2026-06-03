from collections.abc import Mapping
from pathlib import Path
from typing import Any

import orjson


def append_jsonl(path: Path, item: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as file:
        file.write(orjson.dumps(item))
        file.write(b"\n")


def write_jsonl(path: Path, items: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        for item in items:
            file.write(orjson.dumps(item))
            file.write(b"\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("rb") as file:
        for line in file:
            if line.strip():
                value = orjson.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows
