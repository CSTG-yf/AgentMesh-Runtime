from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from typing import Any

_UNTRUSTED_LABEL = "UNTRUSTED_EVIDENCE_DATA_ONLY"


@dataclass(frozen=True)
class EvidencePackAudit:
    candidate_count: int
    accepted_count: int
    deduplicated_count: int
    filtered_count: int
    injected_bytes: int
    partial_count: int = 0


@dataclass(frozen=True)
class PackedEvidence:
    digest: str
    accepted_ids: tuple[str, ...]
    audit: EvidencePackAudit


def pack_evidence(
    evidence: list[dict[str, Any]],
    *,
    min_score: float,
    max_items: int,
    max_chars: int,
    max_bytes: int | None = None,
) -> PackedEvidence:
    byte_cap = max_chars * 4 if max_bytes is None else max_bytes
    ranked = sorted(evidence, key=_score, reverse=True)
    seen: set[str] = set()
    eligible: list[dict[str, Any]] = []
    deduplicated_count = 0
    filtered_count = 0
    for item in ranked:
        if _score(item) < min_score:
            filtered_count += 1
            continue
        key = _deduplication_key(item)
        if key in seen:
            deduplicated_count += 1
            continue
        seen.add(key)
        eligible.append(item)

    accepted: list[dict[str, Any]] = []
    lines: list[str] = []
    partial_count = 0
    for item in eligible:
        if len(accepted) >= max_items:
            break
        line, partial = _fit_record(
            item,
            existing_lines=lines,
            max_chars=max_chars,
            max_bytes=byte_cap,
        )
        if line is None:
            continue
        accepted.append(item)
        lines.append(line)
        partial_count += int(partial)

    digest = _render_digest(lines)
    accepted_ids = tuple(
        memory_id
        for item in accepted
        if (memory_id := str(item.get("memory_id") or "").strip())
    )
    return PackedEvidence(
        digest=digest,
        accepted_ids=accepted_ids,
        audit=EvidencePackAudit(
            candidate_count=len(evidence),
            accepted_count=len(accepted),
            deduplicated_count=deduplicated_count,
            filtered_count=filtered_count,
            injected_bytes=len(digest.encode("utf-8")),
            partial_count=partial_count,
        ),
    )


def _fit_record(
    item: dict[str, Any],
    *,
    existing_lines: list[str],
    max_chars: int,
    max_bytes: int,
) -> tuple[str | None, bool]:
    full = _structured_record(item)
    full_line = _encode_record(full)
    if _fits(existing_lines + [full_line], max_chars=max_chars, max_bytes=max_bytes):
        return full_line, False

    snippet = str(full.get("snippet", ""))
    low = 0
    high = len(snippet)
    best: str | None = None
    while low <= high:
        midpoint = (low + high) // 2
        candidate = {**full, "snippet": snippet[:midpoint]}
        encoded = _encode_record(candidate)
        if _fits(existing_lines + [encoded], max_chars=max_chars, max_bytes=max_bytes):
            best = encoded
            low = midpoint + 1
        else:
            high = midpoint - 1
    if best is not None:
        return best, True

    minimum = _minimum_record(item)
    minimum_line = _encode_record(minimum)
    if _fits(
        existing_lines + [minimum_line],
        max_chars=max_chars,
        max_bytes=max_bytes,
    ):
        return minimum_line, True
    return None, False


def _structured_record(item: dict[str, Any]) -> dict[str, Any]:
    record = _minimum_record(item)
    record.update(
        {
            "score": _score(item),
            "title": str(item.get("title") or "evidence"),
            "snippet": str(item.get("snippet") or ""),
        }
    )
    return record


def _minimum_record(item: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "memory_id": str(item.get("memory_id") or "evidence"),
    }
    for key in ("source_agent", "provenance_trace_id"):
        value = item.get(key)
        if value is not None and str(value):
            record[key] = str(value)
    refs = item.get("evidence_refs")
    if isinstance(refs, list):
        record["evidence_refs"] = [str(value) for value in refs[:3]]
    return record


def _encode_record(record: dict[str, Any]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _render_digest(lines: list[str]) -> str:
    if not lines:
        return ""
    return "\n".join([_UNTRUSTED_LABEL, *lines])


def _fits(lines: list[str], *, max_chars: int, max_bytes: int) -> bool:
    digest = _render_digest(lines)
    return len(digest) <= max_chars and len(digest.encode("utf-8")) <= max_bytes


def _score(item: dict[str, Any]) -> float:
    value = item.get("score", item.get("memory_score", 0))
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if isfinite(score) else 0.0


def _deduplication_key(item: dict[str, Any]) -> str:
    memory_id = str(item.get("memory_id") or "").strip()
    if memory_id:
        return f"id:{memory_id}"
    title = " ".join(str(item.get("title") or "").lower().split())
    snippet = " ".join(str(item.get("snippet") or "").lower().split())
    return f"content:{title}|{snippet}"
