from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class EvidencePackAudit:
    candidate_count: int
    accepted_count: int
    deduplicated_count: int
    filtered_count: int
    injected_bytes: int


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
) -> PackedEvidence:
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
    for item in eligible:
        if len(accepted) >= max_items:
            break
        separator_chars = 1 if lines else 0
        remaining = max_chars - sum(len(line) for line in lines) - separator_chars
        if remaining <= 0:
            break
        line = _evidence_line(item)[:remaining]
        if not line:
            break
        accepted.append(item)
        lines.append(line)
    digest = "\n".join(lines)
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
        ),
    )


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


def _evidence_line(item: dict[str, Any]) -> str:
    memory_id = str(item.get("memory_id") or "evidence").strip()
    title = " ".join(str(item.get("title") or "evidence").split())
    snippet = " ".join(str(item.get("snippet") or "").split())
    attributes = [f"score={_score(item):.3f}"]
    source_agent = _compact_value(item.get("source_agent"))
    if source_agent:
        attributes.append(f"source={source_agent}")
    trace_id = _compact_value(item.get("provenance_trace_id"))
    if trace_id:
        attributes.append(f"trace={trace_id}")
    evidence_refs = item.get("evidence_refs")
    if isinstance(evidence_refs, list):
        refs = [
            compact
            for value in evidence_refs[:3]
            if (compact := _compact_value(value))
        ]
        if refs:
            attributes.append(f"refs={','.join(refs)}")
    return f"[{memory_id}] {title} ({'; '.join(attributes)}): {snippet}"


def _compact_value(value: object, limit: int = 160) -> str:
    return " ".join(str(value or "").split())[:limit]
