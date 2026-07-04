from pydantic import BaseModel, ConfigDict, Field


class ContextPart(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    text: str
    required: bool = False


class ContextAudit(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    max_chars: int
    original_chars: int
    retained_chars: int
    truncated: bool
    safe_fallback: bool
    included_parts: list[str]
    dropped_parts: list[str]
    partial_parts: list[str] = Field(default_factory=list)


class PackedContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    audit: ContextAudit
    retained_parts: dict[str, str] = Field(
        default_factory=dict,
        exclude=True,
        repr=False,
    )


def pack_context(
    role: str,
    max_chars: int,
    parts: list[ContextPart],
) -> PackedContext:
    names = [part.name for part in parts]
    if len(names) != len(set(names)):
        raise ValueError("duplicate context part name")
    non_empty = [part for part in parts if part.text.strip()]
    ordered = [part for part in non_empty if part.required]
    ordered.extend(part for part in non_empty if not part.required)
    original_payload_chars = sum(len(part.text) for part in ordered)

    required = [part for part in ordered if part.required]
    optional = [part for part in ordered if not part.required]
    retained = [part.text for part in required]
    retained_parts = {part.name: part.text for part in required}
    included = [part.name for part in required]
    dropped: list[str] = []
    required_text = "\n\n".join(retained)

    if len(required_text) > max_chars:
        dropped.extend(part.name for part in optional)
        return _packed(
            role=role,
            max_chars=max_chars,
            original_payload_chars=original_payload_chars,
            retained_text=required_text,
            safe_fallback=True,
            included=included,
            dropped=dropped,
            partial=[],
            retained_payload_chars=sum(len(part.text) for part in required),
            retained_parts=retained_parts,
        )

    retained_text = required_text
    retained_payload_chars = sum(len(part.text) for part in required)
    partial: list[str] = []
    for index, part in enumerate(optional):
        separator = "\n\n" if retained_text else ""
        remaining = max_chars - len(retained_text) - len(separator)
        if remaining <= 0:
            dropped.extend(item.name for item in optional[index:])
            break
        retained_part = part.text[:remaining]
        retained_text += separator + retained_part
        retained_parts[part.name] = retained_part
        retained_payload_chars += len(retained_part)
        if remaining < len(part.text):
            partial.append(part.name)
            dropped.extend(item.name for item in optional[index + 1 :])
            break
        included.append(part.name)

    return _packed(
        role=role,
        max_chars=max_chars,
        original_payload_chars=original_payload_chars,
        retained_text=retained_text,
        safe_fallback=False,
        included=included,
        dropped=dropped,
        partial=partial,
        retained_payload_chars=retained_payload_chars,
        retained_parts=retained_parts,
    )


def _packed(
    *,
    role: str,
    max_chars: int,
    original_payload_chars: int,
    retained_text: str,
    safe_fallback: bool,
    included: list[str],
    dropped: list[str],
    partial: list[str],
    retained_payload_chars: int | None = None,
    retained_parts: dict[str, str] | None = None,
) -> PackedContext:
    retained_chars = (
        retained_payload_chars
        if retained_payload_chars is not None
        else len(retained_text)
    )
    return PackedContext(
        text=retained_text,
        retained_parts=retained_parts or {},
        audit=ContextAudit(
            role=role,
            max_chars=max_chars,
            original_chars=original_payload_chars,
            retained_chars=retained_chars,
            truncated=False if safe_fallback else bool(partial or dropped),
            safe_fallback=safe_fallback,
            included_parts=included,
            dropped_parts=dropped,
            partial_parts=partial,
        ),
    )
