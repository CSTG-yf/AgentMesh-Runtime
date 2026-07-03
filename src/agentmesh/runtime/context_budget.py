from pydantic import BaseModel, ConfigDict


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


class PackedContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    audit: ContextAudit


def pack_context(
    role: str,
    max_chars: int,
    parts: list[ContextPart],
) -> PackedContext:
    non_empty = [part for part in parts if part.text.strip()]
    ordered = [part for part in non_empty if part.required]
    ordered.extend(part for part in non_empty if not part.required)
    original_text = "\n\n".join(part.text for part in ordered)

    required = [part for part in ordered if part.required]
    optional = [part for part in ordered if not part.required]
    retained = [part.text for part in required]
    included = [part.name for part in required]
    dropped: list[str] = []
    required_text = "\n\n".join(retained)

    if len(required_text) > max_chars:
        dropped.extend(part.name for part in optional)
        return _packed(
            role=role,
            max_chars=max_chars,
            original_text=original_text,
            retained_text=required_text,
            safe_fallback=True,
            included=included,
            dropped=dropped,
        )

    retained_text = required_text
    for index, part in enumerate(optional):
        separator = "\n\n" if retained_text else ""
        remaining = max_chars - len(retained_text) - len(separator)
        if remaining <= 0:
            dropped.extend(item.name for item in optional[index:])
            break
        retained_text += separator + part.text[:remaining]
        included.append(part.name)
        if remaining < len(part.text):
            dropped.extend(item.name for item in optional[index + 1 :])
            break

    return _packed(
        role=role,
        max_chars=max_chars,
        original_text=original_text,
        retained_text=retained_text,
        safe_fallback=False,
        included=included,
        dropped=dropped,
    )


def _packed(
    *,
    role: str,
    max_chars: int,
    original_text: str,
    retained_text: str,
    safe_fallback: bool,
    included: list[str],
    dropped: list[str],
) -> PackedContext:
    return PackedContext(
        text=retained_text,
        audit=ContextAudit(
            role=role,
            max_chars=max_chars,
            original_chars=len(original_text),
            retained_chars=len(retained_text),
            truncated=len(retained_text) < len(original_text),
            safe_fallback=safe_fallback,
            included_parts=included,
            dropped_parts=dropped,
        ),
    )
