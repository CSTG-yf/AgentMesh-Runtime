import pytest

from agentmesh.runtime.context_budget import ContextPart, pack_context


def test_pack_context_preserves_required_parts_and_truncates_optional_evidence() -> None:
    task = "Return sum 77 and even count 5."
    execution = "sandbox exit 0"

    packed = pack_context(
        role="summarizer",
        max_chars=45,
        parts=[
            ContextPart(name="task", text=task, required=True),
            ContextPart(name="execution", text=execution, required=True),
            ContextPart(name="evidence", text="x" * 200),
        ],
    )

    assert task in packed.text
    assert execution in packed.text
    assert "x" * 200 not in packed.text
    assert packed.audit.original_chars == len(task) + len(execution) + 200
    assert packed.audit.retained_chars == len(task) + len(execution)
    assert packed.audit.truncated is False
    assert packed.audit.safe_fallback is True
    assert packed.audit.included_parts == ["task", "execution"]
    assert packed.audit.dropped_parts == ["evidence"]
    assert packed.audit.partial_parts == []


def test_pack_context_returns_complete_required_text_when_it_exceeds_budget() -> None:
    packed = pack_context(
        role="planner",
        max_chars=5,
        parts=[
            ContextPart(name="task", text="required task", required=True),
            ContextPart(name="evidence", text="optional"),
        ],
    )

    assert packed.text == "required task"
    assert packed.audit.safe_fallback is True
    assert packed.audit.truncated is False
    assert packed.audit.included_parts == ["task"]
    assert packed.audit.dropped_parts == ["evidence"]


def test_pack_context_reports_partial_optional_payload_without_separator_chars() -> None:
    packed = pack_context(
        role="retriever",
        max_chars=8,
        parts=[
            ContextPart(name="task", text="task", required=True),
            ContextPart(name="intent", text="abcdef"),
        ],
    )

    assert packed.text == "task\n\nab"
    assert packed.audit.original_chars == 10
    assert packed.audit.retained_chars == 6
    assert packed.audit.truncated is True
    assert packed.audit.included_parts == ["task"]
    assert packed.audit.partial_parts == ["intent"]
    assert packed.audit.dropped_parts == []
    assert packed.retained_parts == {"task": "task", "intent": "ab"}


def test_pack_context_retained_parts_tracks_dropped_and_unicode_partial_text() -> None:
    dropped = pack_context(
        role="executor",
        max_chars=4,
        parts=[
            ContextPart(name="task", text="required", required=True),
            ContextPart(name="evidence_digest", text="证据"),
        ],
    )
    partial = pack_context(
        role="executor",
        max_chars=3,
        parts=[ContextPart(name="evidence_digest", text="证据内容")],
    )

    assert dropped.retained_parts == {"task": "required"}
    assert "证据" not in dropped.audit.model_dump_json()
    assert partial.retained_parts == {"evidence_digest": "证据内"}
    assert len(partial.retained_parts["evidence_digest"].encode("utf-8")) == 9


def test_pack_context_rejects_duplicate_part_names() -> None:
    with pytest.raises(ValueError, match="duplicate context part name"):
        pack_context(
            role="planner",
            max_chars=20,
            parts=[
                ContextPart(name="task", text="one", required=True),
                ContextPart(name="task", text="two"),
            ],
        )
