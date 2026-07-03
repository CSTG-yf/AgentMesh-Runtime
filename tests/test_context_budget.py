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
    assert packed.audit.original_chars == len(task) + 2 + len(execution) + 2 + 200
    assert packed.audit.retained_chars == len(packed.text)
    assert packed.audit.truncated is True
    assert packed.audit.safe_fallback is True
    assert packed.audit.included_parts == ["task", "execution"]
    assert packed.audit.dropped_parts == ["evidence"]


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
    assert packed.audit.truncated is True
    assert packed.audit.included_parts == ["task"]
    assert packed.audit.dropped_parts == ["evidence"]
