from agentmesh.memory.tagger import LLMMemoryTagger, RuleBasedMemoryTagger


def test_rule_based_memory_tagger_marks_code_tasks() -> None:
    classification = RuleBasedMemoryTagger().classify(
        task="Write Python code that validates a small task.",
        summary="Sandbox executed validation code and returned JSON status.",
    )

    assert "code" in classification.tags


def test_llm_memory_tagger_preserves_rule_based_tags() -> None:
    class TagOnlyLLM:
        def complete(self, **kwargs):
            return '{"topic":"sort","tags":["runtime"],"importance_score":0.8}'

    classification = LLMMemoryTagger(TagOnlyLLM()).classify(
        task="Write Python code that validates a small task.",
        summary="Sandbox executed validation code and returned JSON status.",
    )

    assert "runtime" in classification.tags
    assert "code" in classification.tags
