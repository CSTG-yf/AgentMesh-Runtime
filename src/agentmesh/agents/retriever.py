from agentmesh.agents.state_refs import first_text_payload, read_state_payloads
from agentmesh.llm.client import ChatMessage
from agentmesh.memory.hybrid_store import HybridMemoryStore
from agentmesh.memory.search import MemorySearchResult
from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.schema import AMPMessage
from agentmesh.runtime.agent import BaseAgent
from agentmesh.runtime.registry import RuntimeContext
from agentmesh.state.embedding import create_embedding_encoder
from agentmesh.state.store import StateStore


class RetrieverAgent(BaseAgent):
    name = "retriever"
    capabilities = [
        "memory.keyword_search",
        "memory.tag_search",
        "memory.semantic_search",
        "evidence.refine",
        "evidence.collect",
    ]

    def handle(self, message: AMPMessage, context: RuntimeContext) -> AMPMessage:
        query = str(message.params.get("query", ""))
        if not query:
            query = first_text_payload(
                context=context,
                state_refs=message.state_refs,
                consumer=self.name,
            )
        else:
            read_state_payloads(
                context=context,
                state_refs=message.state_refs,
                consumer=self.name,
            )
        # Derive query_tags from the query text itself (not from params)
        query_tags = _derive_tags(query)
        llm_evidence: str | None = None
        if context.llm_client is not None:
            try:
                llm_evidence = context.llm_client.complete(
                    agent_name=self.name,
                    messages=[
                        ChatMessage(
                            role="system",
                            content=context.prompts.render(self.name, {"query": query}),
                        ),
                        ChatMessage(role="user", content=query),
                    ],
                    variables={"query": query},
                )
            except Exception:
                llm_evidence = None
        title = "deterministic-local-evidence"
        if message.action == "evidence.refine":
            title = "refined-local-evidence"
        evidence: list[dict[str, object]] = [
            {
                "title": title,
                "snippet": f"Evidence for {query[:80]}",
            }
        ]
        memory_results = _memory_results(context=context, query=query, query_tags=query_tags)
        evidence.extend(
            {
                "title": result.memory.task_topic,
                "snippet": result.memory.summary,
                "memory_id": result.memory.memory_id,
                "memory_score": round(float(result.score), 4),
                "semantic_similarity": round(float(result.semantic_similarity), 4),
                "tag_overlap_score": round(float(result.tag_overlap_score), 4),
                "reason": result.reason,
            }
            for result in memory_results
        )
        if llm_evidence:
            evidence.insert(0, {"title": "llm-evidence", "snippet": llm_evidence})
        return AMPMessage(
            trace_id=context.trace_id,
            source_agent=self.name,
            target_agent=message.source_agent,
            msg_type=MsgType.RESULT,
            action=message.action,
            result={"evidence": evidence},
            state_refs=message.state_refs,
        )


def _memory_results(
    context: RuntimeContext,
    query: str,
    query_tags: list[str],
) -> list[MemorySearchResult]:
    if not query:
        return []
    if context.memory_store is not None:
        return _run_memory_search(context.memory_store, query, query_tags)
    state_store = StateStore(
        context.paths,
        payload_backend=context.config.state.payload_backend,
        shm_threshold_bytes=context.config.state.shm_threshold_bytes,
    )
    try:
        store = HybridMemoryStore(
            paths=context.paths,
            state_store=state_store,
            encoder=create_embedding_encoder(context.config.embedding),
        )
        return _run_memory_search(store, query, query_tags)
    finally:
        state_store.close()


def _run_memory_search(
    store: HybridMemoryStore,
    query: str,
    query_tags: list[str],
) -> list[MemorySearchResult]:
    results = [
        result
        for result in store.semantic_search_with_scores(query, query_tags=query_tags)
        if _should_reuse_memory_result(
            score=float(result.score),
            semantic_similarity=float(result.semantic_similarity),
            tag_overlap_score=float(result.tag_overlap_score),
            query_tags=query_tags,
        )
    ]
    for result in results:
        store.increment_reuse(result.memory.memory_id)
    return results


def _derive_tags(text: str) -> list[str]:
    lowered = text.lower()
    tag_terms = {
        "protocol": ["protocol", "\u534f\u8bae"],
        "state": ["state", "\u72b6\u6001"],
        "memory": ["memory", "\u8bb0\u5fc6"],
        "benchmark": ["benchmark", "\u8bc4\u6d4b", "\u57fa\u51c6"],
        "agent": ["agent"],
        "runtime": ["runtime"],
        "code": ["\u4ee3\u7801", "\u811a\u672c", "\u6392\u5e8f", "sort"],
    }
    tags: list[str] = []
    for candidate, terms in tag_terms.items():
        if any(term in lowered for term in terms):
            tags.append(candidate)
    return tags or ["general"]


def _should_reuse_memory_result(
    *,
    score: float,
    semantic_similarity: float,
    tag_overlap_score: float,
    query_tags: list[str],
) -> bool:
    if query_tags == ["general"]:
        return semantic_similarity >= 0.20 and score >= 0.45
    if tag_overlap_score > 0 and score >= 0.30:
        return True
    if query_tags and tag_overlap_score <= 0:
        return semantic_similarity >= 0.55 and score >= 0.55
    return semantic_similarity >= 0.25 and score >= 0.50
