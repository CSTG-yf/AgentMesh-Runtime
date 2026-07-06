from agentmesh.agents.state_refs import first_text_payload
from agentmesh.llm.client import ChatMessage
from agentmesh.memory.hybrid_store import HybridMemoryStore
from agentmesh.memory.search import MemorySearchResult
from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.schema import AMPMessage
from agentmesh.runtime.agent import BaseAgent
from agentmesh.runtime.registry import RuntimeContext
from agentmesh.state.embedding import create_embedding_encoder
from agentmesh.state.store import StateStore
from agentmesh.storage.jsonl import append_jsonl


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
        # Derive query_tags from the query text itself (not from params)
        query_tags = _derive_tags(query)
        deterministic_evidence = (
            message.params.get("deterministic_retrieval_evidence") is True
        )
        llm_evidence: str | None = None
        if context.llm_client is not None and not deterministic_evidence:
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
        evidence: list[dict[str, object]] = []
        if not deterministic_evidence:
            title = "deterministic-local-evidence"
            if message.action == "evidence.refine":
                title = "refined-local-evidence"
            evidence.append(
                {
                    "title": title,
                    "snippet": f"Evidence for {query[:80]}",
                }
            )
        memory_results = _memory_results(context=context, query=query, query_tags=query_tags)
        memory_hits = [_memory_hit_log_item(result) for result in memory_results]
        _append_memory_hit_log(
            context=context,
            action=message.action or "",
            query=query,
            query_tags=query_tags,
            memory_hits=memory_hits,
        )
        evidence.extend(
            {
                "title": result.memory.task_topic,
                "snippet": _compact_snippet(result.memory.summary, 300),
                "memory_id": result.memory.memory_id,
                "memory_score": round(float(result.score), 4),
                "source_agent": result.memory.source_agent,
                "provenance_trace_id": result.memory.provenance_trace_id,
                "evidence_refs": result.memory.evidence_refs,
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
        return _run_memory_search(
            context.memory_store,
            query,
            query_tags,
            default_limit=context.config.protocol.memory_reuse_default_limit,
        )
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
        return _run_memory_search(
            store,
            query,
            query_tags,
            default_limit=context.config.protocol.memory_reuse_default_limit,
        )
    finally:
        state_store.close()


def _compact_snippet(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 1)]}…"


def _run_memory_search(
    store: HybridMemoryStore,
    query: str,
    query_tags: list[str],
    default_limit: int = 1,
) -> list[MemorySearchResult]:
    limit = _memory_reuse_limit(
        query=query,
        query_tags=query_tags,
        default_limit=default_limit,
    )
    results = [
        result
        for result in store.semantic_search_with_scores(
            query,
            limit=limit,
            query_tags=query_tags,
        )
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


def _memory_reuse_limit(
    *,
    query: str,
    query_tags: list[str],
    default_limit: int = 1,
) -> int:
    lowered = query.lower()
    if query_tags == ["general"]:
        return 1
    if any(term in lowered for term in ["memory", "reuse", "prior", "remember", "记忆", "复用"]):
        return max(1, min(2, default_limit + 1))
    return max(1, default_limit)


def _memory_hit_log_item(result: MemorySearchResult) -> dict[str, object]:
    memory = result.memory
    return {
        "memory_id": memory.memory_id,
        "source_agent": memory.source_agent,
        "task_topic": memory.task_topic,
        "summary": memory.summary,
        "tags": memory.tags,
        "memory_type": memory.memory_type,
        "domain": memory.domain,
        "provenance_trace_id": memory.provenance_trace_id,
        "state_refs": memory.state_refs,
        "evidence_refs": memory.evidence_refs,
        "score": round(float(result.score), 4),
        "semantic_similarity": round(float(result.semantic_similarity), 4),
        "tag_overlap_score": round(float(result.tag_overlap_score), 4),
        "validity_score": round(float(result.validity_score), 4),
        "confidence_score": round(float(result.confidence_score), 4),
        "reuse_score": round(float(result.reuse_score), 4),
        "recency_score": round(float(result.recency_score), 4),
        "reason": result.reason,
    }


def _append_memory_hit_log(
    *,
    context: RuntimeContext,
    action: str,
    query: str,
    query_tags: list[str],
    memory_hits: list[dict[str, object]],
) -> None:
    if not memory_hits:
        return
    append_jsonl(
        context.paths.protocol_memory_hits,
        {
            "trace_id": context.trace_id,
            "agent": "retriever",
            "action": action,
            "query": query,
            "query_tags": query_tags,
            "memory_hit_count": len(memory_hits),
            "memory_hits": memory_hits,
        },
    )


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
    """Strict filter: only return memory when it's clearly relevant.

    Low-similarity results waste evidence bandwidth and confuse downstream
    agents.  Thresholds are set to reject noise while keeping clearly useful
    hits (e.g. "write a quicksort" hitting a prior quicksort record).
    """
    if query_tags == ["general"]:
        return semantic_similarity >= 0.55 and score >= 0.60
    if tag_overlap_score >= 0.5:
        return semantic_similarity >= 0.45 and score >= 0.55
    if tag_overlap_score > 0:
        return semantic_similarity >= 0.55 and score >= 0.60
    if query_tags:
        return semantic_similarity >= 0.72 and score >= 0.70
    return semantic_similarity >= 0.60 and score >= 0.65
