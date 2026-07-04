from agentmesh.memory.evidence_pack import pack_evidence


def test_pack_evidence_filters_deduplicates_sorts_and_limits_digest() -> None:
    packed = pack_evidence(
        [
            {"memory_id": "m1", "title": "one", "snippet": "first", "score": 0.9},
            {"memory_id": "m1", "title": "duplicate", "snippet": "ignored", "score": 0.8},
            {"memory_id": "m2", "title": "low", "snippet": "filtered", "score": 0.1},
            {"memory_id": "m3", "title": "three", "snippet": "third", "score": 0.7},
        ],
        min_score=0.5,
        max_items=3,
        max_chars=80,
    )

    assert packed.accepted_ids == ("m1", "m3")
    assert packed.audit.candidate_count == 4
    assert packed.audit.accepted_count == 2
    assert packed.audit.deduplicated_count == 1
    assert packed.audit.filtered_count == 1
    assert len(packed.digest) <= 80
    assert packed.audit.injected_bytes == len(packed.digest.encode("utf-8"))


def test_pack_evidence_deduplicates_normalized_content_without_memory_id() -> None:
    packed = pack_evidence(
        [
            {"title": " Same  Title ", "snippet": "Same\n snippet", "score": 0.9},
            {"title": "same title", "snippet": " same snippet ", "score": 0.8},
        ],
        min_score=0,
        max_items=3,
        max_chars=200,
    )

    assert packed.audit.accepted_count == 1
    assert packed.audit.deduplicated_count == 1


def test_pack_evidence_tracks_unicode_bytes_and_honors_item_and_char_limits() -> None:
    packed = pack_evidence(
        [
            {"memory_id": "甲", "title": "标题", "snippet": "中文证据", "score": 1},
            {"memory_id": "乙", "title": "second", "snippet": "unused", "score": 0.9},
        ],
        min_score=0,
        max_items=1,
        max_chars=28,
    )

    assert len(packed.digest) <= 28
    assert packed.audit.accepted_count == 1
    assert packed.audit.injected_bytes == len(packed.digest.encode("utf-8"))
    assert packed.audit.injected_bytes > len(packed.digest)
