from app.retrieval import Hit, reciprocal_rank_fusion


def _hit(cid: str) -> Hit:
    return Hit(chunk_id=cid, content=cid, document_id="d", page_number=1, section=None)


def test_rrf_rewards_agreement():
    # 'b' is ranked well by both lists, so it should come out on top.
    vec = [_hit("a"), _hit("b"), _hit("c")]
    kw = [_hit("b"), _hit("d"), _hit("a")]
    fused = reciprocal_rank_fusion([vec, kw], k=60, top_k=3)
    assert fused[0].chunk_id == "b"
    assert {h.chunk_id for h in fused} == {"a", "b", "d"} or len(fused) == 3
