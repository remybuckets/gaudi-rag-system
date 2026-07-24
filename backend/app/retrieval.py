"""Retrieval pipeline: question -> most relevant chunks.

Stage 2. Vector and keyword search hit the DB (stubbed with the exact SQL you
need in comments). Reciprocal Rank Fusion is implemented in full — that is the
'hybrid beats vector-only' win, and it's pure logic you can unit-test today.
"""

from dataclasses import dataclass

from app.config import get_settings


@dataclass
class Hit:
    chunk_id: str
    content: str
    document_id: str
    page_number: int | None
    section: str | None


def vector_search(query: str, top_k: int) -> list[Hit]:
    """Semantic search via pgvector cosine distance.

    TODO: embed the query (input_type="query"), then:
        SELECT id, content, document_id, page_number, section
        FROM chunks
        ORDER BY embedding <=> %(qvec)s
        LIMIT %(top_k)s;
    (`<=>` is cosine distance with vector_cosine_ops.)
    """
    raise NotImplementedError("Wire up query embedding + pgvector search.")


def keyword_search(query: str, top_k: int) -> list[Hit]:
    """Lexical search via Postgres full-text search.

    TODO:
        SELECT id, content, document_id, page_number, section
        FROM chunks
        WHERE content_tsv @@ plainto_tsquery('english', %(q)s)
        ORDER BY ts_rank(content_tsv, plainto_tsquery('english', %(q)s)) DESC
        LIMIT %(top_k)s;
    """
    raise NotImplementedError("Wire up full-text search query.")


def reciprocal_rank_fusion(
    ranked_lists: list[list[Hit]],
    k: int | None = None,
    top_k: int | None = None,
) -> list[Hit]:
    """Fuse multiple ranked lists into one. Score = sum of 1/(k + rank).

    No score normalisation needed — this is the clean, standard way to combine
    vector and keyword results.
    """
    settings = get_settings()
    k = k if k is not None else settings.rrf_k
    top_k = top_k if top_k is not None else settings.retrieval_top_k

    scores: dict[str, float] = {}
    hits: dict[str, Hit] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked):  # rank is 0-based
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            hits[hit.chunk_id] = hit

    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [hits[cid] for cid in ordered_ids[:top_k]]


def hybrid_search(query: str, top_k: int | None = None) -> list[Hit]:
    """Run vector + keyword search and fuse with RRF."""
    settings = get_settings()
    top_k = top_k if top_k is not None else settings.retrieval_top_k
    # Retrieve a wider pool from each, then fuse down to top_k.
    pool = max(top_k * 3, 20)
    vec = vector_search(query, pool)
    kw = keyword_search(query, pool)
    return reciprocal_rank_fusion([vec, kw], top_k=top_k)
