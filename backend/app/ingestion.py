"""Ingestion pipeline: PDF -> text -> chunks (+metadata) -> embeddings -> Postgres.

Stage 1 of the three pipelines. Extraction and chunking are implemented;
embedding + storage have clear TODOs where your DB and API keys plug in.
"""

from dataclasses import dataclass
from functools import lru_cache

import fitz  # pymupdf
import voyageai

from app.config import get_settings

# Voyage caps texts-per-request.
_EMBED_BATCH_SIZE = 128


@lru_cache
def _voyage_client() -> voyageai.Client:
    return voyageai.Client(api_key=get_settings().voyage_api_key)


@dataclass
class Chunk:
    content: str
    page_number: int
    chunk_index: int
    section: str | None = None
    token_count: int | None = None


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...]. page_number is 1-based for citations."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        pages.append((i, text))
    doc.close()
    return pages


def chunk_pages(
    pages: list[tuple[int, str]],
    target_words: int = 220,  # ~500-800 tokens
    overlap_words: int = 30,  # ~10-15% overlap
) -> list[Chunk]:
    """Simple word-window chunker with overlap, preserving page numbers.

    TODO(structure-aware): detect headings / clause numbers (e.g. "6.2",
    "Requirement B1") and split on those first so a chunk maps to one clause.
    That is where retrieval quality on Approved Documents really improves.
    """
    chunks: list[Chunk] = []
    idx = 0
    for page_number, text in pages:
        if not text:
            continue
        words = text.split()
        start = 0
        while start < len(words):
            window = words[start : start + target_words]
            content = " ".join(window)
            chunks.append(
                Chunk(
                    content=content,
                    page_number=page_number,
                    chunk_index=idx,
                    token_count=len(window),  # rough proxy; swap for a real tokenizer
                )
            )
            idx += 1
            if start + target_words >= len(words):
                break
            start += target_words - overlap_words
    return chunks


def embed_texts(
    texts: list[str],
    input_type: str = "document",  # use "query" when embedding a user question
) -> list[list[float]]:
    """
    Embed a batch of chunk texts.

    Returns: One vector per input text, in the same order
    Each vector's length == settings.embedding_dim
    """
    if not texts:
        return []

    settings = get_settings()
    client = _voyage_client()

    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        # TODO(issue #3): wrap this call in retry/backoff for rate limits.
        result = client.embed(batch, model=settings.embedding_model, input_type=input_type)
        vectors.extend(result.embeddings)

        # Sanity Check: catches a model/dimension mismatch immediately instead
        # of letting it fail deep inside a Postgres INSERT. Must equal vector(...) in
        # db/init.sql
        if len(vectors[0]) != settings.embedding_dim:
            raise ValueError(
                f"Got embedding dim {len(vectors[0])}, but EMBEDDING_DIM is "
                f"{settings.embedding_dim}. Update EMBEDDING_DIM and db/init.sql to match."
            )
        return vectors


def ingest_pdf(pdf_path: str, filename: str) -> str:
    """Full pipeline for one PDF. Returns the new document id.

    Steps: extract -> chunk -> embed (batched) -> INSERT document + chunks.
    TODO: insert into `documents`, batch-embed, then INSERT chunks with
    embedding/page_number/section via app.db.get_conn().
    """
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages)
    _ = get_settings()  # config available here
    # TODO: embeddings = embed_texts([c.content for c in chunks])
    # TODO: write documents + chunks rows, return document_id
    raise NotImplementedError(
        f"Extracted {len(pages)} pages -> {len(chunks)} chunks. Now wire embedding + DB insert."
    )
