"""Ingestion pipeline: PDF -> text -> chunks (+metadata) -> embeddings -> Postgres.

Stage 1 of the three pipelines. Extraction and chunking are implemented;
embedding + storage have clear TODOs where your DB and API keys plug in.
"""
from dataclasses import dataclass

import fitz  # pymupdf

from app.config import get_settings


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
    target_words: int = 220,     # ~500-800 tokens
    overlap_words: int = 30,     # ~10-15% overlap
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


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of chunk texts. TODO: call Voyage (or OpenAI) in batches.

    import voyageai
    client = voyageai.Client(api_key=get_settings().voyage_api_key)
    result = client.embed(texts, model=get_settings().embedding_model, input_type="document")
    return result.embeddings
    """
    raise NotImplementedError("Wire up the embeddings provider (see docstring).")


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
        f"Extracted {len(pages)} pages -> {len(chunks)} chunks. "
        "Now wire embedding + DB insert."
    )
