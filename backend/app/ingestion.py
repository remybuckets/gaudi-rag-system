"""Ingestion pipeline: PDF -> text -> chunks (+metadata) -> embeddings -> Postgres.

Stage 1 of the three pipelines. Extraction and chunking are implemented;
embedding + storage have clear TODOs where your DB and API keys plug in.
"""

import logging
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import fitz  # pymupdf
import numpy as np
import voyageai
from psycopg.types.json import Jsonb

from app.config import get_settings
from app.db import get_conn

logger = logging.getLogger(__name__)

# Voyage caps texts-per-request.
_EMBED_BATCH_SIZE = 128
_MAX_EMBED_RETRIES = 5
_BASE_RETRY_DELAY = 5  # CHANGE SOMETHINGS


_RETRYABLE_ERROR_NAMES = frozenset(
    {
        "RateLimitError",
        "ServiceUnavailableError",
        "Timeout",
        "APIConnectionError",
        "APIError",
        "ConnectionError",
    }
)


def _embed_batch_with_retry(client, batch: list[str], model: str, input_type: str):
    """One Voyage call with exponential backoff + jitter on transient errors."""
    delay = _BASE_RETRY_DELAY
    for attempt in range(1, _MAX_EMBED_RETRIES + 1):
        try:
            return client.embed(batch, model=model, input_type=input_type)
        except Exception as exc:
            name = type(exc).__name__
            if name not in _RETRYABLE_ERROR_NAMES or attempt == _MAX_EMBED_RETRIES:
                raise
            sleep_for = delay + random.uniform(0, delay * 0.5)
            logger.warning(
                "Voyage %s (attempt %d/%d); retrying in %.1fs",
                name,
                attempt,
                _MAX_EMBED_RETRIES,
                sleep_for,
            )
            time.sleep(sleep_for)
            delay *= 2

    raise RuntimeError("Unreachable")


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


@dataclass
class IngestResult:
    """What /upload returns and what scripts/ingest.oy prints."""

    document_id: str
    page_count: int
    chunk_count: int


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
    total = len(texts) + _EMBED_BATCH_SIZE - 1
    for n, start in enumerate(range(0, len(texts), _EMBED_BATCH_SIZE), start=1):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        logger.info("Embedding batch %d/%d (%d texts)", n, total, len(batch))
        result = _embed_batch_with_retry(client, batch, settings.embedding_model, input_type)
        vectors.extend(result.embeddings)

    if len(vectors[0]) != settings.embedding_dim:
        raise ValueError(
            f"Got embedding dim {len(vectors[0])}, but EMBEDDING_DIM is "
            f"{settings.embedding_dim}. Update EMBEDDING_DIM and db/init.sql to match."
        )
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Embedded {len(vectors)} vectors for {len(texts)} texts - batching lost data."
        )

    return vectors

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


def ingest_pdf(pdf_path: str, filename: str) -> IngestResult:
    """Full pipeline for one PDF. Returns the new document id.

    Steps: extract -> chunk -> embed (batched) -> INSERT document + chunks.
    TODO: insert into `documents`, batch-embed, then INSERT chunks with
    embedding/page_number/section via app.db.get_conn().
    """
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError(
            f"{filename}: no extractable text found - scanned/image-only PDF? "
            "Needs OCR before ingestion."
        )

    embeddings = embed_texts([c.content for c in chunks])
    title = Path(filename).stem

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (filename, title, page_count, metadata)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (filename, title, len(pages), Jsonb({"source_path": str(pdf_path)})),
        )
        document_id = cur.fetchone()[0]

        cur.executemany(
            """
            INSERT INTO chunks (
                document_id, content, embedding,
                page_number, section, chunk_index, token_count
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    document_id,
                    c.content,
                    np.asarray(v, dtype=np.float32),
                    c.page_number,
                    c.section,
                    c.chunk_index,
                    c.token_count,
                )
                for c, v in zip(chunks, embeddings, strict=True)
            ],
        )
    logger.info(
        "Ingested %s: %d ages -> %d chunks (document_id=%s)",
        filename,
        len(pages),
        len(chunks),
        document_id,
    )
    return IngestResult(str(document_id), len(pages), len(chunks))
