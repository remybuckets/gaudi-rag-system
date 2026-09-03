"""Ingestion pipeline: PDF -> text -> chunks (+metadata) -> embeddings -> Postgres.

Stage 1 of the three pipelines. Extraction and chunking are implemented;
embedding + storage have clear TODOs where your DB and API keys plug in.
"""

import hashlib
import logging
import random
import re
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
_BASE_RETRY_DELAY = 1
_MIN_PAGE_CHARS = 50
_MIN_CHUNK_CHARS = 50

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
    skipped: bool = False


@dataclass
class _Line:
    text: str
    page_number: int
    section: str | None


def _labelled_lines(pages: list[tuple[int, str]]) -> list[_Line]:
    """Flatten pages to cleaned lines, each tagged with the clause it sits under.

    Runs across the whole document, not per page, so a clause continuing onto
    the next page keeps its label instead of resetting to None.
    """

    noise = find_repeated_lines(pages)
    out: list[_Line] = []
    current: str | None = None
    for page_number, text in pages:
        for raw in text.splitlines():
            if _is_noise_line(raw) or _normalise(raw) in noise:
                continue
            if _SECTION_RESET.match(raw):
                current = None
            found = detect_section(raw)
            if found:
                current = found
            out.append(_Line(raw.strip(), page_number, current))
    return out


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...]. page_number is 1-based for citations."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        pages.append((i, text))
    doc.close()
    return pages


# Anchored at line start (^) and match, not search: "set out in 6.2 below" must
# not read as a heading, or prose gets shredded into one-line fragments.
_HEADING_PATTERNS = (
    # "Requirement B4", "Requirement B4:" — the strongest signal, so tried first.
    re.compile(r"^\s*(Requirement\s+[A-T]\d{1,2})\b", re.IGNORECASE),
    # "B1", "K2" — a bare part letter + number, alone on its line.
    re.compile(r"^\s*([A-T]\d{1,2})\s*$"),
    # "6.2", "6.2.1 Ventilation openings" — numbered clause opening a line.
    re.compile(r"^\s*(\d{1,2}(?:\.\d{1,3}){1,3})\s+\S"),
    # Appendices end the numbered clauses; without this the last clause number
    # leaks into every appendix and back-matter page.
    re.compile(r"^\s*(Appendix\s+[A-Z])\b"),
    # "Section 1: Ventilation provision" — Part F's real structural boundary.
    # Without it a cover-page requirement label carries through all front matter.
    re.compile(r"^\s*(Section\s+\d{1,2})\b"),
)

# Back matter has no clause numbering, so the last clause seen must be cleared
# rather than inherited — a citation naming a clause for index text is worse
# than no clause at all.
_SECTION_RESET = re.compile(
    r"^\s*(List of approved documents|Index|Contents|Standards referred to|"
    r"Other documents referred to)\b",
    re.IGNORECASE,
)


def detect_section(line: str) -> str | None:
    """Return the clause label a line introduces, or None if it's body text.

    Approved Documents number every clause, so this label is what turns a
    citation from "page 47" into "Requirement B4, page 47".
    """
    if len(line) > 120:
        return None
    for pattern in _HEADING_PATTERNS:
        m = pattern.match(line)
        if m:
            return m.group(1).strip()
    return None


_ROMAN_OR_PAGE_NO = re.compile(r"^(?:[ivxlcdm]{1,7}|\d{1,3})$", re.IGNORECASE)


def _is_noise_line(line: str) -> bool:
    """Page furniture: watermarks, bare page numbers, empty/tab-only lines."""
    s = line.strip()
    if not s:
        return True
    if _ROMAN_OR_PAGE_NO.match(s):
        return True
    parts = s.split()

    return len(parts) >= 6 and all(len(p) == 1 for p in parts)


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _normalise(line: str) -> str:
    """Canonical form for comparing lines across pages."""
    return " ".join(_CONTROL_CHARS.sub(" ", line).split())


def find_repeated_lines(
    pages: list[tuple[int, str]],
    edge_lines: int = 3,
    min_page_fraction: float = 0.5,
) -> set[str]:
    """Lines recurring at the top/bottom of most pages - running headers.

    Every Document has different ones, so theyre learned per document rather
    than pattern-matched
    """

    counts: dict[str, int] = {}
    for _, text in pages:
        lines = [_normalise(ln) for ln in text.splitlines() if _normalise(ln)]
        edges = lines[:edge_lines] + lines[-edge_lines:]
        for line in set(edges):
            counts[line] = counts.get(line, 0) + 1

    threshold = max(2, int(len(pages) * min_page_fraction))
    return {line for line, n in counts.items() if n >= threshold}


def _split_run(
    run: list[_Line],
    target_words: int,
    overlap_words: int,
) -> list[list[_Line]]:
    """Split one clause's lines into windows no larger than target_words.

    Most clauses fit in one window. The ones that don't are the clauses
    followed by a long Table or Diagram (1.58, 3.9, Appendix A): a single
    2000-word chunk embeds to the average of everything in it and then
    retrieves for nothing in particular.
    """
    windows: list[list[_Line]] = []
    current: list[_Line] = []
    words = 0
    for line in run:
        n = len(line.text.split())
        if current and words + n > target_words:
            windows.append(current)
            carry: list[_Line] = []
            carried = 0
            for prev in reversed(current):
                if carried >= overlap_words:
                    break
                carry.insert(0, prev)
                carried += len(prev.text.split())
            current = carry
            words = carried
        current.append(line)
        words += n
    if current:
        windows.append(current)
    return windows


def _emit_run(
    run: list[_Line],
    chunks: list[Chunk],
    idx: int,
    target_words: int,
    overlap_words: int,
) -> int:
    """Turn one clause's lines into Chunks. Returns the next chunk_index."""
    for window in _split_run(run, target_words, overlap_words):
        content = " ".join(ln.text for ln in window)
        if len(content.strip()) < _MIN_CHUNK_CHARS:
            continue
        chunks.append(
            Chunk(
                content=content,
                page_number=window[0].page_number,
                chunk_index=idx,
                section=window[0].section,
                token_count=len(content.split()),
            )
        )
        idx += 1
    return idx


def check_extraction(pages: list[tuple[int, str]], filename: str) -> list[int]:
    empty = [n for n, text in pages if len(text.strip()) < _MIN_PAGE_CHARS]
    if empty:
        logger.warning(
            "%s: %d/%d pages have no extractable text (pages %s) - scanned or image-only?",
            filename,
            len(empty),
            len(pages),
            ", ".join(str(n) for n in empty[:20]),
        )
    return empty


def chunk_pages(
    pages: list[tuple[int, str]],
    target_words: int = 220,  # ~500-800 tokens
    overlap_words: int = 30,  # ~10-15% overlap
) -> list[Chunk]:
    """Chunk on clause boundaries, falling back to word windows inside a clause.

    Consumes _labelled_lines rather than raw page text, so noise filtering,
    header stripping and the cross-page section carry all apply here for free.

    In Approved Documents the clause number sits inline at the start of its
    paragraph, so a change of label *is* a paragraph boundary — splitting on
    it gives semantically whole clauses rather than arbitrary word windows.
    """
    lines = _labelled_lines(pages)
    chunks: list[Chunk] = []
    idx = 0

    run: list[_Line] = []
    for line in lines:
        # Label change closes the run. Compared against the run's own first
        # line, not the previous line, so unlabelled continuation text stays
        # attached to the clause it belongs to.
        if run and line.section != run[0].section:
            idx = _emit_run(run, chunks, idx, target_words, overlap_words)
            run = []
        run.append(line)
    if run:
        idx = _emit_run(run, chunks, idx, target_words, overlap_words)

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
    total = (len(texts) + _EMBED_BATCH_SIZE - 1) // _EMBED_BATCH_SIZE
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


def _file_hash(path: str) -> str:
    """Identity of the file's *contents*, so an edited PDF under the same
    filename is detected as changed rather than skipped."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def ingest_pdf(pdf_path: str, filename: str, force: bool = False) -> IngestResult:
    """Full pipeline for one PDF. Returns the new document id.

    Steps: extract -> chunk -> embed (batched) -> INSERT document + chunks.
    TODO: insert into `documents`, batch-embed, then INSERT chunks with
    embedding/page_number/section via app.db.get_conn().
    """
    content_hash = _file_hash(pdf_path)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, page_count, metadata->>'content_hash' FROM documents WHERE filename = %s",
            (filename,),
        )
        row = cur.fetchone()
        if not force and row is not None and row[2] == content_hash:
            cur.execute("SELECT count(*) FROM chunks WHERE document_id = %s", (row[0],))
            logger.info("Skipping %s: unchanged (document_id=%s)", filename, row[0])
            return IngestResult(str(row[0]), row[1], cur.fetchone()[0], skipped=True)
    pages = extract_pages(pdf_path)
    empty = check_extraction(pages, filename)
    chunks = chunk_pages(pages)
    if len(empty) > len(pages) / 2:
        raise ValueError(
            f"{filename}: {len(empty)}/{len(pages)} pages have no extractable "
            "text - scanned/image-only PDF? Needs OCR before ingestion."
        )
    if not chunks:
        raise ValueError(
            f"{filename}: no extractable text found - {len(pages)} pages "
            "produced no usable chunks after filtering."
        )

    embeddings = embed_texts([c.content for c in chunks])
    title = Path(filename).stem

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE filename =%s", (filename,))
        cur.execute(
            """
            INSERT INTO documents (filename, title, page_count, metadata)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (
                filename,
                title,
                len(pages),
                Jsonb({"source_path": str(pdf_path), "content_hash": content_hash}),
            ),
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
