import fitz
import psycopg
import pytest

from app.config import get_settings
from app.ingestion import ingest_pdf


def _db_up() -> bool:
    try:
        with psycopg.connect(get_settings().database_url, connect_timeout=2):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_up(), reason="no live Postgres")


def _make_pdf(path, pages=3):
    doc = fitz.open()
    for n in range(pages):
        # Clause label per page: the chunker splits on section boundaries, so
        # without these all pages coalesce into one chunk and the 1-based page
        # assertion can't fire.
        doc.new_page().insert_text(
            (72, 72), f"{n + 1}.1 Ventilation\n" + f"page {n + 1} ventilation clause " * 40
        )
    doc.save(path)
    doc.close()


def test_ingest_writes_document_and_chunks(tmp_path):
    pdf = tmp_path / "sample.pdf"
    _make_pdf(pdf)

    with psycopg.connect(get_settings().database_url) as conn:
        conn.execute("TRUNCATE documents CASCADE")  # chunks cascade from documents
        conn.commit()

    result = ingest_pdf(str(pdf), "sample.pdf")
    assert result.page_count == 3
    assert result.chunk_count > 0

    with psycopg.connect(get_settings().database_url) as conn:
        n, embedded, lo, hi = conn.execute(
            "SELECT count(*), count(embedding), min(page_number), max(page_number)"
            " FROM chunks WHERE document_id = %s",
            (result.document_id,),
        ).fetchone()

    assert n == embedded == result.chunk_count
    assert (lo, hi) == (1, 3)  # 1-based pages; off-by-one here breaks citations


def test_ingest_rejects_textless_pdf(tmp_path):
    """No DB and no API key needed: the ValueError fires before either is
    touched, so this belongs in CI rather than behind the integration mark."""
    import fitz

    from app.ingestion import ingest_pdf

    path = tmp_path / "blank.pdf"
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()
    doc.save(str(path))
    doc.close()

    with pytest.raises(ValueError, match="no extractable text"):
        ingest_pdf(str(path), "blank.pdf")
