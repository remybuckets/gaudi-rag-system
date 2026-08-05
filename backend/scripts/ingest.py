"""CLI: ingest one or more PDFs.

    python -m scripts.ingest path/to/doc.pdf [more.pdf ...]

Thin wrapper over app.ingestion so you can test Stage 1 from the terminal
before any HTTP or frontend exists (build-order step 1).
"""

import sys
from pathlib import Path

from app.db import close_pool
from app.ingestion import ingest_pdf


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m scripts.ingest <file.pdf> [file.pdf ...]")
        return 1
    failures = 0
    try:
        for path in argv:
            p = Path(path)
            if not p.exists():
                print(f"skip (not found): {p}")
                failures += 1
                continue
            try:
                result = ingest_pdf(str(p), p.name)
            except Exception as exc:
                print(f"FAILED {p.name}: {type(exc).__name__}: {exc}")
                failures += 1
                continue
            print(f"ingested {p.name} -> {result.chunk_count} chunks, id={result.document_id}")
    finally:
        close_pool()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
