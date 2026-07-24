"""CLI: ingest one or more PDFs.

    python -m scripts.ingest path/to/doc.pdf [more.pdf ...]

Thin wrapper over app.ingestion so you can test Stage 1 from the terminal
before any HTTP or frontend exists (build-order step 1).
"""

import sys
from pathlib import Path

from app.ingestion import ingest_pdf


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m scripts.ingest <file.pdf> [file.pdf ...]")
        return 1
    for path in argv:
        p = Path(path)
        if not p.exists():
            print(f"skip (not found): {p}")
            continue
        doc_id = ingest_pdf(str(p), p.name)
        print(f"ingested {p.name} -> {doc_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
