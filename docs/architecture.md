# Architecture

The system is three independent pipelines. Keeping them separate means when an
answer is bad you can ask *which stage* failed — retrieval or generation —
instead of guessing.

```mermaid
flowchart TD
    subgraph Ingestion["1 · Ingestion (offline, on upload)"]
        A[PDF] --> B[Extract text<br/>pymupdf]
        B --> C[Chunk + metadata<br/>page, section]
        C --> D[Embed<br/>Voyage / OpenAI]
        D --> E[(Postgres + pgvector<br/>chunks + tsvector + HNSW)]
    end

    subgraph Retrieval["2 · Retrieval (per question)"]
        Q[User question] --> V[Vector search<br/>cosine / HNSW]
        Q --> K[Keyword search<br/>full-text tsvector]
        V --> R[Reciprocal Rank Fusion]
        K --> R
        E -.-> V
        E -.-> K
    end

    subgraph Generation["3 · Generation"]
        R --> P[Build cited prompt]
        P --> L[Claude<br/>streaming]
        L --> ANS[Cited answer -> client]
    end
```

## Data model
- **documents** — one row per PDF (filename, title, page_count, metadata).
- **chunks** — one row per chunk: content, `embedding vector(1024)`, page_number,
  section, and a generated `content_tsv` for full-text search. HNSW index on the
  embedding, GIN index on the tsvector.

## Request flow
- `POST /upload` → ingestion pipeline → rows in `documents` + `chunks`.
- `POST /chat` → hybrid retrieval → cited, streamed generation.

## Why hybrid
Vector search handles paraphrase ("thermal performance" ≈ "U-values"); keyword
search nails exact strings ("Part L", "Approved Document B"). RRF fuses both.
The comparison lives in `scripts/evaluate.py`.
