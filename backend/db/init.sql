-- Runs automatically on first container start (empty data volume).
-- To re-run after changes: `docker compose down -v` then `docker compose up -d db`.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per uploaded PDF.
CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    TEXT NOT NULL,
    title       TEXT,
    page_count  INT,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- One row per chunk. This is what retrieval searches over.
-- IMPORTANT: vector(1024) must match EMBEDDING_DIM in .env.
--   voyage-3            -> 1024
--   openai 3-small      -> 1536
--   openai 3-large      -> 3072
-- If you change the model, change this dimension AND rebuild the volume.
CREATE TABLE IF NOT EXISTS chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT  NOT NULL,              -- order within the document
    content      TEXT NOT NULL,              -- the chunk text
    embedding    vector(1024),              -- semantic vector (nullable until embedded)
    page_number  INT,                        -- for citations
    section      TEXT,                       -- heading/clause, for citations
    token_count  INT,
    -- Generated column powers keyword / hybrid search with no extra write logic:
    content_tsv  tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Approximate-nearest-neighbour index: makes vector search fast at scale.
-- cosine ops pairs with normalised embeddings (voyage/openai are normalised).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Full-text index for the keyword half of hybrid search.
CREATE INDEX IF NOT EXISTS chunks_content_tsv
    ON chunks USING gin (content_tsv);

CREATE INDEX IF NOT EXISTS chunks_document_id
    ON chunks (document_id);
