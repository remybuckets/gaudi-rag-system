# gaudi-rag-system

> Production RAG platform that answers questions over UK architecture documents
> (Building Regs, planning policy, standards) with page-level citations.

**Status:** 🚧 Week 1 build in progress. This README is a living deliverable —
the results table and demo go here as they land.

<!-- TODO: architecture diagram (see docs/architecture.md), UI screenshot/gif -->

## Stack
FastAPI · PostgreSQL + pgvector · Claude · Voyage embeddings · Next.js · Docker

See [`docs/architecture.md`](docs/architecture.md) for the three-pipeline design
and data model.

## Quick start
```bash
cp .env.example .env          # then fill in your keys
make db-up                    # start Postgres + pgvector (docker)
cd backend && pip install -e ".[dev]"
make api                      # run the API at http://localhost:8000
```
Ingest and evaluate from the terminal (build-order steps 1–3, before the UI):
```bash
make ingest PDF=data/approved-document-b.pdf
make evaluate                 # vector-only vs hybrid on your test queries
make test
```

## Retrieval results
<!-- Filled by scripts/evaluate.py. This table is the credibility of the project. -->
| method       | hit@8 |
|--------------|-------|
| vector-only  | TBD   |
| hybrid (RRF) | TBD   |

## Repo layout
```
backend/   FastAPI app + pipelines (app/), CLI scripts (scripts/), tests, schema (db/)
frontend/  Next.js chat + upload UI
docs/      architecture + diagram
```

## Corpus
Test documents (Approved Documents, NPPF, a Local Plan) are **not committed** —
they're Crown copyright / OGL and would bloat the repo. Load them locally into
`data/` (gitignored). See the project notes for the sourcing list.

## Roadmap
Tracked as GitHub issues: reranking · multi-document search · conversation memory.

## License
MIT — see [LICENSE](LICENSE).
