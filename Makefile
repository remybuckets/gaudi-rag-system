.PHONY: db-up db-down api ingest evaluate test lint fmt

db-up:        ## start Postgres + pgvector
	docker compose up -d db

db-down:
	docker compose down

api:          ## run the FastAPI backend locally
	cd backend && uvicorn app.main:app --reload

ingest:       ## ingest a PDF: make ingest PDF=path/to/file.pdf
	cd backend && python -m scripts.ingest $(PDF)

evaluate:     ## run the vector-vs-hybrid retrieval comparison
	cd backend && python -m scripts.evaluate

test:
	cd backend && pytest -q

lint:
	cd backend && ruff check .

fmt:
	cd backend && ruff format .
