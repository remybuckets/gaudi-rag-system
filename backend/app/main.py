"""FastAPI entrypoint. Thin HTTP layer over the three pipelines.

Heavy imports (pymupdf, anthropic, voyage) are done lazily inside endpoints so
the app boots and /health works without every dependency installed.
"""

import os
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="gaudi-rag-system", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload", status_code=201)
async def upload(file: UploadFile):
    """Ingest a PDF. Stage 1."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a .pdf file.")

    from app.ingestion import ingest_pdf

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        while data := await file.read(1024 * 1024):
            tmp.write(data)
        tmp.close()

        try:
            result = await run_in_threadpool(ingest_pdf, tmp.name, file.filename)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

        return {
            "document_id": result.document_id,
            "page_count": result.page_count,
            "chunk_count": result.chunk_count,
        }
    finally:
        tmp.close()
        os.unlink(tmp.name)


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    """Retrieve (Stage 2) then stream a cited answer (Stage 3)."""
    # from app.retrieval import hybrid_search
    # from app.generation import answer_stream
    # hits = hybrid_search(req.question)
    # return StreamingResponse(answer_stream(req.question, hits), media_type="text/plain")
    raise HTTPException(501, "Chat not implemented yet — see app/retrieval.py + app/generation.py")
