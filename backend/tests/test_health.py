from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_upload_rejects_non_pdf():
    r = client.post("/upload", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert r.status_code == 400
