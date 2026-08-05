import pytest

from app import ingestion

DIM = 1024


class FakeResult:
    """Mimics voyageai's response object: only .embeddings is used."""

    def __init__(self, n: int, dim: int = DIM):
        self.embeddings = [[0.1] * dim for _ in range(n)]


def test_batches_and_returns_one_vector_per_input(monkeypatch):
    calls = []

    class FakeClient:
        def embed(self, batch, model, input_type):
            calls.append(len(batch))
            return FakeResult(len(batch))

    monkeypatch.setattr(ingestion, "_voyage_client", lambda: FakeClient())

    out = ingestion.embed_texts([f"chunk {i}" for i in range(300)])

    assert len(out) == 300
    assert calls == [128, 128, 44]


def test_retries_transient_error_then_succeeds(monkeypatch):
    monkeypatch.setattr(ingestion.time, "sleep", lambda s: None)
    attempts = {"n": 0}

    class RateLimitError(Exception):
        pass  # name must match _RETRYABLE_ERROR_NAMES

    class FlakyClient:
        def embed(self, batch, model, input_type):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RateLimitError("429")
            return FakeResult(len(batch))

    monkeypatch.setattr(ingestion, "_voyage_client", lambda: FlakyClient())

    assert len(ingestion.embed_texts(["a", "b"])) == 2
    assert attempts["n"] == 3


def test_auth_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(ingestion.time, "sleep", lambda s: None)
    attempts = {"n": 0}

    class AuthenticationError(Exception):
        pass  # not in _RETRYABLE_ERROR_NAMES -> must raise immediately

    class DeadClient:
        def embed(self, batch, model, input_type):
            attempts["n"] += 1
            raise AuthenticationError("invalid api key")

    monkeypatch.setattr(ingestion, "_voyage_client", lambda: DeadClient())

    with pytest.raises(AuthenticationError):
        ingestion.embed_texts(["a"])
    assert attempts["n"] == 1  # failed fast, no 30s of pointless backoff


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(ingestion.time, "sleep", lambda s: None)
    attempts = {"n": 0}

    class RateLimitError(Exception):
        pass

    class AlwaysFailing:
        def embed(self, batch, model, input_type):
            attempts["n"] += 1
            raise RateLimitError("429")

    monkeypatch.setattr(ingestion, "_voyage_client", lambda: AlwaysFailing())

    with pytest.raises(RateLimitError):
        ingestion.embed_texts(["a"])
    assert attempts["n"] == ingestion._MAX_EMBED_RETRIES


def test_dimension_mismatch_raises(monkeypatch):
    class WrongDimClient:
        def embed(self, batch, model, input_type):
            return FakeResult(len(batch), dim=1536)  # e.g. OpenAI 3-small

    monkeypatch.setattr(ingestion, "_voyage_client", lambda: WrongDimClient())

    with pytest.raises(ValueError, match="EMBEDDING_DIM"):
        ingestion.embed_texts(["a"])


def test_empty_input_makes_no_api_call(monkeypatch):
    def boom():
        raise AssertionError("should not construct a client for empty input")

    monkeypatch.setattr(ingestion, "_voyage_client", boom)
    assert ingestion.embed_texts([]) == []
