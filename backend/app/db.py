"""Lazy Postgres connection pool. Pool is created on first use so the app
(and the health check / tests) can import without a live database."""

from collections.abc import Iterator
from contextlib import contextmanager

from psycopg_pool import ConnectionPool

from app.config import get_settings

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(get_settings().database_url, min_size=1, max_size=10)
    return _pool


@contextmanager
def get_conn() -> Iterator["ConnectionPool"]:
    with get_pool().connection() as conn:
        yield conn
