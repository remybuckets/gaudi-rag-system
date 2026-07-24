"""CLI: compare vector-only vs hybrid retrieval on your test queries.

    python -m scripts.evaluate

This produces the table that proves 'hybrid beats vector-only' — an acceptance
criterion, and the single most convincing thing in your README. Fill TEST_SET
with (question, expected_substring) pairs once your corpus is loaded.
"""

from app.retrieval import hybrid_search, vector_search

# (question, a substring you KNOW appears in the correct passage)
TEST_SET: list[tuple[str, str]] = [
    # ("What is the minimum ceiling height for a habitable room?", "2.1 m"),
    # ("Which Approved Document covers fire safety?", "Approved Document B"),
]


def hit_at_k(hits, needle: str) -> bool:
    return any(needle.lower() in h.content.lower() for h in hits)


def main() -> int:
    if not TEST_SET:
        print("TEST_SET is empty — add (question, expected_substring) pairs first.")
        return 1
    vec_hits = sum(hit_at_k(vector_search(q, 8), n) for q, n in TEST_SET)
    hyb_hits = sum(hit_at_k(hybrid_search(q), n) for q, n in TEST_SET)
    total = len(TEST_SET)
    print(f"{'method':<14}{'hit@8':>8}")
    print(f"{'vector-only':<14}{vec_hits}/{total:>6}")
    print(f"{'hybrid (RRF)':<14}{hyb_hits}/{total:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
