"""Generation pipeline: (question + retrieved chunks) -> streamed, cited answer.

Stage 3. Prompt assembly is here; the streaming Claude call has a TODO with the
shape you need. Check the current Anthropic docs for the exact streaming +
Citations request format before implementing.
"""
from collections.abc import Iterator

from app.config import get_settings
from app.retrieval import Hit

SYSTEM_PROMPT = (
    "You are a research assistant for UK architects. Answer ONLY using the "
    "provided source passages. If the passages do not contain the answer, say "
    "so. Cite the passage id(s) you used for each claim, e.g. [S2]."
)


def build_context(hits: list[Hit]) -> str:
    """Render retrieved chunks as labelled sources the model can cite."""
    blocks = []
    for i, h in enumerate(hits, start=1):
        loc = f"(p.{h.page_number}" + (f", {h.section}" if h.section else "") + ")"
        blocks.append(f"[S{i}] {loc}\n{h.content}")
    return "\n\n".join(blocks)


def answer_stream(question: str, hits: list[Hit]) -> Iterator[str]:
    """Stream the answer token-by-token.

    TODO: call Claude with streaming, yielding text deltas. Sketch:

        import anthropic
        client = anthropic.Anthropic(api_key=get_settings().anthropic_api_key)
        with client.messages.stream(
            model=get_settings().generation_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user",
                       "content": f"Sources:\n{build_context(hits)}\n\nQuestion: {question}"}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    Consider Anthropic's native Citations feature for structured source spans.
    """
    _ = get_settings()
    raise NotImplementedError("Wire up the streaming Claude call (see docstring).")
