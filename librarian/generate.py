"""Generate a cited answer from retrieved chunks only. Needs ANTHROPIC_API_KEY."""
import re

import anthropic

from .config import GEN_MODEL, IDK_THRESHOLD, PRICE_IN, PRICE_OUT

SYSTEM = """You answer questions using ONLY the numbered sources provided.
Rules:
- After every claim, cite the source number like [1] or [2][3].
- If the sources do not contain the answer, reply exactly: I don't know.
- Do not use outside knowledge. Be concise."""

_client = None


def client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def build_prompt(question, chunks):
    src = "\n\n".join(f"[{i+1}] ({c['title']}) {c['text']}" for i, c in enumerate(chunks))
    return f"Sources:\n{src}\n\nQuestion: {question}"


def cost_usd(in_tok, out_tok):
    return (in_tok * PRICE_IN + out_tok * PRICE_OUT) / 1e6


def answer(question, retrieved, model=GEN_MODEL):
    chunks, scores = retrieved["chunks"], retrieved["scores"]
    if not chunks or (scores and scores[0] is not None and scores[0] < IDK_THRESHOLD):
        return {"answer": "I don't know.", "citations": [], "cost_usd": 0.0, "skipped_llm": True}
    r = client().messages.create(
        model=model, max_tokens=500, system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
    )
    text = r.content[0].text.strip()
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(chunks)})
    return {
        "answer": text,
        "citations": [{"n": n, "chunk_id": chunks[n-1]["chunk_id"], "title": chunks[n-1]["title"]} for n in cited],
        "cost_usd": cost_usd(r.usage.input_tokens, r.usage.output_tokens),
        "in_tokens": r.usage.input_tokens, "out_tokens": r.usage.output_tokens,
        "skipped_llm": False,
    }
