"""Load documents and split them into overlapping word-window chunks.

Metadata comes from an optional header in each file:
    title: ...
    date: 2019-05-01
    author: ...
    ---
    body text...
Replace/extend this loader for PDFs, HTML, tables, code blocks (see design.md).
"""
import json
import re
from pathlib import Path

from .config import CHUNK_OVERLAP, CHUNK_WORDS, DOCS_DIR


def parse_doc(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    meta = {"title": path.stem, "date": None, "author": None}
    if "\n---\n" in text[:600]:
        head, text = text.split("\n---\n", 1)
        for line in head.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip()
    return {"doc_id": path.stem, "meta": meta, "text": text}


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP):
    """Paragraph-aware: pack paragraphs into ~size words, carry `overlap` words."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], []
    for p in paras:
        words = p.split()
        if cur and len(cur) + len(words) > size:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
        cur.extend(words)
        while len(cur) > size * 1.5:  # very long paragraph: hard split
            chunks.append(" ".join(cur[:size]))
            cur = cur[size - overlap:]
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def load_chunks():
    out = []
    for path in sorted(Path(DOCS_DIR).glob("**/*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        d = parse_doc(path)
        for i, c in enumerate(chunk_text(d["text"])):
            out.append({
                "chunk_id": f'{d["doc_id"]}#{i}',
                "doc_id": d["doc_id"],
                "text": c,
                **d["meta"],
            })
    return out


def save_chunks(chunks, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
