"""Offline: chunk -> embed -> save. Run: python -m librarian.build_index"""
import numpy as np
from sentence_transformers import SentenceTransformer

from .chunk import load_chunks, save_chunks
from .config import CHUNKS_PATH, EMB_PATH, EMBED_MODEL


def main():
    chunks = load_chunks()
    if not chunks:
        raise SystemExit("No documents found in data/docs (.txt or .md).")
    save_chunks(chunks, CHUNKS_PATH)
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode([c["text"] for c in chunks], batch_size=64,
                       normalize_embeddings=True, show_progress_bar=True)
    np.save(EMB_PATH, emb.astype("float32"))
    print(f"Indexed {len(chunks)} chunks from {len({c['doc_id'] for c in chunks})} docs")


if __name__ == "__main__":
    main()
