import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
INDEX_DIR = ROOT / "data" / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
EMB_PATH = INDEX_DIR / "embeddings.npy"

# Decisions to TEST and justify in design.md, not defaults to keep.
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", 200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 40))

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
GEN_MODEL = os.getenv("GEN_MODEL", "claude-haiku-4-5-20251001")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-haiku-4-5-20251001")

TOP_K_FIRST = int(os.getenv("TOP_K_FIRST", 20))   # candidates before reranking
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", 5))    # chunks sent to the model
# Rerank score below this => "I don't know". Tune on your eval set.
IDK_THRESHOLD = float(os.getenv("IDK_THRESHOLD", -3.0))

# Price per million tokens (USD). Update from the pricing page for your model.
PRICE_IN = float(os.getenv("PRICE_IN", 1.0))
PRICE_OUT = float(os.getenv("PRICE_OUT", 5.0))
