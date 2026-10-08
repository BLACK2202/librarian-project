import json
import re
from pathlib import Path

EVAL_PATH = Path("data/eval.jsonl")


def load_eval(path=EVAL_PATH):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def toks(s):
    return set(re.findall(r"\w+", s.lower()))


def is_hit(chunk, item, min_overlap=0.5):
    """A retrieved chunk is a hit if it comes from the gold doc AND contains
    >= min_overlap of the gold passage's tokens. Unanswerable items have no gold."""
    if item.get("source_doc") != chunk["doc_id"]:
        return False
    g = toks(item["source_passage"])
    return bool(g) and len(g & toks(chunk["text"])) / len(g) >= min_overlap
