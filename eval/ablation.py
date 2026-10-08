"""Retrieval ablation: bm25 | vector | hybrid | hybrid+rerank.
Run: python -m eval.ablation   (writes eval/ablation.md)"""
import statistics as st

from eval.common import is_hit, load_eval
from librarian.retrieve import Retriever

KS = [1, 3, 5, 10]
CONFIGS = [("bm25", "bm25", False), ("vector", "vector", False),
           ("hybrid", "hybrid", False), ("hybrid+rerank", "hybrid", True)]


def main():
    items = [i for i in load_eval() if i.get("source_doc")]  # answerable only
    R = Retriever()
    rows = []
    for name, mode, rerank in CONFIGS:
        hits = {k: 0 for k in KS}
        lat = []
        for it in items:
            ret = R.retrieve(it["question"], mode=mode, rerank=rerank, k_final=max(KS))
            lat.append(ret["latency_s"] * 1000)
            for k in KS:
                hits[k] += any(is_hit(c, it) for c in ret["chunks"][:k])
        n = len(items)
        rows.append((name, [hits[k] / n for k in KS], st.median(lat)))
    md = "| config | " + " | ".join(f"recall@{k}" for k in KS) + " | p50 ms |\n"
    md += "|---|" + "---|" * (len(KS) + 1) + "\n"
    for name, rec, lat in rows:
        md += f"| {name} | " + " | ".join(f"{r:.3f}" for r in rec) + f" | {lat:.0f} |\n"
    print(md)
    open("eval/ablation.md", "w").write(md)


if __name__ == "__main__":
    main()
