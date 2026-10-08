"""Generation eval: correctness, faithfulness, citation accuracy, IDK behaviour,
latency (p50/p95) and cost. LLM-as-judge: SPOT-CHECK 20 judgements by hand and
report the agreement in your report.
Run: python -m eval.generation_eval [--n 100]"""
import argparse
import json
import re
import statistics as st
import time

from eval.common import load_eval
from librarian.config import JUDGE_MODEL
from librarian.generate import answer, client
from librarian.retrieve import Retriever


def judge(prompt):
    r = client().messages.create(model=JUDGE_MODEL, max_tokens=200,
                                 messages=[{"role": "user", "content": prompt}])
    m = re.search(r"\b(YES|NO)\b", r.content[0].text.upper())
    return m.group(1) == "YES" if m else False


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()
    items = load_eval()[: args.n]
    R = Retriever()
    res = []
    for it in items:
        t0 = time.perf_counter()
        ret = R.retrieve(it["question"])
        out = answer(it["question"], ret)
        lat = time.perf_counter() - t0
        said_idk = "i don't know" in out["answer"].lower()
        row = {"id": it.get("id"), "lat": lat, "cost": out["cost_usd"], "idk": said_idk,
               "answerable": bool(it.get("source_doc"))}
        if row["answerable"] and not said_idk:
            ctx = "\n".join(c["text"] for c in ret["chunks"])
            row["correct"] = judge(f"Question: {it['question']}\nReference answer: {it['answer']}\n"
                                   f"Candidate answer: {out['answer']}\nDoes the candidate agree with the reference? Reply YES or NO.")
            row["faithful"] = judge(f"Sources:\n{ctx}\n\nAnswer: {out['answer']}\n"
                                    "Is EVERY claim in the answer supported by the sources? Reply YES or NO.")
            cited = [ret["chunks"][c["n"] - 1]["text"] for c in out["citations"]]
            row["cite_ok"] = bool(cited) and judge(
                "Cited passages:\n" + "\n".join(cited) + f"\n\nAnswer: {out['answer']}\n"
                "Do the cited passages actually say what the answer claims? Reply YES or NO.")
        elif row["answerable"]:
            row["correct"] = False  # said IDK on an answerable question
        res.append(row)
    ans = [r for r in res if r["answerable"]]
    done = [r for r in ans if "faithful" in r]
    unans = [r for r in res if not r["answerable"]]
    summary = {
        "n": len(res),
        "answer_correctness": sum(r.get("correct", False) for r in ans) / max(1, len(ans)),
        "faithfulness": sum(r["faithful"] for r in done) / max(1, len(done)),
        "citation_accuracy": sum(r["cite_ok"] for r in done) / max(1, len(done)),
        "idk_when_should": sum(r["idk"] for r in unans) / max(1, len(unans)) if unans else None,
        "latency_p50_s": st.median(r["lat"] for r in res),
        "latency_p95_s": pct([r["lat"] for r in res], 0.95),
        "cost_per_request_usd": sum(r["cost"] for r in res) / len(res),
    }
    summary["cost_per_100x_usd"] = summary["cost_per_request_usd"] * 100
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "rows": res}, open("eval/generation_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
