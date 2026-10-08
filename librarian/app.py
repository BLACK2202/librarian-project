"""Service. Run: uvicorn librarian.app:app --host 0.0.0.0 --port 8000"""
import json
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .generate import answer
from .retrieve import Retriever

app = FastAPI(title="The Librarian")
R = Retriever()
LOG = Path("data/requests.jsonl")


class Ask(BaseModel):
    question: str
    filters: dict | None = None


class Feedback(BaseModel):
    question: str
    answer: str
    helpful: bool
    comment: str = ""


@app.post("/ask")
def ask(body: Ask):
    t0 = time.perf_counter()
    ret = R.retrieve(body.question, filters=body.filters)
    out = answer(body.question, ret)
    out["latency_s"] = round(time.perf_counter() - t0, 3)
    out["retrieval_latency_s"] = round(ret["latency_s"], 3)
    out["sources"] = [{"chunk_id": c["chunk_id"], "title": c["title"], "text": c["text"]} for c in ret["chunks"]]
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps({"q": body.question, "latency_s": out["latency_s"], "cost_usd": out["cost_usd"]}) + "\n")
    return out


@app.post("/feedback")
def feedback(fb: Feedback):
    with open("data/feedback.jsonl", "a") as f:
        f.write(json.dumps(fb.model_dump(), ensure_ascii=False) + "\n")
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def home():
    return """<!doctype html><meta charset=utf-8><title>The Librarian</title>
<body style="font-family:sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem">
<h2>The Librarian</h2>
<input id=q style="width:80%;padding:8px" placeholder="Ask a question...">
<button onclick=go()>Ask</button>
<div id=out style="margin-top:1rem;white-space:pre-wrap"></div>
<script>
let last={};
async function go(){
 const q=document.getElementById('q').value;
 document.getElementById('out').textContent='...';
 const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
 const d=await r.json(); last={question:q,answer:d.answer};
 document.getElementById('out').innerHTML=d.answer.replace(/</g,'&lt;')+'\\n\\n<b>Sources</b>\\n'+
  d.sources.map((s,i)=>'['+(i+1)+'] '+s.title+': '+s.text.slice(0,200).replace(/</g,'&lt;')+'...').join('\\n')+
  '\\n\\n<button onclick="fb(true)">Helpful</button> <button onclick="fb(false)">Wrong</button>';
}
async function fb(h){await fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...last,helpful:h})});alert('Thanks');}
</script></body>"""
