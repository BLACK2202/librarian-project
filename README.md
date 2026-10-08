# The Librarian: RAG over <YOUR CORPUS>

CS496 Project 2. Team: <names>. Live service: <URL>

## Setup (one command)
    pip install -r requirements.txt && export ANTHROPIC_API_KEY=...   # never commit keys

## Run
    python -m librarian.build_index && uvicorn librarian.app:app --port 8000

Open http://localhost:8000 and ask a question.

## Evaluate
    python -m eval.ablation           # recall@k: bm25 / vector / hybrid / +rerank
    python -m eval.generation_eval    # correctness, faithfulness, citations, IDK, p50/p95, cost

## Layout
- `data/docs/` corpus, `data/eval.jsonl` 100+ Q&A pairs with source passages (about 10% unanswerable)
- `librarian/` chunking, index, hybrid retrieval + rerank, cited generation, FastAPI app
- `eval/` retrieval, ablation and generation evals
- `design.md`, `optimisation.md`, `postmortem.md`

## Results
<paste ablation table + generation summary here>

## Contributions
<who did what>
