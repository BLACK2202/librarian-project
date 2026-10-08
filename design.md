# Design
## Corpus and users
<what, who, licence/permission, size, why it needs search>
## Chunking
Current: paragraph-aware, CHUNK_WORDS=200, CHUNK_OVERLAP=40.
Tested sizes: <100 / 200 / 400 words, recall@5 for each> -> chosen: <...> because <...>
## Index choice
Flat numpy matrix + in-memory BM25. Justify (corpus size, no extra infra) and say when it stops scaling.
## Retrieval
BM25 + vector, fused with RRF (k=60), optional metadata filters, cross-encoder rerank.
## Generation
Answers only from sources, [n] citations, "I don't know" if top rerank score < IDK_THRESHOLD (tuned: <value>).
## Known failure cases
<tables, code blocks, cross-references, long documents, multi-part questions... with examples and counts>
