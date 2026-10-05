# RAG Workbook — Quantum Protein Folding Solver

This is the hands-on companion to the explanation in chat. Read the code in this order:
`protein_rag/chunker.py` → `bm25.py` → `retriever.py` → `pipeline.py`, then the 3 edits in `app.py`.

## The idea in one picture

```
 BEFORE (no RAG)                          AFTER (RAG)
 question ─────────────► LLM              question ─► SEARCH kb/ ─► top passages ─┐
 (+ analysis numbers)                     question + analysis numbers + passages ─┴─► LLM ─► answer + sources
```

The LLM never "learns" your files. We **find** the right paragraphs and **paste** them into the prompt each time.

## Where each RAG stage lives

| Stage | What happens | File |
|---|---|---|
| 1 Load | read every `.md` in `kb/` | `chunker.py → load_documents` |
| 2 Chunk | cut by `##` heading, max ~900 chars, small overlap | `chunker.py → load_chunks` |
| 3 Index | BM25 keyword index built at server start | `bm25.py → BM25Index` |
| 4 Retrieve | score all chunks, keep the best, drop weak ones | `retriever.py → Retriever.search` |
| 5 Augment | number the passages, add citation rules | `pipeline.py → build_reference_block` |
| 6 Generate | your existing `_hf_chat(...)` call | `app.py` (unchanged) |

## Exercises (do these, you will learn faster than by reading)

1. **See retrieval without any LLM.** Run `python -m protein_rag "why does proline break a helix"`. Try 10 questions of your own.
2. **Measure it.** Run `python scripts/rag_eval.py`. Then add 5 of your own questions to `CASES` and see if they pass. A question that fails means: the knowledge is missing, or the wording is too different.
3. **Break it on purpose.** In `bm25.py` set `b=0` or `k1=0.5` and re-run the eval. Notice which questions change rank. This is what "tuning" means.
4. **Change chunk size.** In `chunker.py` set `MAX_CHARS = 300`, then `2000`. Small chunks = precise but lose context. Big chunks = more context but noisy. Re-run the eval and read the chunks with the CLI.
5. **Add knowledge.** Create `kb/chaperones.md` with a `# Title` and `## Heading` sections. Restart the server. Ask the chat about chaperones. No code change needed — that is the whole point of RAG.
6. **See the prompt.** In `app.py`, inside `ai_chat`, add `print(system_context)` right before `_hf_chat(...)`. You will see the exact text the LLM receives.

## Writing good knowledge files

- One idea per `##` section, 40–150 words. The chunker uses headings as boundaries.
- Put the keywords people will actually type in the heading and first sentence.
- State facts plainly. The LLM will repeat them, so wrong text in `kb/` becomes wrong answers.
- Keep numbers that the app computes OUT of `kb/` (they change per analysis); those come from the analysis context.

## Knobs you can turn

| Knob | Where | Effect |
|---|---|---|
| `k` (passages per question) | `app.py` calls `search(..., k=4)` | more = more context, more noise and cost |
| `REL_CUTOFF`, `MIN_TOP_SCORE` | `retriever.py` | higher = stricter, fewer irrelevant passages |
| `MAX_BLOCK_CHARS` | `pipeline.py` | cap on how much text is pasted into the prompt |
| `RAG_RULES` | `pipeline.py` | the instructions that force citations and "I don't know" |
| `RAG_ENABLED=0` | env var | turn RAG off instantly (back to the old behaviour) |
| `RAG_DIR` | env var | point at another knowledge folder |

## Upgrade path: embeddings (meaning search)

BM25 matches **words**. In testing, paraphrased questions still found the right file in the top 4, but ranked it first only about half the time. Embeddings fix this by matching **meaning**.

1. Set `RAG_EMBED=hf` (uses your existing `HF_API_KEY`).
2. Restart. `retriever.py` fuses BM25 ranks and embedding ranks with Reciprocal Rank Fusion.
3. If the embedding call fails, it prints a warning and falls back to BM25, so it cannot break chat.

> The embedding backend (`embeddings.py`) was written but **not tested against the live Hugging Face service**. Run `python -m protein_rag --embed-test` on your machine first. If your HF plan or the endpoint changed, the fix is the URL/model in that one file.

A heavier alternative for later: a local model (`sentence-transformers`) — but it needs ~1 GB of RAM and will not fit Render's free tier.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Answer says "knowledge base doesn't cover this" | no chunk matched | check with the CLI; add or reword knowledge |
| Sources look unrelated | threshold too loose or chunks too big | raise `MIN_TOP_SCORE`, shrink `MAX_CHARS` |
| No Sources box in the UI | RAG failed at boot | look for `RAG disabled:` in the server log |
| Edited `kb/` but nothing changed | index is built at start | restart the server |
| Model ignores the excerpts | small model (8B) following rules loosely | shorten `RAG_RULES`, lower `k`, or try a stronger model via `HF_MODEL` |

## Honest limits of what is here

- The knowledge base is small (8 files, ~60 chunks) and was written for this project. Its quality caps the answer quality.
- The evaluation set was written by the same person who wrote the documents, so scores are optimistic. Build your own questions from what real users ask.
- RAG reduces made-up answers but does not eliminate them; the model can still misread an excerpt.
- The science text is a student-level summary. Verify before citing it in a report.
