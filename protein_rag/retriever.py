"""
STAGE 4 of RAG (RETRIEVE): given a question, return the best chunks.

  BM25 ranking  --+
                  +--> Reciprocal Rank Fusion --> top-k chunks
  dense ranking --+     (only if embeddings are enabled)

Two safety valves keep junk out of the prompt:
  * a relative cut-off  - drop chunks scoring far below the best one
  * an absolute floor   - if even the best chunk barely matches, return nothing
    (the LLM is then told "no knowledge-base passage found" instead of being fed noise)
"""
import numpy as np

from .bm25 import BM25Index
from .chunker import load_chunks

RRF_K = 60            # standard constant for Reciprocal Rank Fusion
REL_CUTOFF = 0.35     # keep hits >= 35% of the top BM25 score
MIN_TOP_SCORE = 1.0   # best BM25 score must reach this, else "nothing relevant"


class Retriever:
    def __init__(self, folder, embedder=None):
        self.folder = folder
        self.chunks = load_chunks(folder)
        texts = [f"{c['title']}. {c['text']}" for c in self.chunks]   # heading helps matching
        self.bm25 = BM25Index().fit(texts)
        self.embedder, self.vectors = embedder, None
        if embedder is not None and self.chunks:
            try:
                self.vectors = embedder.embed_corpus(texts)
            except Exception as e:                      # never let embeddings break the app
                print(f"[rag] embeddings disabled ({e}); using BM25 only")
                self.embedder = None

    def search(self, query, k=4):
        if not self.chunks or not query.strip():
            return []
        bm = np.array(self.bm25.scores(query))
        top_bm = float(bm.max())
        bm_rank = np.argsort(-bm)

        if self.vectors is not None:
            try:
                sims = self.vectors @ self.embedder.embed_query(query)
                dn_rank = np.argsort(-sims)
                fused = {}
                for rank, i in enumerate(bm_rank[:20]):
                    if bm[i] > 0:
                        fused[i] = fused.get(i, 0) + 1 / (RRF_K + rank + 1)
                for rank, i in enumerate(dn_rank[:20]):
                    fused[i] = fused.get(i, 0) + 1 / (RRF_K + rank + 1)
                order = sorted(fused, key=fused.get, reverse=True)[:k]
                return [self._hit(i, fused[i], bm[i], float(sims[i])) for i in order]
            except Exception as e:
                print(f"[rag] dense search failed ({e}); falling back to BM25")

        if top_bm < MIN_TOP_SCORE:
            return []
        hits = []
        for i in bm_rank[:k]:
            if bm[i] >= REL_CUTOFF * top_bm and bm[i] > 0:
                hits.append(self._hit(i, float(bm[i]), float(bm[i]), None))
        return hits

    def _hit(self, i, score, bm25, dense):
        c = self.chunks[i]
        return {**c, "score": round(float(score), 4), "bm25": round(float(bm25), 3),
                "dense": None if dense is None else round(dense, 3)}
