"""
OPTIONAL dense (meaning-based) retrieval.  OFF by default.

BM25 matches WORDS.  Embeddings match MEANING: "why does the chain clump together"
can find a chunk about "aggregation" even though no word is shared.
An embedding model turns text into a vector; similar meaning = similar direction.

Enable with the environment variable  RAG_EMBED=hf  (re-uses your HF_API_KEY).
If the call fails for any reason the retriever silently falls back to BM25 only,
so turning this on can never break the chat.

NOTE: this backend was written against Hugging Face's hosted feature-extraction
API but has NOT been tested against the live service from the build sandbox.
Test it with:  python -m protein_rag --embed-test
"""
import hashlib
import json
import os

import numpy as np
import requests

DEFAULT_MODEL = os.environ.get("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class HFEmbedder:
    def __init__(self, api_key, model=DEFAULT_MODEL, cache_path=None):
        self.api_key, self.model, self.cache_path = api_key, model, cache_path
        self.url = f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"

    def _call(self, texts):
        r = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=60,
        )
        if r.status_code != 200:
            raise RuntimeError(f"embedding API {r.status_code}: {r.text[:200]}")
        arr = np.array(r.json(), dtype="float32")
        if arr.ndim != 2:
            raise RuntimeError(f"unexpected embedding shape {arr.shape}")
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)

    def embed_corpus(self, texts):
        """Embed all chunks once and cache them on disk (keyed by a hash of the texts)."""
        key = hashlib.sha256(("\n".join(texts) + self.model).encode()).hexdigest()
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path) as f:
                    cached = json.load(f)
                if cached.get("key") == key:
                    return np.array(cached["vectors"], dtype="float32")
            except Exception:
                pass
        vecs = np.vstack([self._call(texts[i:i + 32]) for i in range(0, len(texts), 32)])
        if self.cache_path:
            try:
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
                with open(self.cache_path, "w") as f:
                    json.dump({"key": key, "vectors": vecs.tolist()}, f)
            except OSError:
                pass
        return vecs

    def embed_query(self, text):
        return self._call([text])[0]
