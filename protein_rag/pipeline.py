"""
STAGE 5 of RAG (AUGMENT): turn retrieved chunks into prompt text for the LLM.
(GENERATE - the LLM call itself - stays in app.py via _hf_chat.)
"""
import os
import re

from .bm25 import STOPWORDS, tokenize
from .retriever import Retriever

_DEICTIC = re.compile(r"\b(this|it|its|that|these|my|here|result|results|analysis|protein|sequence)\b", re.I)
MAX_BLOCK_CHARS = 3600

RAG_RULES = """
HOW TO USE THE KNOWLEDGE-BASE EXCERPTS
- General science and "how does this app work" facts must come from the numbered excerpts. Cite them inline like [1] or [2].
- Numbers about THIS protein (energy, risk score, percentages) come from the analysis context, not from the excerpts.
- If the excerpts do not contain the answer, say so plainly ("the knowledge base doesn't cover this") and then give only a clearly-labelled general answer. Never invent citations.
- The quantum part is a simulated toy model and the disease risk is a screening hint, not a diagnosis. Say so when it matters.
""".strip()


def _content_tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


def build_query(question, context=None, history=None):
    """
    Turn the raw chat question into a better search query.
    A bare follow-up like "why?" has no searchable words, so we borrow the previous
    user question; a question about "this protein" gets the analysis keywords added.
    """
    q = question.strip()
    parts = [q]
    if len(_content_tokens(q)) <= 3 and history:
        prev = [h.get("content", "") for h in history if h.get("role") == "user"]
        prev = [p for p in prev if p.strip() and p.strip() != q]
        if prev:
            parts.insert(0, prev[-1])
    if context and _DEICTIC.search(q):
        final = context.get("final") or {}
        risk = context.get("disease_risk") or {}
        kw = [final.get("dominant_structure"), final.get("fold_topology"), final.get("stability")]
        kw += [d.get("disease") for d in (risk.get("diseases") or [])[:2]]
        parts.append(" ".join(str(x) for x in kw if x and x != "-"))
    return " ".join(parts)


def build_reference_block(hits, max_chars=MAX_BLOCK_CHARS):
    """Numbered, source-labelled excerpts to paste into the system prompt."""
    if not hits:
        return "KNOWLEDGE-BASE EXCERPTS: (none found for this question)"
    lines, used = ["KNOWLEDGE-BASE EXCERPTS:"], 0
    for n, h in enumerate(hits, 1):
        entry = f"[{n}] ({h['title']})\n{h['text']}"
        if used + len(entry) > max_chars:
            break
        lines.append(entry)
        used += len(entry)
    return "\n\n".join(lines)


def public_sources(hits):
    """What we send to the browser so the UI can show 'Sources'."""
    return [{"n": n, "title": h["title"], "source": h["source"], "score": h["score"]}
            for n, h in enumerate(hits, 1)]


# ---- singleton so the index is built once per server process -----------------
_RAG = None


def get_rag():
    """Return the shared Retriever, or None if RAG is disabled (RAG_ENABLED=0)."""
    global _RAG
    if os.environ.get("RAG_ENABLED", "1") == "0":
        return None
    if _RAG is None:
        folder = os.environ.get("RAG_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb")
        embedder = None
        if os.environ.get("RAG_EMBED", "").lower() == "hf" and os.environ.get("HF_API_KEY"):
            from .embeddings import HFEmbedder
            embedder = HFEmbedder(os.environ["HF_API_KEY"],
                                  cache_path=os.path.join(folder, ".embeddings_cache.json"))
        _RAG = Retriever(folder, embedder)
        print(f"[rag] ready: {len(_RAG.chunks)} chunks from {folder} "
              f"({'hybrid BM25+embeddings' if _RAG.vectors is not None else 'BM25'})")
    return _RAG
