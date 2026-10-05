"""Try retrieval from a terminal:  python -m protein_rag "why does proline break a helix" """
import sys

from . import get_rag

if "--embed-test" in sys.argv:
    import os
    from .embeddings import HFEmbedder
    v = HFEmbedder(os.environ["HF_API_KEY"]).embed_query("protein misfolding")
    print("embedding OK, dimension =", len(v))
    sys.exit(0)

q = " ".join(sys.argv[1:]) or "what is the instability index"
rag = get_rag()
hits = rag.search(q, k=4)
print(f"\nQUESTION: {q}\n")
if not hits:
    print("(no relevant chunk found)")
for n, h in enumerate(hits, 1):
    print(f"[{n}] score={h['score']}  {h['source']}  |  {h['title']}")
    print("    " + h["text"][:220].replace("\n", " ") + ("..." if len(h["text"]) > 220 else ""))
