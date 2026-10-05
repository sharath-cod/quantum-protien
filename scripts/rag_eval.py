"""
Measure retrieval quality.   Run from the project root:   python scripts/rag_eval.py

For each test question we list which knowledge file SHOULD be retrieved.
  hit@k  = fraction of questions where the right file is in the top-k results
  MRR    = mean of 1/rank of the first right result (1.0 = always rank 1)
  The 'off-topic' questions must return NOTHING (the system should not invent relevance).
Add your own questions as you add knowledge - this is your regression test.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protein_rag import get_rag  # noqa: E402

CASES = [
    ("why does proline break a helix", "secondary_structure.md"),
    ("what instability index value means the protein is unstable", "sequence_metrics.md"),
    ("what does the isoelectric point tell me", "sequence_metrics.md"),
    ("how does VQE find the minimum energy", "quantum_vqe.md"),
    ("what gates are in the ansatz circuit", "quantum_vqe.md"),
    ("why are only 6 qubits used", "quantum_vqe.md"),
    ("what does the noise robustness test do", "quantum_vqe.md"),
    ("which diseases can the app flag", "misfolding_diseases.md"),
    ("why is Huntington's disease flagged", "misfolding_diseases.md"),
    ("what does E22K mean", "mutations.md"),
    ("which mutation makes amyloid beta aggregate faster", "mutations.md"),
    ("what is Levinthal's paradox", "protein_folding_basics.md"),
    ("which lab assay detects amyloid fibrils", "lab_followups.md"),
    ("how do I check my predicted structure experimentally", "lab_followups.md"),
    ("what is the sequence of insulin A chain", "reference_proteins.md"),
    ("how is the risk score from 0 to 100 calculated", "app_guide.md"),
    ("how do I save my analysis", "app_guide.md"),
    ("what is Chou-Fasman", "secondary_structure.md"),
]
OFF_TOPIC = ["who won the cricket world cup", "best pizza recipe in Bengaluru", "how do I fix my wifi router"]

K = 4
rag = get_rag()
hits_at_k, rr, misses = 0, 0.0, []
for q, want in CASES:
    res = rag.search(q, k=K)
    files = [h["source"] for h in res]
    if want in files:
        hits_at_k += 1
        rr += 1 / (files.index(want) + 1)
    else:
        misses.append((q, want, files))

print(f"Questions: {len(CASES)}   chunks indexed: {len(rag.chunks)}")
print(f"hit@{K} = {hits_at_k}/{len(CASES)} = {hits_at_k/len(CASES):.0%}")
print(f"MRR    = {rr/len(CASES):.3f}")
for q, want, got in misses:
    print(f"  MISS: {q!r}\n        wanted {want}, got {got}")

bad = [q for q in OFF_TOPIC if rag.search(q, k=K)]
print(f"off-topic correctly ignored: {len(OFF_TOPIC)-len(bad)}/{len(OFF_TOPIC)}")
for q in bad:
    print(f"  FALSE POSITIVE: {q!r} -> {[h['source'] for h in rag.search(q, k=K)]}")
