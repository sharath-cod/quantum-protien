"""
STAGE 3 of RAG (INDEX) - a from-scratch BM25 keyword index.

BM25 is the classic ranking function behind search engines.  Intuition:
  * a chunk scores higher when it contains the query words      (term frequency)
  * rare words count more than common ones                      (inverse document frequency)
  * long chunks are slightly penalised so they don't win by size (length normalisation)
"""
import math
import re
from collections import Counter

STOPWORDS = set("""
a an and are as at be but by can do does for from had has have how i if in into is it its
me my of on or our so than that the their them then there these they this to was we were
what when where which while who whom why will with would you your about also been being
both each few more most other some such only own same too very just should could
""".split())

_IRREGULAR = {"helices": "helix", "helixes": "helix", "analyses": "analysis"}


def stem(w):
    """Tiny suffix stripper so 'mutations'/'mutation', 'folding'/'folds' match."""
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    for suf in ("ations", "ation", "ing", "ed", "es", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    if w.endswith("s") and len(w) > 3 and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [stem(w) for w in words if len(w) > 1 and w not in STOPWORDS]


class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.doc_tf, self.doc_len, self.df = [], [], Counter()
        self.avg_len, self.n = 0.0, 0

    def fit(self, texts):
        self.doc_tf = [Counter(tokenize(t)) for t in texts]
        self.doc_len = [sum(tf.values()) for tf in self.doc_tf]
        self.n = len(texts)
        self.avg_len = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.df = Counter()
        for tf in self.doc_tf:
            self.df.update(tf.keys())
        return self

    def _idf(self, term):
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query):
        """BM25 score of the query against every chunk (list aligned with fit order)."""
        q_terms = tokenize(query)
        out = [0.0] * self.n
        for i, tf in enumerate(self.doc_tf):
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1))
            s = 0.0
            for t in q_terms:
                f = tf.get(t, 0)
                if f:
                    s += self._idf(t) * f * (self.k1 + 1) / (f + norm)
            out[i] = s
        return out
