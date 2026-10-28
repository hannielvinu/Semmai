"""Neural / lexical semantic similarity.

Backends (chosen automatically, best first):
  * ``sbert``  - Sentence-BERT (``paraphrase-multilingual-MiniLM-L12-v2``)
                 when the ``sentence-transformers`` package and the model are
                 available (multilingual, as in the project design).
  * ``tfidf``  - character n-gram (2-4) vectors + cosine similarity, a
                 language-independent fallback that needs no download
                 (uses scikit-learn when installed, otherwise an identical
                 pure-Python implementation - so the demo runs on plain Python).

The similarity score is a *soft* signal in Semmai: the proof engine makes
the hard decisions, similarity catches drift that the symbolic layer cannot
represent.
"""
from __future__ import annotations

import math
import os
from collections import Counter
from functools import lru_cache

from .textfeat import char_wb_ngrams

try:                                    # optional: faster vectoriser
    import numpy as np
    from sklearn.feature_extraction.text import HashingVectorizer
except ImportError:                     # plain-Python mode (no pip installs needed)
    np = None
    HashingVectorizer = None

SBERT_MODEL = os.environ.get("SEMMAI_SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")


class Similarity:
    def __init__(self, backend: str = "auto"):
        self.backend = "tfidf"
        self._model = None
        if backend in ("auto", "sbert") and os.environ.get("SEMMAI_DISABLE_SBERT") != "1":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._model = SentenceTransformer(SBERT_MODEL)
                self.backend = "sbert"
            except Exception:
                if backend == "sbert":
                    raise
        self._hv = None
        if HashingVectorizer is not None:
            self._hv = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=2 ** 18,
                                         alternate_sign=False, norm="l2")

    @lru_cache(maxsize=20000)
    def _embed(self, text: str):
        if self.backend == "sbert":
            v = self._model.encode([text], normalize_embeddings=True)[0]
            return np.asarray(v, dtype=np.float32)
        if self._hv is not None:
            return self._hv.transform([text.lower()])
        c = Counter(char_wb_ngrams(text.lower(), 2, 4))
        n = math.sqrt(sum(v * v for v in c.values())) or 1.0
        return {k: v / n for k, v in c.items()}

    def score(self, a: str, b: str) -> float:
        ea, eb = self._embed(a), self._embed(b)
        if self.backend == "sbert":
            return float(np.dot(ea, eb))
        if isinstance(ea, dict):
            if len(eb) < len(ea):
                ea, eb = eb, ea
            return float(sum(v * eb.get(k, 0.0) for k, v in ea.items()))
        return float(ea.multiply(eb).sum())
