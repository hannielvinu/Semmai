"""Neural / lexical semantic similarity.

Backends (chosen automatically, best first):
  * ``sbert``  - Sentence-BERT (``paraphrase-multilingual-MiniLM-L12-v2``)
                 when the ``sentence-transformers`` package and the model are
                 available (multilingual, as in the project design).
  * ``tfidf``  - character n-gram (2-4) vectors + cosine similarity, a
                 language-independent fallback that needs no download.

The similarity score is a *soft* signal in Semmai: the proof engine makes
the hard decisions, similarity catches drift that the symbolic layer cannot
represent.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

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
        self._hv = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=2 ** 18,
                                     alternate_sign=False, norm="l2")

    @lru_cache(maxsize=20000)
    def _embed(self, text: str):
        if self.backend == "sbert":
            v = self._model.encode([text], normalize_embeddings=True)[0]
            return np.asarray(v, dtype=np.float32)
        return self._hv.transform([text.lower()])

    def score(self, a: str, b: str) -> float:
        ea, eb = self._embed(a), self._embed(b)
        if self.backend == "sbert":
            return float(np.dot(ea, eb))
        return float(ea.multiply(eb).sum())
