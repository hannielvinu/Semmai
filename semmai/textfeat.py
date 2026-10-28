"""Character n-gram features identical to scikit-learn's ``analyzer="char_wb"``.

Used so that the similarity scorer and the exported intent model run in
plain Python (no numpy / scikit-learn needed for the demo)."""
from __future__ import annotations

import re

_WS = re.compile(r"\s\s+")


def char_wb_ngrams(text: str, min_n: int, max_n: int) -> list:
    text = _WS.sub(" ", text)
    out = []
    for w in text.split():
        w = " " + w + " "
        w_len = len(w)
        for n in range(min_n, max_n + 1):
            offset = 0
            out.append(w[offset:offset + n])
            while offset + n < w_len:
                offset += 1
                out.append(w[offset:offset + n])
            if offset == 0:
                break
    return out
