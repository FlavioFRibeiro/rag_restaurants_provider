from __future__ import annotations

import re
from typing import Dict, List


_QUOTE_RE = re.compile(r"[\"']([^\"']+)[\"']")
_STOPWORDS = {
    "quali",
    "quale",
    "piatti",
    "piatto",
    "sono",
    "preparati",
    "preparato",
    "usando",
    "utilizzando",
    "utilizzate",
    "includono",
    "include",
    "contengono",
    "contiene",
    "con",
    "senza",
    "non",
    "ma",
    "che",
    "sia",
}


def _tokenize(query: str) -> List[str]:
    tokens: List[str] = []
    for raw in query.lower().split():
        token = raw.strip(".,;:!?()[]{}\"'")
        if token:
            tokens.append(token)
    return tokens


def extract_keywords(query: str) -> Dict[str, List[str]]:
    keywords: List[str] = []
    seen: set[str] = set()

    for phrase in _QUOTE_RE.findall(query):
        normalized = " ".join(phrase.lower().split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            keywords.append(normalized)

    tokens = [token for token in _tokenize(query) if len(token) >= 4 and token not in _STOPWORDS]
    for token in tokens:
        if token not in seen:
            seen.add(token)
            keywords.append(token)

    for idx in range(len(tokens) - 1):
        if tokens[idx] in _STOPWORDS or tokens[idx + 1] in _STOPWORDS:
            continue
        bigram = f"{tokens[idx]} {tokens[idx + 1]}"
        if bigram not in seen:
            seen.add(bigram)
            keywords.append(bigram)

    return {"keywords": keywords}
