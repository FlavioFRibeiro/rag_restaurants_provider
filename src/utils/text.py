from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    lowered = text.lower().strip()
    cleaned = _PUNCT_RE.sub(" ", lowered)
    collapsed = _WHITESPACE_RE.sub(" ", cleaned)
    return collapsed.strip()


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    clean = text.strip()
    if not clean:
        return []

    chunks: list[str] = []
    start = 0
    length = len(clean)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks
