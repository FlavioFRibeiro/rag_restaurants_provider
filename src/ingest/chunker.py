from __future__ import annotations

from dataclasses import dataclass

from ..utils.text import chunk_text


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_id: int


def chunk_document(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[Chunk]:
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [Chunk(text=chunk, chunk_id=i) for i, chunk in enumerate(chunks)]
