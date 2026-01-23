from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from ..utils.io import read_json, read_jsonl
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    metadata: dict
    score: float


class Retriever:
    def __init__(self, index_dir: Path, top_k: int = 6, model_name: str | None = None) -> None:
        self.index_dir = index_dir
        self.top_k = top_k
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        self.documents = list(read_jsonl(index_dir / "documents.jsonl"))
        self.embed_config = read_json(index_dir / "embed_config.json")
        self.normalize = bool(self.embed_config.get("normalize", True))
        if model_name is None:
            model_name = self.embed_config.get("model_name", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not query:
            return []
        if not self.documents:
            return []
        k = top_k or self.top_k
        k = min(k, len(self.documents))

        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        ).astype("float32")

        scores, indices = self.index.search(query_vec, k)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self.documents[idx]
            results.append(
                RetrievedChunk(
                    text=doc["text"],
                    metadata=doc["metadata"],
                    score=float(score),
                )
            )
        return results
