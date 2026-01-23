from __future__ import annotations

from typing import Optional

from ..utils.logging import get_logger
from .llm import LLMProvider
from .prompts import build_prompt
from .retriever import RetrievedChunk, Retriever

logger = get_logger(__name__)


def _format_retrieval_only(chunks: list[RetrievedChunk], max_chars: int = 2000) -> str:
    if not chunks:
        return "NOT_FOUND"
    parts: list[str] = []
    total = 0
    for chunk in chunks:
        text = chunk.text.strip()
        if not text:
            continue
        if total + len(text) > max_chars and parts:
            break
        parts.append(text)
        total += len(text)
    if not parts:
        return "NOT_FOUND"
    return "\n---\n".join(parts)


def _postprocess_answer(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return "NOT_FOUND"
    if cleaned.upper() == "NOT_FOUND":
        return "NOT_FOUND"
    return cleaned


def answer_question(
    question: str,
    retriever: Retriever,
    llm_provider: Optional[LLMProvider],
    top_k: int = 6,
) -> str:
    chunks = retriever.search(question, top_k=top_k)
    if not chunks:
        return "NOT_FOUND"

    if llm_provider is None:
        return _format_retrieval_only(chunks)

    prompt = build_prompt(question, [c.text for c in chunks])
    try:
        response = llm_provider.generate(prompt)
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return "NOT_FOUND"
    return _postprocess_answer(response)
