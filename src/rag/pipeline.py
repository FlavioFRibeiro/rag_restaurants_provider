from __future__ import annotations

from typing import Optional

from ..utils.logging import get_logger
from ..utils.io import read_jsonl
from .llm import LLMProvider
from .prompts import build_prompt
from .bm25_retriever import BM25DishRetriever, MenuItem
from .easy_query_parser import extract_keywords
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


def _normalize_match(text: str) -> str:
    return " ".join(text.lower().split())


def _load_menu_dishes(retriever: Retriever) -> list[MenuItem]:
    if hasattr(retriever, "_menu_dishes_cache"):
        return retriever._menu_dishes_cache  # type: ignore[attr-defined]
    path = retriever.index_dir.parent / "menu_dishes.jsonl"
    dishes: list[MenuItem] = []
    if path.exists():
        for item in read_jsonl(path):
            dishes.append(
                MenuItem(
                    dish_name=item.get("dish_name", ""),
                    ingredients=item.get("ingredients") or [],
                    techniques=item.get("techniques") or [],
                    source_file=item.get("source_file", ""),
                    doc_type=item.get("doc_type", "menu_dish"),
                    restaurant_name=item.get("restaurant_name"),
                )
            )
    retriever._menu_dishes_cache = dishes  # type: ignore[attr-defined]
    return dishes


def _get_bm25(retriever: Retriever, dishes: list[MenuItem]) -> BM25DishRetriever | None:
    if hasattr(retriever, "_bm25_cache"):
        return retriever._bm25_cache  # type: ignore[attr-defined]
    bm25 = BM25DishRetriever(dishes) if dishes else None
    retriever._bm25_cache = bm25  # type: ignore[attr-defined]
    return bm25


def _match_dish(dish: MenuItem, keywords: list[str]) -> bool:
    if not keywords:
        return False
    ingredient_texts = [_normalize_match(item) for item in dish.ingredients]
    technique_texts = [_normalize_match(item) for item in dish.techniques]
    for keyword in keywords:
        keyword_norm = _normalize_match(keyword)
        if not keyword_norm:
            continue
        if any(keyword_norm in text for text in ingredient_texts):
            return True
        if any(keyword_norm in text for text in technique_texts):
            return True
    return False


def answer_easy(question: str, retriever: Retriever, top_k: int = 30) -> str:
    dishes = _load_menu_dishes(retriever)
    if not dishes:
        return "NOT_FOUND"
    bm25 = _get_bm25(retriever, dishes)
    keywords = extract_keywords(question).get("keywords", [])
    if not keywords:
        return "NOT_FOUND"
    candidates = bm25.search(question, top_k=top_k) if bm25 else []
    if not candidates:
        return "NOT_FOUND"
    answers: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        dish = candidate.item
        if _match_dish(dish, keywords):
            if dish.dish_name not in seen:
                seen.add(dish.dish_name)
                answers.append(dish.dish_name)
    return "\n".join(answers) if answers else "NOT_FOUND"


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
    use_llm: bool = False,
) -> str:
    easy_answer = answer_easy(question, retriever, top_k=30)
    if easy_answer != "NOT_FOUND":
        return easy_answer
    if not use_llm:
        return "NOT_FOUND"

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
