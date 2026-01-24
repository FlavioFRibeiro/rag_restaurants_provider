from __future__ import annotations

from typing import Optional

import re

from ..utils.logging import get_logger
from ..utils.io import read_jsonl
from .llm import LLMProvider
from .prompts import build_prompt
from .bm25_retriever import BM25DishRetriever, MenuItem
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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\\w\\+]+", text.lower())


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


def _build_term_index(dishes: list[MenuItem]) -> dict[str, list[str]]:
    ingredients: set[str] = set()
    techniques: set[str] = set()
    for dish in dishes:
        for item in dish.ingredients:
            norm = _normalize_match(item)
            if norm and len(norm) >= 4:
                ingredients.add(norm)
        for item in dish.techniques:
            norm = _normalize_match(item)
            if norm and len(norm) >= 4:
                techniques.add(norm)
    return {
        "ingredients": sorted(ingredients, key=len, reverse=True),
        "techniques": sorted(techniques, key=len, reverse=True),
    }


def _collect_terms(question: str, known_terms: list[str]) -> list[str]:
    normalized = _normalize_match(question)
    matches = [term for term in known_terms if term and term in normalized]
    filtered: list[str] = []
    for term in matches:
        if any(term != other and term in other for other in matches):
            continue
        filtered.append(term)
    return filtered


def _extract_terms(question: str, term_index: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    ingredients = _collect_terms(question, term_index.get("ingredients", []))
    techniques = _collect_terms(question, term_index.get("techniques", []))
    ingredients = [term for term in ingredients if not any(term in tech for tech in techniques)]
    return ingredients, techniques


def _extract_negatives(question: str, terms: list[str]) -> list[str]:
    normalized = _normalize_match(question)
    negations = {"senza", "escludendo", "evitando", "non"}
    negatives: list[str] = []
    for term in terms:
        if not term:
            continue
        pattern = re.compile(r"(senza|evitando|escludendo|non)[^\n]{0,60}" + re.escape(term))
        if pattern.search(normalized):
            negatives.append(term)
    return negatives


def _get_bm25(retriever: Retriever, dishes: list[MenuItem]) -> BM25DishRetriever | None:
    if hasattr(retriever, "_bm25_cache"):
        return retriever._bm25_cache  # type: ignore[attr-defined]
    bm25 = BM25DishRetriever(dishes) if dishes else None
    retriever._bm25_cache = bm25  # type: ignore[attr-defined]
    return bm25


def _match_dish(
    dish: MenuItem,
    required_ingredients: list[str],
    required_techniques: list[str],
    forbidden_ingredients: list[str],
    forbidden_techniques: list[str],
) -> bool:
    ingredient_texts = [_normalize_match(item) for item in dish.ingredients]
    technique_texts = [_normalize_match(item) for item in dish.techniques]

    for term in forbidden_ingredients:
        if any(term in text for text in ingredient_texts):
            return False
    for term in forbidden_techniques:
        if any(term in text for text in technique_texts):
            return False

    for term in required_ingredients:
        if not any(term in text for text in ingredient_texts):
            return False
    for term in required_techniques:
        if not any(term in text for text in technique_texts):
            return False
    return True


def answer_easy(question: str, retriever: Retriever, top_k: int = 30) -> str:
    dishes = _load_menu_dishes(retriever)
    if not dishes:
        return "NOT_FOUND"
    term_index = _build_term_index(dishes)
    required_ingredients, required_techniques = _extract_terms(question, term_index)
    if not required_ingredients and not required_techniques:
        return "NOT_FOUND"
    forbidden_ingredients = _extract_negatives(question, term_index.get("ingredients", []))
    forbidden_techniques = _extract_negatives(question, term_index.get("techniques", []))
    required_ingredients = [term for term in required_ingredients if term not in forbidden_ingredients]
    required_techniques = [term for term in required_techniques if term not in forbidden_techniques]
    bm25 = _get_bm25(retriever, dishes)
    candidates = [item.item for item in (bm25.search(question, top_k=top_k) if bm25 else [])]
    if not candidates:
        candidates = dishes
    answers: list[str] = []
    seen: set[str] = set()
    for dish in candidates:
        if _match_dish(
            dish,
            required_ingredients,
            required_techniques,
            forbidden_ingredients,
            forbidden_techniques,
        ):
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
