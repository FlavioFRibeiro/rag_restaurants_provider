from __future__ import annotations

from typing import Optional
from pathlib import Path

import re
import json
from datetime import datetime

from ..config import get_env
from ..utils.logging import get_logger
from ..utils.io import ensure_dir, read_jsonl
from .llm import LLMProvider
from .prompts import build_prompt, build_llm_selector_prompt
from .bm25_retriever import BM25DishRetriever, MenuItem, MenuItemWithScore
from .retriever import RetrievedChunk, Retriever

logger = get_logger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = get_env(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


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


_BULLET_PREFIX_RE = re.compile(r"^\\s*(?:[-*•]|\\d+[.)])\\s*")


def _split_answer_lines(answer: str) -> list[str]:
    if not answer:
        return []
    if answer.strip().upper() == "NOT_FOUND":
        return []
    return [line.strip() for line in answer.splitlines() if line.strip()]

def _llm_cache_key(question_id: str | None, candidate_names: list[str]) -> str:
    qid = question_id or "unknown"
    candidates_json = json.dumps(candidate_names, ensure_ascii=False, separators=(",", ":"))
    return f"{qid}|{candidates_json}"


def _load_llm_cache(retriever: Retriever) -> tuple[dict[str, dict], Path]:
    if hasattr(retriever, "_llm_cache"):
        return retriever._llm_cache, retriever._llm_cache_path  # type: ignore[attr-defined]
    path = retriever.index_dir.parent / "llm_cache.jsonl"
    cache: dict[str, dict] = {}
    if path.exists():
        for record in read_jsonl(path):
            key = record.get("key")
            if not key:
                key = _llm_cache_key(record.get("question_id"), record.get("candidate_names") or [])
            cache[key] = record
    retriever._llm_cache = cache  # type: ignore[attr-defined]
    retriever._llm_cache_path = path  # type: ignore[attr-defined]
    return cache, path


def _append_llm_cache(path: Path, record: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def _format_llm_candidates(candidates: list[MenuItem]) -> str:
    blocks: list[str] = []
    for idx, dish in enumerate(candidates, start=1):
        ingredients = ", ".join(dish.ingredients)
        techniques = ", ".join(dish.techniques)
        blocks.append(
            f"{idx}.\n"
            f"Name: {dish.dish_name}\n"
            f"Ingredients: {ingredients}\n"
            f"Techniques: {techniques}"
        )
    return "\n\n".join(blocks)


def _validate_llm_response(raw: str, candidates: list[str]) -> list[str]:
    if not raw:
        return []
    candidate_set = set(candidates)
    seen: set[str] = set()
    valid: list[str] = []
    for line in raw.splitlines():
        name = _BULLET_PREFIX_RE.sub("", line).strip()
        if not name or name not in candidate_set:
            continue
        if name in seen:
            continue
        seen.add(name)
        valid.append(name)
    return valid


def _llm_select(
    question: str,
    candidates: list[MenuItem],
    retriever: Retriever,
    llm_provider: Optional[LLMProvider],
    question_id: str | None = None,
) -> tuple[list[str] | None, str | None, bool]:
    if not llm_provider or not candidates:
        return None, None, False
    candidate_names = [dish.dish_name for dish in candidates]
    cache, cache_path = _load_llm_cache(retriever)
    key = _llm_cache_key(question_id, candidate_names)
    if key in cache:
        cached = cache[key]
        return cached.get("valid_names") or None, cached.get("response_raw"), True

    formatted = _format_llm_candidates(candidates)
    prompt = build_llm_selector_prompt(question, formatted)
    try:
        response_raw = llm_provider.generate(prompt)
    except Exception as exc:
        logger.warning("LLM reasoning failed: %s", exc)
        return None, None, False

    valid_names = _validate_llm_response(response_raw, candidate_names)
    record = {
        "key": key,
        "question_id": question_id,
        "candidate_names": candidate_names,
        "response_raw": response_raw,
        "valid_names": valid_names,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    cache[key] = record
    _append_llm_cache(cache_path, record)
    return valid_names or None, response_raw, False

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


def _get_bm25_top_dishes(
    question: str, retriever: Retriever, dishes: list[MenuItem], top_k: int = 20
) -> list[MenuItemWithScore]:
    bm25 = _get_bm25(retriever, dishes)
    results = bm25.search(question, top_k=top_k) if bm25 else []
    seen: set[str] = set()
    deduped: list[MenuItemWithScore] = []
    for item in results:
        name = item.item.dish_name
        if name in seen:
            continue
        seen.add(name)
        deduped.append(item)
    return deduped


def _build_llm_candidates(
    dishes: list[MenuItem],
    deterministic_names: list[str],
    bm25_results: list[MenuItemWithScore],
    max_candidates: int = 20,
) -> list[MenuItem]:
    dish_by_name: dict[str, MenuItem] = {}
    for dish in dishes:
        if dish.dish_name and dish.dish_name not in dish_by_name:
            dish_by_name[dish.dish_name] = dish

    candidates: list[MenuItem] = []
    seen: set[str] = set()
    for name in deterministic_names:
        dish = dish_by_name.get(name)
        if not dish or name in seen:
            continue
        seen.add(name)
        candidates.append(dish)
        if len(candidates) >= max_candidates:
            return candidates

    for item in bm25_results:
        name = item.item.dish_name
        if name in seen:
            continue
        seen.add(name)
        candidates.append(item.item)
        if len(candidates) >= max_candidates:
            break

    return candidates


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
    question_id: str | None = None,
) -> str:
    deterministic_answer = answer_easy(question, retriever, top_k=30)
    deterministic_names = _split_answer_lines(deterministic_answer)
    llm_reasoning = _env_flag("ENABLE_LLM_REASONING", False)

    if llm_reasoning:
        call_llm = False
        if not deterministic_names:
            call_llm = True
        elif len(deterministic_names) > 8:
            call_llm = True
        else:
            return deterministic_answer

        if call_llm:
            dishes = _load_menu_dishes(retriever)
            bm25_results = _get_bm25_top_dishes(question, retriever, dishes, top_k=20)
            candidates = _build_llm_candidates(
                dishes,
                deterministic_names,
                bm25_results,
                max_candidates=20,
            )
            if candidates:
                llm_names, _, _ = _llm_select(
                    question,
                    candidates,
                    retriever,
                    llm_provider,
                    question_id=question_id,
                )
            else:
                llm_names = None

            if not deterministic_names:
                return "\n".join(llm_names) if llm_names else "NOT_FOUND"

            if not llm_names:
                return deterministic_answer

            deterministic_set = set(deterministic_names)
            llm_set = set(llm_names)
            if llm_set.issubset(deterministic_set):
                if len(deterministic_names) > 8:
                    return "\n".join(llm_names)
                return deterministic_answer
            return deterministic_answer

    if deterministic_answer != "NOT_FOUND":
        return deterministic_answer
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
