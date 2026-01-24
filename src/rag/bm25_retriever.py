from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from rank_bm25 import BM25Okapi


@dataclass(frozen=True)
class MenuItem:
    dish_name: str
    ingredients: List[str]
    techniques: List[str]
    source_file: str
    doc_type: str = "menu_dish"
    restaurant_name: str | None = None


@dataclass(frozen=True)
class MenuItemWithScore:
    item: MenuItem
    score: float


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in text.lower().split():
        token = raw.strip(".,;:!?()[]{}\"'")
        if token:
            tokens.append(token)
    return tokens


def _dish_text(dish: MenuItem) -> str:
    parts: List[str] = [dish.dish_name]
    parts.extend(dish.ingredients)
    parts.extend(dish.techniques)
    return " ".join(parts)


class BM25DishRetriever:
    def __init__(self, dishes: Iterable[MenuItem]) -> None:
        self.dishes = list(dishes)
        corpus = [_tokenize(_dish_text(dish)) for dish in self.dishes]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int = 20) -> List[MenuItemWithScore]:
        if not query or not self.dishes or self.bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        results: List[MenuItemWithScore] = []
        for idx, score in ranked[:top_k]:
            results.append(MenuItemWithScore(item=self.dishes[idx], score=float(score)))
        return results
