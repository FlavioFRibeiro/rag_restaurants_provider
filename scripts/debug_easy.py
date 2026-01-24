from __future__ import annotations

import argparse
from pathlib import Path

from src.rag.bm25_retriever import BM25DishRetriever, MenuItem
from src.rag.easy_query_parser import extract_keywords
from src.rag.pipeline import _match_dish
from src.utils.io import read_jsonl


def load_dishes(path: Path) -> list[MenuItem]:
    dishes: list[MenuItem] = []
    if not path.exists():
        return dishes
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
    return dishes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug EASY retrieval and filtering.")
    parser.add_argument("--question", required=True, help="Question to debug.")
    parser.add_argument("--index-dir", required=True, help="Index directory (contains menu_dishes.jsonl).")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    index_dir = Path(args.index_dir)
    menu_path = index_dir.parent / "menu_dishes.jsonl"
    dishes = load_dishes(menu_path)
    if not dishes:
        print("No dishes found at", menu_path)
        return 1

    keywords = extract_keywords(args.question).get("keywords", [])
    print("Keywords:", keywords)

    bm25 = BM25DishRetriever(dishes)
    candidates = bm25.search(args.question, top_k=args.top_k)

    print("\nTop candidates:")
    for item in candidates:
        print(f"- {item.item.dish_name} (score={item.score:.4f})")

    print("\nMatched dishes:")
    matched = [item.item.dish_name for item in candidates if _match_dish(item.item, keywords)]
    if matched:
        for name in matched:
            print(name)
    else:
        print("NOT_FOUND")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
