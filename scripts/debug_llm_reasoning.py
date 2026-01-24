from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_env
from src.rag.llm import get_llm_provider
from src.rag.pipeline import (
    _get_bm25_top_dishes,
    _get_bm25,
    _load_menu_dishes,
    _llm_select,
    _build_llm_candidates,
    _split_answer_lines,
    answer_easy,
)
from src.rag.retriever import Retriever


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug LLM reasoning on Easy pipeline.")
    parser.add_argument("--question", required=True, help="Question text.")
    parser.add_argument("--index-dir", default="data/processed/index", help="Index directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()

    retriever = Retriever(index_dir=Path(args.index_dir))
    dishes = _load_menu_dishes(retriever)
    if not dishes:
        raise SystemExit("No menu dishes found. Build the index first.")

    bm25 = _get_bm25(retriever, dishes)
    bm25_results = bm25.search(args.question, top_k=20) if bm25 else []
    print("bm25 top 20:")
    for item in bm25_results:
        print(f"- {item.item.dish_name} | {item.score:.3f}")

    deterministic = answer_easy(args.question, retriever, top_k=30)
    print("\ndeterministic answer:")
    print(deterministic)
    deterministic_names = _split_answer_lines(deterministic)

    reason = "NO_CALL"
    if not deterministic_names:
        reason = "NOT_FOUND"
    elif len(deterministic_names) > 8:
        reason = "MANY_RESULTS"
    print(f"\nllm_call_reason: {reason}")

    if reason == "NO_CALL":
        print("\nfinal decision: DETERMINISTIC (LLM not called)")
        return 0

    candidates = _build_llm_candidates(
        dishes,
        deterministic_names,
        _get_bm25_top_dishes(args.question, retriever, dishes, top_k=20),
        max_candidates=20,
    )
    print("\ncandidates sent to LLM:")
    for dish in candidates:
        print(f"- {dish.dish_name}")

    llm_provider = get_llm_provider()
    if llm_provider is None:
        print("\nLLM provider not configured.")
        return 0

    llm_names, llm_raw, from_cache = _llm_select(
        args.question,
        candidates,
        retriever,
        llm_provider,
        question_id="debug",
    )
    print("\nLLM raw response:")
    print(llm_raw or "")
    print(f"\nLLM cache: {'hit' if from_cache else 'miss'}")

    if llm_names:
        print("\nfinal decision: LLM")
        print("\n".join(llm_names))
    else:
        print("\nfinal decision: DETERMINISTIC (fallback)")
        print(deterministic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
