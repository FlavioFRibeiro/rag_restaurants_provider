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
    _llm_rerank_select,
    _build_term_index,
    _extract_terms,
    _build_llm_candidates,
    _format_target_terms,
    _get_target_terms,
    _select_rerank_results,
    RERANK_TOP_K,
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

    if reason == "NOT_FOUND":
        llm_provider = get_llm_provider()
        if llm_provider is None:
            print("\nLLM provider not configured.")
            return 0
        term_index = _build_term_index(dishes)
        req_ing, req_tech = _extract_terms(args.question, term_index)
        should_rerank = bool(req_ing or req_tech)
        print(f"\nrerank gating: {should_rerank}")
        if req_ing:
            print(f"required_ingredients: {', '.join(req_ing)}")
        if req_tech:
            print(f"required_techniques: {', '.join(req_tech)}")
        if should_rerank:
            print(f"\nrerank bm25 top {RERANK_TOP_K}:")
            rerank_results = _get_bm25_top_dishes(args.question, retriever, dishes, top_k=RERANK_TOP_K)
            for item in rerank_results:
                print(f"- {item.item.dish_name} | {item.score:.3f}")
            candidates = [item.item for item in rerank_results]
            target_terms = _format_target_terms(_get_target_terms(args.question))
            scores_map, llm_raw, from_cache = _llm_rerank_select(
                args.question,
                candidates,
                retriever,
                llm_provider,
                target_terms=target_terms,
                top_k=RERANK_TOP_K,
                question_id="debug",
            )
            print("\nLLM raw response:")
            print(llm_raw or "")
            print(f"\nLLM cache: {'hit' if from_cache else 'miss'}")
            if scores_map:
                ranked = sorted(scores_map.items(), key=lambda item: item[1], reverse=True)
                print("\nrerank top 5 scores:")
                for name, score in ranked[:5]:
                    print(f"- {name} | {score:.2f}")
                selected = _select_rerank_results(scores_map, rerank_results)
                if selected:
                    print("\nfinal decision: RERANK")
                    print("\n".join(selected))
                    return 0
                print("\nrerank selected empty, falling back to selector.")
            else:
                print("\nrerank invalid, falling back to selector.")

        print("\nselector fallback (bm25 top 20):")
        fallback_results = _get_bm25_top_dishes(args.question, retriever, dishes, top_k=20)
        candidates = _build_llm_candidates(
            dishes,
            [],
            fallback_results,
            max_candidates=20,
        )
        for dish in candidates:
            print(f"- {dish.dish_name}")
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
            print("\nfinal decision: SELECTOR")
            print("\n".join(llm_names))
        else:
            print("\nfinal decision: NOT_FOUND (selector fallback)")
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
