from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.ingest.menu_parser import parse_menu_items, title_score, _clean_line, _normalize_text
from src.ingest.pdf_loader import load_pdf_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug title selection for a menu PDF.")
    parser.add_argument("--pdf-path", required=True, help="Path to a menu PDF.")
    return parser.parse_args(argv)


def _detect_mode(text: str) -> str:
    normalized = _normalize_text(text)
    lines = [_clean_line(line) for line in normalized.splitlines() if line.strip()]
    has_headers = any(line.lower() in {"ingredienti", "tecniche"} for line in lines)
    return "structured" if has_headers else "narrative"


def _parse_with_flags(text: str, source_file: str, scoring: bool, join: bool):
    old_scoring = os.environ.get("PARSER_TITLE_SCORING")
    old_join = os.environ.get("PARSER_JOIN_TITLE")
    os.environ["PARSER_TITLE_SCORING"] = "1" if scoring else "0"
    os.environ["PARSER_JOIN_TITLE"] = "1" if join else "0"
    try:
        return parse_menu_items(text, source_file)
    finally:
        if old_scoring is None:
            os.environ.pop("PARSER_TITLE_SCORING", None)
        else:
            os.environ["PARSER_TITLE_SCORING"] = old_scoring
        if old_join is None:
            os.environ.pop("PARSER_JOIN_TITLE", None)
        else:
            os.environ["PARSER_JOIN_TITLE"] = old_join


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        raise SystemExit(f"File not found: {pdf_path}")

    text = load_pdf_text(pdf_path)
    mode = _detect_mode(text)
    items_old = _parse_with_flags(text, pdf_path.as_posix(), scoring=False, join=False)
    items_new = _parse_with_flags(text, pdf_path.as_posix(), scoring=True, join=True)

    print(f"mode: {mode}")
    print(f"items: {len(items_new)}")
    for idx, dish in enumerate(items_new):
        score = title_score(dish.dish_name)
        print(f"- {dish.dish_name} | score={score}")
        if idx < len(items_old) and items_old[idx].dish_name != dish.dish_name:
            print(f"  upgrade: {items_old[idx].dish_name} -> {dish.dish_name}")

    with_delimiter = [
        dish for dish in items_new if " - " in dish.dish_name or ": " in dish.dish_name
    ]
    pct_delimiter = (len(with_delimiter) / len(items_new) * 100.0) if items_new else 0.0
    print(f"titles_with_delimiter_pct: {pct_delimiter:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
