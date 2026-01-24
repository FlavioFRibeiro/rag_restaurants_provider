from __future__ import annotations

import argparse
from pathlib import Path

from src.ingest.menu_parser import parse_menu_items, _clean_line, _normalize_text
from src.ingest.pdf_loader import load_pdf_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for menu title parsing.")
    parser.add_argument("--pdf-paths", nargs="+", required=True, help="Paths to menu PDFs.")
    return parser.parse_args(argv)


def _has_delimiter_line(text: str) -> bool:
    normalized = _normalize_text(text)
    for line in normalized.splitlines():
        cleaned = _clean_line(line)
        if " - " in cleaned or ": " in cleaned:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    failed = False
    for path_str in args.pdf_paths:
        pdf_path = Path(path_str)
        if not pdf_path.exists():
            print(f"missing: {pdf_path}")
            failed = True
            continue
        text = load_pdf_text(pdf_path)
        items = parse_menu_items(text, pdf_path.as_posix())
        if len(items) == 0:
            print(f"fail: no dishes parsed for {pdf_path}")
            failed = True
            continue
        if any(item.dish_name.endswith(".") for item in items):
            print(f"fail: dish title ends with '.' in {pdf_path}")
            failed = True
        if _has_delimiter_line(text):
            if not any(" - " in item.dish_name for item in items):
                print(f"fail: no titles with ' - ' in {pdf_path}")
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
