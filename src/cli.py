from __future__ import annotations

import argparse

from .eval.run_eval import main as run_eval_main
from .ingest.build_index import main as build_index_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Datapizza RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build_index", help="Build FAISS index")
    subparsers.add_parser("run_eval", help="Run evaluation")

    args, remaining = parser.parse_known_args(argv)

    if args.command == "build_index":
        return build_index_main(remaining)
    if args.command == "run_eval":
        return run_eval_main(remaining)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
