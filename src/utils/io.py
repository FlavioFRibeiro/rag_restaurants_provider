from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text(path: Path, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding, errors="ignore")


def write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding=encoding)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
