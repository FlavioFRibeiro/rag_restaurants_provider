from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    data_raw: Path
    data_questions: Path
    data_ground_truth: Path
    data_processed: Path
    runs: Path


def load_env(env_path: Path | None = None) -> None:
    if env_path is None:
        env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def get_env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


def get_paths(root: Path | None = None) -> Paths:
    if root is None:
        root = Path(__file__).resolve().parents[1]
    data = root / "data"
    return Paths(
        root=root,
        data=data,
        data_raw=data / "raw",
        data_questions=data / "questions",
        data_ground_truth=data / "ground_truth",
        data_processed=data / "processed",
        runs=root / "runs",
    )
