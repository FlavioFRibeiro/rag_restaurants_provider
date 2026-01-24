from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import requests

from ..utils.logging import get_logger

logger = get_logger(__name__)


class LLMProvider:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


@dataclass
class OpenAIProvider(LLMProvider):
    api_key: str
    model: str
    timeout: int = 60

    def generate(self, prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 512,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


@dataclass
class GeminiProvider(LLMProvider):
    api_key: str
    model: str = "gemini-1.5-flash"
    timeout: int = 60

    def generate(self, prompt: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 512},
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return ""
        return (parts[0].get("text") or "").strip()


def get_llm_provider() -> Optional[LLMProvider]:
    if os.getenv("NO_LLM", "").lower() in {"1", "true", "yes"}:
        return None

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAIProvider(api_key=openai_key, model=model)

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        return GeminiProvider(api_key=gemini_key, model=model)

    return None
