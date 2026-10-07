"""Small multilingual semantic FAQ retriever with in-memory conversation context."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


_URDU = re.compile(r"[\u0600-\u06ff]")
_TOKENS = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class FAQ:
    id: str
    question_en: str
    question_ur: str
    answer_en: str
    answer_ur: str


class SemanticFAQRetriever:
    def __init__(self, faqs: Sequence[FAQ], model_name: str) -> None:
        self.faqs = tuple(faqs)
        self.model = SentenceTransformer(model_name, device="cpu")
        corpus = [self._document(faq) for faq in self.faqs]
        self.vectors = self._normalise(self.model.encode(corpus, normalize_embeddings=True))

    @staticmethod
    def _normalise(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)

    @staticmethod
    def _document(faq: FAQ) -> str:
        return "passage: " + " ".join((faq.question_en, faq.answer_en, faq.question_ur, faq.answer_ur))

    def retrieve(self, query: str) -> Optional[FAQ]:
        vector = self._normalise(self.model.encode(["query: " + query], normalize_embeddings=True))[0]
        index = int(np.argmax(self.vectors @ vector))
        return self.faqs[index]


def load_faqs(path: str) -> List[FAQ]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [FAQ(**row) for row in rows]


def select_language(message: str, requested: Optional[str]) -> str:
    if requested in {"english", "urdu"}:
        return requested
    return "urdu" if _URDU.search(message) else "english"


def contextual_query(history: Sequence[Dict[str, str]], message: str, language: str) -> str:
    """Use only short follow-ups; stored customer content remains unchanged."""

    prior_users = [item["content"] for item in history if item["role"] == "user"]
    if len(_TOKENS.findall(message)) > 3 or not prior_users:
        return message
    prefix = "پچھلا سوال: " if language == "urdu" else "Previous question: "
    suffix = "\nموجودہ سوال: " if language == "urdu" else "\nCurrent follow-up: "
    return prefix + prior_users[-1] + suffix + message
