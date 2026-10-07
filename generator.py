"""Optional, strictly grounded text generation for the FAQ chatbot."""

from __future__ import annotations

import os

from rag import FAQ


class GroundedGenerator:
    """Uses Groq only when configured; otherwise returns verified FAQ wording."""

    def __init__(self) -> None:
        self._client = None
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            try:
                from groq import Groq

                self._client = Groq(api_key=api_key)
            except Exception:
                # A missing optional SDK must not make the safe FAQ fallback fail.
                self._client = None
        self._model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    def answer(self, question: str, faq: FAQ, language: str) -> str:
        fallback = faq.answer_ur if language == "urdu" else faq.answer_en
        if self._client is None:
            return fallback
        language_name = "Urdu" if language == "urdu" else "English"
        prompt = (
            f"Answer in {language_name}. Use only the verified FAQ answer below. "
            "Give a complete, concise answer. Do not add facts, instructions, or numbers.\n\n"
            f"Customer question: {question}\n"
            f"Verified FAQ answer: {fallback}"
        )
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=220,
            )
            answer = (completion.choices[0].message.content or "").strip()
            return answer or fallback
        except Exception:
            # Keep the verified answer available when the optional provider is down.
            return fallback
