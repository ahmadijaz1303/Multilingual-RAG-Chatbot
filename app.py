"""Generic multilingual RAG chatbot API with optional grounded Groq generation."""

from __future__ import annotations

import os
import uuid
from collections import defaultdict
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from generator import GroundedGenerator
from rag import FAQ, SemanticFAQRetriever, contextual_query, load_faqs, select_language


load_dotenv()
app = FastAPI(title="Multilingual RAG Chatbot", version="1.0.0")
_history: Dict[str, List[Dict[str, str]]] = defaultdict(list)
_retriever: Optional[SemanticFAQRetriever] = None
_generator: Optional[GroundedGenerator] = None


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = Field(default=None, max_length=100)
    message: str = Field(..., min_length=1, max_length=4000)
    language: Optional[str] = Field(default=None, pattern="^(english|urdu)$")


def retriever() -> SemanticFAQRetriever:
    global _retriever
    if _retriever is None:
        _retriever = SemanticFAQRetriever(
            load_faqs(os.getenv("FAQ_DATA_PATH", "data/sample_faq.json")),
            os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
        )
    return _retriever


def grounded_answer(faq: FAQ, language: str) -> str:
    return faq.answer_ur if language == "urdu" else faq.answer_en


def generator() -> GroundedGenerator:
    global _generator
    if _generator is None:
        _generator = GroundedGenerator()
    return _generator


@app.post("/chat")
def chat(payload: ChatRequest):
    conversation_id = payload.conversation_id or str(uuid.uuid4())
    history = _history[conversation_id]
    language = select_language(payload.message, payload.language)
    query = contextual_query(history, payload.message, language)
    try:
        faq = retriever().retrieve(query)
    except Exception as error:
        raise HTTPException(status_code=503, detail="Retrieval is unavailable.") from error
    history.append({"role": "user", "content": payload.message})
    answer = generator().answer(payload.message, faq, language)
    history.append({"role": "assistant", "content": answer})
    return {"conversation_id": conversation_id, "message": answer, "language": language}


@app.get("/chat/{conversation_id}")
def history(conversation_id: str):
    return {"conversation_id": conversation_id, "messages": _history.get(conversation_id, [])}
