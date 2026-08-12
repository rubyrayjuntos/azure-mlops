from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def collect_knowledge_documents(knowledge_dir: str | Path | None = None) -> list[dict[str, Any]]:
    root = Path(knowledge_dir) if knowledge_dir else DEFAULT_KNOWLEDGE_DIR
    if not root.exists():
        return []

    docs: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        normalized = _normalize_text(text)
        if not normalized:
            continue
        docs.append(
            {
                "source": path.name,
                "title": path.stem,
                "content": normalized,
                "characters": len(normalized),
                "words": len(normalized.split()),
            }
        )
    return docs


def assess_retrieval_quality(knowledge_dir: str | Path | None = None) -> dict[str, Any]:
    docs = collect_knowledge_documents(knowledge_dir)
    if not docs:
        return {
            "documents": 0,
            "unique_sources": 0,
            "avg_words_per_document": 0,
            "avg_chars_per_document": 0,
            "coverage": "empty",
            "health": "needs_content",
        }

    word_counts = [doc["words"] for doc in docs]
    char_counts = [doc["characters"] for doc in docs]
    unique_sources = len({doc["source"] for doc in docs})
    avg_words = sum(word_counts) / len(word_counts)
    avg_chars = sum(char_counts) / len(char_counts)
    min_words = min(word_counts)

    if avg_words >= 120 and avg_chars >= 500 and min_words >= 40:
        coverage = "strong"
        health = "ready"
    elif avg_words >= 60 and avg_chars >= 250:
        coverage = "adequate"
        health = "review"
    else:
        coverage = "thin"
        health = "needs_review"

    return {
        "documents": len(docs),
        "unique_sources": unique_sources,
        "avg_words_per_document": round(avg_words, 2),
        "avg_chars_per_document": round(avg_chars, 2),
        "coverage": coverage,
        "health": health,
    }
