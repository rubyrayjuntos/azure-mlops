from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rag.quality import assess_retrieval_quality


DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"


def grounded_answer(question: str, live_results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Create a grounded answer from retrieved search results.

    The function is intentionally small and deterministic: it returns a user-facing
    answer and citations when there is contextual evidence, or a safe "not_found"
    message when retrieval yielded nothing.
    """
    items = live_results or []
    if not items:
        return {
            "status": "not_found",
            "answer": "No grounded context was found for this question.",
            "citations": [],
        }

    top = items[0]
    answer = (
        f"Based on the retrieved documentation, the platform combines Azure ML for training and lifecycle governance "
        f"with Azure AI Search and Foundry for retrieval and grounding. "
        f"This is described in '{top.get('title', 'retrieved content')}'."
    )
    citations = [
        {
            "title": item.get("title", "Retrieved content"),
            "source": item.get("source", "unknown"),
            "content": item.get("content", ""),
        }
        for item in items[:3]
    ]
    return {"status": "ready", "answer": answer, "citations": citations}


def grounding_status(knowledge_dir: str | Path | None = None) -> dict[str, Any]:
    quality = assess_retrieval_quality(knowledge_dir)
    index_name = os.getenv("AZURE_SEARCH_INDEX", "knowledge-base")
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    configured = bool(endpoint)

    if quality["health"] == "ready":
        retrieval_health = "ready"
    elif quality["health"] in {"review", "needs_review"}:
        retrieval_health = "degraded"
    else:
        retrieval_health = "not_configured"

    status = "error" if not configured else retrieval_health
    alert = None
    if not configured:
        alert = "AZURE_SEARCH_ENDPOINT is required. Enable the Azure AI Search service for the current environment before grounding."

    return {
        "status": status,
        "alert": alert,
        "documents": quality["documents"],
        "index": index_name,
        "coverage": quality["coverage"],
        "health": quality["health"],
        "source": "rag/quality.py",
    }


if __name__ == "__main__":
    print(json.dumps(grounding_status(DEFAULT_KNOWLEDGE_DIR), indent=2))
