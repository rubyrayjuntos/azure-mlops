import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT / "knowledge"


def _document_count() -> int:
    if not KNOWLEDGE_DIR.exists():
        return 0
    count = 0
    for path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        if text.strip():
            count += 1
    return count


def rag_health() -> dict[str, object]:
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    key = os.getenv("AZURE_SEARCH_KEY")
    index_name = os.getenv("AZURE_SEARCH_INDEX", "knowledge-base")
    docs = _document_count()
    model = (
        os.getenv("AZURE_OPENAI_MODEL")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or os.getenv("FOUNDRY_MODEL")
        or os.getenv("MODEL_NAME")
    )
    latency_ms = os.getenv("AI_RUNTIME_LATENCY_MS")

    status = "error"
    alert = None
    if endpoint and key:
        status = "ready" if docs > 0 else "empty"
    else:
        alert = "AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY are required. Enable the Azure AI Search service in the target environment."

    parsed_latency = None
    if latency_ms is not None and latency_ms.strip():
        try:
            parsed_latency = int(float(latency_ms))
        except ValueError:
            parsed_latency = None

    return {
        "status": status,
        "index": index_name,
        "documents": docs,
        "endpoint_configured": bool(endpoint),
        "key_configured": bool(key),
        "alert": alert,
        "knowledge_dir": str(KNOWLEDGE_DIR),
        "model": model,
        "latency_ms": parsed_latency,
    }


if __name__ == "__main__":
    print(json.dumps(rag_health(), indent=2))
