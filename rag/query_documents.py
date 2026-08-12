import logging
import os

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
except ImportError:  # pragma: no cover - the production runtime requires the Azure Search SDK to be installed.
    AzureKeyCredential = None
    SearchClient = None


logger = logging.getLogger(__name__)


def _search_config():
    return {
        "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
        "key": os.getenv("AZURE_SEARCH_KEY"),
        "index_name": os.getenv("AZURE_SEARCH_INDEX", "knowledge-base"),
    }


def _raise_missing_search_configuration():
    message = (
        "Azure AI Search is not configured for this platform. "
        "Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY, and enable the Azure AI Search service "
        "in the target environment before using retrieval or grounding."
    )
    logger.error(message)
    raise RuntimeError(message)


def search(query: str, top_k: int = 3):
    config = _search_config()
    endpoint = config["endpoint"]
    key = config["key"]
    index_name = config["index_name"]

    if not endpoint or not key or SearchClient is None:
        _raise_missing_search_configuration()

    try:
        client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(key))
        vector = [0.01] * 1536
        results = client.search(
            search_text=query,
            vector_queries=[
                {
                    "kind": "vector",
                    "vector": vector,
                    "k": top_k,
                    "fields": "contentVector",
                }
            ],
            select=["title", "source", "content"],
            top=top_k,
        )
        return list(results)
    except Exception as exc:
        message = (
            "Azure AI Search query failed. Ensure the search service is provisioned, reachable, "
            "and configured with the current endpoint and key."
        )
        logger.exception(message)
        raise RuntimeError(message) from exc


if __name__ == "__main__":
    results = search("How does the platform combine Azure ML and Foundry grounding?", top_k=3)
    for item in results:
        print(item["title"], "-", item["source"])
        print(item["content"])
        print("---")
