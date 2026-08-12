import os
import re
from typing import Iterable, List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

try:
    from azure.search.documents.indexes.models import HnswVectorSearchAlgorithmConfiguration
except ImportError:  # pragma: no cover - compatibility for older SDKs.
    HnswVectorSearchAlgorithmConfiguration = HnswAlgorithmConfiguration




def _search_config():
    return {
        "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
        "key": os.getenv("AZURE_SEARCH_KEY"),
        "index_name": os.getenv("AZURE_SEARCH_INDEX", "knowledge-base"),
    }


def build_index() -> SearchIndex:
    config = _search_config()
    index_name = config["index_name"]
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="title", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
            analyzer_name="en.microsoft",
        ),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="default-vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswVectorSearchAlgorithmConfiguration(
                name="hnsw-config",
                parameters={"m": 4, "efConstruction": 400, "efSearch": 1000, "metric": "cosine"},
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="default-vector-profile",
                algorithm_configuration_name="hnsw-config",
            )
        ],
    )

    return SearchIndex(name=index_name, fields=fields, vector_search=vector_search)


def create_or_update_index() -> None:
    config = _search_config()
    endpoint = config["endpoint"]
    key = config["key"]
    if not endpoint or not key:
        raise RuntimeError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY must be set before indexing documents.")
    index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    index = build_index()
    index_client.create_or_update_index(index=index)


def _safe_key(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")
    return cleaned or "document"


def build_document(title: str, source: str, content: str, vector: Iterable[float]) -> dict:
    return {
        "id": f"{_safe_key(source)}-{_safe_key(title)}",
        "title": title,
        "source": source,
        "content": content,
        "contentVector": list(vector),
    }


def upload_documents(documents: List[dict]) -> None:
    config = _search_config()
    endpoint = config["endpoint"]
    key = config["key"]
    index_name = config["index_name"]
    if not endpoint or not key:
        raise RuntimeError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY must be set before uploading documents.")
    client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(key))
    result = client.upload_documents(documents=documents)
    failures = [r for r in result if getattr(r, "error_code", None) or getattr(r, "status_code", None) not in (200, 201, 202)]
    if failures:
        raise RuntimeError(f"Failed to upload documents: {failures}")


if __name__ == "__main__":
    create_or_update_index()

    sample_documents = [
        build_document(
            "Platform architecture",
            "internal-platform",
            "This platform combines Azure ML for training and model lifecycle governance with Azure AI Search and Foundry for retrieval and grounding.",
            [0.01] * 1536,
        ),
        build_document(
            "Production hardening",
            "internal-platform",
            "The platform should enforce managed identity, private networking, least privilege, and explicit monitoring for production workloads.",
            [0.02] * 1536,
        ),
    ]

    upload_documents(sample_documents)
    print(f"Indexed {len(sample_documents)} documents in {_search_config()['index_name']}")
