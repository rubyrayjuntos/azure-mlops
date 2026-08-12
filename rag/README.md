# Retrieval + vector store + grounding slice

This is the first dev slice for the hybrid LLMOps/RAG platform. It is intentionally small, deterministic, and repeatable:

- Terraform provisions the Azure AI Search service used as the retrieval plane.
- A Python indexing script loads a small knowledge corpus into a search index.
- A grounding script performs a similarity search and returns ranked chunks for Foundry or app-layer orchestration.

This keeps the same operational principle as the MLOps foundation: define infrastructure declaratively where possible, then treat indexing and retrieval logic as code in the repository.

## Components

- `infrastructure/modules/ai-search/` — declares the Azure AI Search service.
- `rag/index_documents.py` — creates or updates the index and writes documents.
- `rag/query_documents.py` — pulls the top results with vector similarity or lexical ranking.
- `rag/healthcheck.py` — produces a deterministic readiness signal for the retrieval and grounding slice.
- `rag/knowledge/` — corpus input for the first slice.

## Production-first operating model

This slice keeps the same discipline as the Azure ML foundation:

- infrastructure is declarative and versioned in Terraform
- environment variables are explicit and generated per environment
- retrieval and grounding logic are code-defined and CI-verifiable
- the healthcheck is read-only and safe to run in automation

The hybrid control plane treats retrieval, grounding, and the AI runtime as first-class operational units instead of ad hoc app features.

## Full hybrid LLMOps control-plane responsibilities

The retrieval slice is not just a search demo. It is the first live component of the runtime control plane that will sit above the Azure ML project-factory foundation.

This platform does not support a silent fallback mode. If the Azure AI Search service is not enabled for the current environment, retrieval and grounding must fail loudly and emit an alert so the correct infrastructure state can be enabled.

### 1. Retrieval plane
- Azure AI Search service health
- index existence and readiness
- document count and index freshness
- query latency and retrieval quality trend

### 2. Grounding plane
- chunk quality and coverage
- document relevance and provenance
- candidate count / top-k quality
- hard failure when the Azure AI Search runtime is missing or misconfigured

### 3. AI runtime plane
- Foundry grounding or agent runtime health
- prompt versioning and grounding configuration
- token and latency telemetry
- policy/safety and traceability review

### 4. Governance boundary
- Azure ML remains the lifecycle authority for model training and promotion
- Azure AI Search + Foundry become the AI runtime and grounding authority
- the control plane surfaces both sides in one operational view

## Suggested deployment flow

1. Run the infrastructure workflow for the target environment to provision the search service.
2. Configure environment variables for the search service endpoint and admin key.
3. Run the indexing script to populate the knowledge base.
4. Run `python3 rag/healthcheck.py` and `python3 -c "from rag.quality import assess_retrieval_quality; import json; print(json.dumps(assess_retrieval_quality(), indent=2))"` to confirm content quality and readiness.
5. Query it from a Foundry grounding layer or API endpoint to retrieve candidate chunks.

## Key quality metrics

The quality model calculates:
- document count
- average words per document
- average characters per document
- unique source coverage
- readiness classification: `ready`, `review`, or `needs_review`

## Environment variables

- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_KEY`
- `AZURE_SEARCH_INDEX`

The Azure AI Search service is intentionally provisioned with local authentication disabled to align with production-grade security assumptions.
