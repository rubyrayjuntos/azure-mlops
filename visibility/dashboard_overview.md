# Hybrid platform control plane overview

This repository now supports a layered view of operational status across the full AI platform:

## Layer 1: MLOps lifecycle
- training runs
- evaluation status
- model registry promotion
- deployment and monitoring status

## Layer 2: Retrieval plane
- Azure AI Search service health
- index readiness
- document count and freshness
- retrieval quality coverage

## Layer 3: Grounding and AI runtime
- content availability
- chunk coverage quality
- grounding health
- Foundry or agent runtime state

## Core governing principle
Azure ML remains the control plane for deterministic model lifecycle operations. Azure AI Search and Foundry become the control plane for retrieval, grounding, and response generation. Together they form the hybrid platform operating surface.

## Operational signals to watch
- workflow and deployment health
- drift or monitoring alerts
- retrieval quality coverage
- index/document readiness
- agent or Foundry runtime availability
- prompt and grounding quality end-to-end
