## Three-repo factory model

Not a single-repo project. Casually reading `Azure/mlops-v2`'s top-level README implies
one repo; the real architecture (documented in `Azure/mlops-v2/documentation/structure/README.md`)
is three:

1. `Azure/mlops-v2` — the "project factory" / `sparse_checkout.sh` generator script. Cloned
   locally (`~/mlops-root/mlops-v2`), never forked — it's a one-shot tool, not a dependency.
2. `rubyrayjuntos/mlops-templates` (fork of `Azure/mlops-templates`) — shared reusable
   GitHub Actions workflows (read-yaml, tf-gha-install-terraform, register-environment,
   register-dataset, create-compute, create-endpoint, create-deployment, run-pipeline,
   allocate-traffic). Local clone: `~/mlops-root/mlops-templates`.
3. `rubyrayjuntos/mlops-project-template` (fork of `Azure/mlops-project-template`) — the
   scaffold `sparse_checkout.sh` copies from to generate a new project. Local clone:
   `~/mlops-root/mlops-project-template`.

This repo (`azure-mlops`) is one generated project instance from repo 3.

## Reference-pinning policy (deliberate, asymmetric)

- Cross-references between the two owned forks (`uses: rubyrayjuntos/mlops-templates/...`)
  use `@main`, NOT a pinned commit SHA. Rationale: these are forks under full control;
  pinning would force a coordinated SHA bump in every consumer project every time the
  shared library gets fixed, which defeats the point of a shared factory library.
- References to genuine third-party marketplace actions (`azure/login`, `azure/CLI`,
  `actions/checkout`, `hashicorp/setup-terraform`) ARE pinned to a commit SHA in both forks
  — these are real external dependencies where a floating tag is a real supply-chain risk.

## Identity model

- One Azure AD app/service principal (id `4a03064f-784b-4f7b-a429-46fad02549b5`) serves
  every project — not one-per-project, not one-per-environment.
- Granted `Owner` (not the officially-documented `Contributor`) at the subscription level,
  once. `Owner` is required because Terraform's `azurerm_role_assignment` resources and the
  deploy workflow's own `az role assignment create` calls both need
  `Microsoft.Authorization/roleAssignments/write`, which `Contributor` doesn't grant.
- The only genuinely per-project, per-branch step: an OIDC federated-credential entry on
  that same app (GitHub OIDC subject matching is exact-string, e.g.
  `repo:<owner>/<repo>:ref:refs/heads/<branch>` — no wildcards possible).
- Onboarding automation for a brand new project: `~/mlops-root/scripts/onboard-project.sh
  <owner> <repo> <branch>` adds the federated credential and sets the 3 GitHub secrets
  (`AZURE_CLIENT_ID`/`AZURE_TENANT_ID`/`AZURE_SUBSCRIPTION_ID`, same values for every
  project since it's the same app). Remaining manual step after running it: edit
  `config-infra-dev.yml`/`config-infra-prod.yml` in the new project for a unique
  namespace/postfix (Azure global-uniqueness constraints on storage account names).

## The two-copy trap (see `mem:known_bugs` item 4)

`mlops-project-template` has a top-level `.github/workflows/*.yml` (the template repo's
own CI, never shipped) and nested per-type real sources (what `sparse_checkout.sh` actually
copies). Always verify which one you're editing.

## Workflow-fix propagation rule

Any fix to a `.github/workflows/*.yml` file that also exists in `mlops-project-template`
(true for every pipeline file in this repo) must be applied in BOTH places — the fork's
nested source (so future generated projects inherit it) and this repo's copy (so the
current deployment picks it up). One without the other silently diverges.

## Hybrid Azure ML + Azure AI Foundry architecture for the enterprise platform

This repo is the spine of a production MLOps platform, and the next phase is a hybrid
MLOps + LLMOps/RAG platform. Azure ML remains the control plane for determinism,
lineage, controlled model delivery, and operational governance. Azure AI Foundry becomes
the user-facing AI layer for LLMs, agent workflows, RAG, prompt orchestration, evaluation,
and safety controls.

### Architectural principle

- Azure ML owns the data-science and production ML lifecycle:
  - training pipelines
  - feature engineering
  - experiment tracking
  - model registry and lineage
  - compute orchestration
  - drift and monitoring
  - batch and online model serving
- Azure AI Foundry owns the generative AI and agent layer:
  - model catalog access
  - prompt flow orchestration
  - RAG indexing and retrieval
  - agent/tool calling
  - evaluation and responsible AI controls
- The boundary is not a replacement model; it is a shared platform model where both
  workloads coexist under common governance, identity, networking, and data-plane
  contracts.

### Recommended production layering

1. Infrastructure and platform foundation
   - Terraform provisions workspaces, storage, Key Vault, ACR, VNet, private endpoints,
     and security boundaries.
   - Azure ML workspace is the compute + model lifecycle control plane.
   - AI Foundry hub/project is the generative AI orchestration plane.

2. Data and knowledge plane
   - ADLS Gen2 / Lakehouse stores structured and unstructured enterprise data.
   - Feature store and curated ML datasets remain in Azure ML.
   - Vector DBs (Cosmos DB Mongo vCore, PostgreSQL pgvector, or Milvus) handle embeddings
     and retrieval for RAG.
   - Purview and lineage connectors track both ML and LLM assets.

3. Model development and training plane
   - Azure ML handles classical ML training, sweeps, distributed compute, registration,
     benchmarking, drift evaluation, and deployment promotion gates.
   - Model registry is the authoritative source for production ML artifacts.
   - Foundry is used for selecting and evaluating foundation models, but not as a
     replacement for the custom training lifecycle.

4. Generative AI and RAG plane
   - AI Foundry manages prompt orchestration, retrieval, grounding, safety, and agent
     runtime patterns.
   - RAG pipelines read from shared enterprise knowledge stores and use embeddings plus
     retrieval ranking.
   - Deterministic evaluation is required for prompt quality, groundedness, toxicity,
     latency, and tool-call correctness.

5. Agentic integration plane
   - Agents consume both Azure ML endpoints and Foundry-backed model endpoints.
   - Classical ML predictions provide structured risk or scoring signals.
   - LLMs provide reasoning, summarization, and human-like interaction.
   - Tools, APIs, and business systems remain behind secure identity and RBAC boundaries.

6. Deployment and runtime plane
   - Azure ML serves trained prediction models via batch and online endpoints.
   - AI Foundry serves agentic, chat, and RAG experiences built on LLMs.
   - Hybrid inference is a normal pattern: model-based decisioning + LLM reasoning.

7. Governance, security, and operations
   - Managed identities and RBAC are mandatory.
   - Private networking and egress restrictions apply to prod-grade environments.
   - Responsible AI evaluation, access controls, and review gates are enforced before
     production exposure.
   - Monitoring is unified across model drift, inference health, response quality, and
     operational latency.

### Production architecture pattern

1. Data ingestion -> ADLS / lakehouse
2. Feature engineering -> Azure ML pipelines
3. Training -> Azure ML compute + tracking
4. Model registration -> Azure ML registry
5. Deployment -> Azure ML endpoint
6. RAG / embeddings -> AI Foundry + vector store
7. Agent orchestration -> AI Foundry
8. Agent calls ML endpoint for scoring / classification / prediction
9. Monitoring -> ML metrics + LLM eval + system traces
10. Governance -> Purview + RBAC + Responsible AI + approval gates

### Design intent for this repo

This repo is not just a sample taxi fare demo. It is the production platform substrate
for a deterministic enterprise AI stack: Azure ML as the discipline layer for training,
lineage, registry, and deployment; Azure AI Foundry as the orchestration layer for LLMs,
RAG, and agents; both connected through shared data, shared governance, and controlled
service boundaries.

This is the right foundation for the next phase: a hybrid MLOps + LLMOps/RAG platform
that keeps model operations auditable and production-grade while expanding into agentic,
knowledge-grounded AI experiences.

## LLMOps/RAG platform architecture layered on this foundation

The platform should be built as a production-grade AI application stack, not as a one-off
prototype. The base Azure ML platform stays responsible for deterministic data science,
model governance, and operational control; the LLMOps/RAG layer sits above it and handles
retrieval, grounding, tool use, prompt evaluation, and agent runtime behavior.

### 1. Platform topology

- Azure ML remains the operational backbone for:
  - data pipelines
  - feature engineering
  - model training and benchmarking
  - model registry and deployment approvals
  - drift and performance monitoring
  - batch and online inference serving
- Azure AI Foundry becomes the orchestrated AI runtime for:
  - LLM model selection and endpoint hosting
  - RAG pipeline execution
  - prompt flow orchestration
  - tool calling / function calling
  - agent runtime and session orchestration
  - evaluation, safety, and response quality measurement

### 2. Shared enterprise data plane

The LLMOps layer must share the same enterprise data foundation instead of creating a
parallel isolated stack.

- ADLS Gen2 / lakehouse hosts raw and curated enterprise data.
- Structured operational data is still governed through Azure ML datasets and feature
  pipelines.
- Unstructured documents, manuals, policy files, tickets, help content, and PDFs are
  indexed into a retrieval plane.
- Vector stores (Cosmos DB for Mongo vCore, PostgreSQL pgvector, or Azure AI Search hybrid
  retrieval) support embeddings and similarity search.
- Purview provides lineage and governance across both traditional ML and retrieval assets.

### 3. RAG and retrieval architecture

The retrieval layer should follow a clean knowledge lifecycle.

1. Ingest and normalize source documents
   - PDFs, DOCX, markdown, HTML, emails, tickets, knowledge base pages, transcripts.
   - Tag by ownership, domain, date, sensitivity, and lifecycle state.
2. Chunk and structure content
   - chunk by semantic boundaries, section heading, or document metadata.
   - preserve provenance and document identifiers.
3. Embed and index
   - generate vectors with an embedding model managed in Foundry.
   - store alongside metadata: source system, tenant, document ID, last updated, access
     policy, and confidence score.
4. Retrieve with hybrid ranking
   - combine vector similarity with keyword search and metadata filters.
   - apply enterprise access policies before returning chunks to the model.
5. Ground responses
   - pass only approved context into the model prompt.
   - maintain citations, document references, and traceability.
6. Post-process answers
   - enforce answer templates, refusal logic, and source verification.

### 4. LLMOps control plane

LLMOps should be treated as a first-class production discipline, not a loose app backend.

- Version prompt templates, retrieval prompts, and tool instructions.
- Track prompt and model combinations as first-class artifacts.
- Maintain evaluation datasets for grounding, safety, relevance, and tool correctness.
- Run regression evaluations before every production release.
- Capture traces for each run: prompt, retrieved chunks, tool calls, model response,
  latency, cost, and error classification.
- Store these traces in an operational analytics layer for continuous improvement.

### 5. Agent platform pattern

The platform should support multi-agent and tool-augmented workflows.

- Planner / orchestrator agent decides execution strategy.
- Retrieval agent fetches grounded context.
- Tool-calling agent invokes structured business functions or APIs.
- Predictor agent may call Azure ML endpoints for a scored prediction or risk model.
- Reviewer / safety agent checks answer quality, policy compliance, and grounds against
  approved sources.

Core design rules:
- Agents must be explicit about when they call tools and why.
- Tool outputs are structured and schema-validated before being passed upstream.
- High-risk actions require explicit approval or policy checks.
- Deterministic pipelines remain authoritative for decisions that require reproducibility.

### 6. Hybrid inference pattern

This is the success pattern for the platform.

- A user request enters the Foundry agent runtime.
- The agent performs retrieval from enterprise knowledge stores.
- The agent calls an Azure ML model endpoint when a business prediction is needed.
- The system merges model outputs and grounded retrieval context into a final answer.
- The final answer is checked against policy, safety, and grounding criteria.

This creates an enterprise pattern where:
- classical ML handles prediction and scoring
- LLMs handle reasoning, summarization, and conversation
- retrieval provides domain grounding and traceability
- agents connect the two under controlled policy boundaries

### 7. Safety, policy, and governance

LLMOps/RAG requires production-grade controls beyond simple model hosting.

- content safety filters at the foundation model layer
- prompt injection defenses and tool-call restrictions
- document access controls derived from enterprise identity and authorization policies
- red-team and eval suites for adversarial prompts and jailbreak scenarios
- response provenance and citation requirements
- explicit trust tiers for public vs internal vs restricted knowledge sources
- human approval gates for privileged actions and sensitive outputs

### 8. Observability and operations

The platform needs one runtime telemetry model across AI workloads.

- model latency, token usage, throughput, and cost
- retrieval quality and retrieval latency
- successful vs failed tool calls
- hallucination or grounding failures
- model drift and response drift over time
- exception traces and root cause analysis for agent failures

Azure Monitor, Application Insights, AML monitoring, and Foundry evaluation traces should
feed a shared operational dashboard. This provides a production signal layer that is as
important as code quality for AI systems.

### 9. Release and promotion model

Use a disciplined promotion model similar to the existing ML pipeline model.

- dev: prototype and eval with synthetic and representative datasets
- test: prompt and retrieval regression tests, safety checks, tool validation
- prod: gated release with model version approval, retrieval validation, and runtime
  monitoring checks
- rollback: revert to prior prompt, model, or retrieval configuration if eval regression
  or production quality drops

### 10. What this means for this repo next

This repo should become the foundation for the next phase of work:

- introduce a reusable RAG platform layer built on Azure AI Foundry and vector search
- integrate retrieval pipelines with the existing Azure ML foundation
- add agent runtime scaffolding with deterministic tool contracts
- connect the platform to enterprise data, identity, and governance controls
- keep the deterministic MLOps platform as the source of truth for model lifecycle and
  operational correctness while Foundry handles generative AI experience orchestration

This is the correct production architecture for a hybrid platform: not ML vs LLMs, but
single platform governance with two complementary execution planes.

### Production outcome

The end state is a hybrid AI platform where:
- classical ML models remain rigorously governed and continuously monitored
- LLMs and RAG systems are evaluated, grounded, and constrained
- agents orchestrate both under enterprise controls
- the whole platform is auditable, observable, and ready for multi-system production use
