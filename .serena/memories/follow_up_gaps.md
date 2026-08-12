# Follow-up Gaps After Monitor/Retrain Delivery

Planning backlog for work after the current monitor-retrain-champion-challenger plan is
fully validated. Do not start these items while Task 12 or its final monitor-workflow
verification is still active. Reassess priority against the completed execution ledger
before creating the next implementation plan.

## P0 — Make monitoring resources and RBAC declarative

Current monitoring setup relies on live/manual Azure changes: the `monitoring` blob
container and the online endpoint identity's storage write role. A clean environment or
new generated project can therefore deploy successfully but fail to persist inference
logs.

Plan must:
- Provision the monitoring container through Terraform/factory infrastructure.
- Assign least-privilege `Storage Blob Data Contributor` to the online endpoint managed
  identity at the narrowest practical scope.
- Preserve the training/batch user-assigned identity permissions needed by baseline,
  batch logging, and drift jobs.
- Cover dependency ordering where the endpoint identity does not exist until endpoint
  creation.
- Propagate project changes to the authoritative nested source in
  `mlops-project-template`; see `mem:architecture`.

Done evidence: a clean Dev deployment creates all resources and role assignments without
manual `az` commands, then both online and batch inference logs are written successfully.

## P0 — Prove the complete drift-to-redeployment control loop

Individual components and the healthy/insufficient-data paths are not equivalent to an
end-to-end proof of the trigger path.

Plan must:
- Generate deterministic shifted inference data large enough to pass the minimum-row
  threshold and produce `DRIFT_DETECTED` under the configured statistical/effect-size
  gates.
- Verify drift triggers retraining exactly once.
- Verify the existing champion/challenger evaluation remains the sole promotion gate.
- Exercise both outcomes: challenger rejected with no redeploy; challenger promoted with
  conditional batch/online redeployment.
- Assert artifacts, model version transitions, endpoint traffic, and post-deploy scoring
  rather than relying only on green workflow status.

Done evidence: reproducible Dev run IDs and Azure state checks demonstrate both promotion
branches without manual mutation of workflow outputs.

## P1 — Add operational alerting and run observability

Scheduled monitoring can fail or detect drift without a durable operator signal. Blob
artifacts and GitHub run status alone are insufficient for unattended operation.

Plan must:
- Define alerts for monitor workflow failure, `DRIFT_DETECTED`, retraining failure,
  deployment failure, and repeated `INSUFFICIENT_DATA`.
- Make each run traceable across GitHub Actions, AML jobs, registered model versions, and
  endpoint deployments using stable correlation metadata.
- Define retention and a compact run summary containing status, sample counts, tests,
  adjusted p-values, effect sizes, promotion result, and deployment result.
- Avoid introducing ADX unless blob/Application Insights/Azure Monitor cannot satisfy the
  concrete query and alert requirements.

Done evidence: forced failure and forced drift tests each create one actionable alert with
links or identifiers sufficient to trace the originating workflow and AML resources.

## P1 — Modernize pinned GitHub Actions runtimes

Current runs warn that Node.js 20 is deprecated and actions are being forced to Node.js
24. This affects pinned `actions/checkout`, `azure/login`, and the custom/shared YAML-read
action. It is not the current deployment blocker but will become a reliability risk.

Plan must:
- Inventory all first-party, third-party, and custom actions across the project and both
  factory forks.
- Upgrade to Node 24-compatible releases while preserving commit-SHA pinning for external
  actions.
- Update custom actions or replace abandoned ones.
- Run `actionlint` and execute representative infra, training, batch, online, and monitor
  workflows.

Done evidence: no Node runtime deprecation annotations remain and representative workflow
runs succeed from a newly generated project copy.

## P2 — Resolve local AML CLI tenant mismatch

Local `az ml *-endpoint invoke` can fail with `Tenant mismatch` even after a clean login;
Azure ML Studio is the current workaround. This impairs repeatable local smoke testing but
does not invalidate deployed endpoint health.

Plan must isolate Azure CLI/ML extension token acquisition, subscription tenant metadata,
and endpoint data-plane authentication, then either fix the local path or codify a
non-interactive CI smoke-test path that makes the local limitation irrelevant.

Done evidence: a documented command invokes both endpoint types from a clean session, or
CI performs the equivalent authenticated smoke tests with no Studio-only step.

## Planning order

1. Declarative monitoring resources/RBAC.
2. Complete drift-to-redeployment proof.
3. Operational alerting and correlation.
4. Actions runtime modernization.
5. Local CLI tenant mismatch.

## Production-grade enterprise platform roadmap

This project has moved beyond demo status. It is the spine of an enterprise MLOps
platform, and the next phase is a hybrid MLOps + LLMOps/RAG platform. Keep all work
anchored to production-grade operations, governance, and platform extensibility rather
than demo-only convenience.

### TODO 1 — Harden the production platform baseline
- Remove public-by-default assumptions from online endpoint and supporting infra.
- Default production environments to private networking, private endpoints, and
  restricted egress.
- Convert endpoint authentication to Entra-backed identity patterns instead of API keys.
- Scope storage permissions to the exact monitoring container rather than broad account
  scope unless a stricter use case requires otherwise.
- Treat the current Azure ML deployment as a platform substrate, not a one-off demo.

### TODO 2 — Create a production-grade control plane
- Define environment guardrails for dev, test, prod separation, approval gates, and
  tenant/subscription normalization.
- Add policy checks for endpoint lifecycle, resource tagging, identity posture, network
  isolation, and least-privilege RBAC.
- Require deployment evidence, artifact traceability, and rollback readiness for every
  model or endpoint promotion.
- Establish clear ownership boundaries between data science, platform engineering,
  and production operations.

### TODO 3 — Add enterprise monitoring and alerting
- Alert on workflow failures, drift detection, retraining failures, deployment failures,
  and repeated insufficient-data states.
- Correlate Azure ML jobs, model versions, endpoint states, and GitHub Actions runs with
  stable metadata.
- Define operational dashboards for MLOps health, platform reliability, and model
  lifecycle status.
- Keep alerting actionable and minimal so production operators are not buried in noise.

### TODO 4 — Extend from MLOps to hybrid MLOps + LLMOps/RAG platform
- Define how the same platform governs both training/inference pipelines and LLM/RAG
  workloads.
- Add reusable model serving patterns for classic ML, fine-tuned models, and retrieval
  augmented generation workloads.
- Separate inference contracts for batch prediction, online scoring, and RAG-backed
  assistant services.
- Standardize evaluation, observability, and rollback across both ML and GenAI workloads.

### TODO 5 — Prepare the LLMOps/RAG next phase
- Bring in the pre-mapped LLMOps/RAG architecture, agent/tooling patterns, prompt
  evaluation, retrieval/data grounding, and production safety controls.
- Build a platform layer that enables RAG apps without re-creating MLOps plumbing for
  each workload.
- Treat LLMOps as an extension of the same platform operating model, not a separate
  disconnected project.

### TODO 6 — Close the backlog and explicitly validate production readiness
- Finish the current backlog items before expanding the platform footprint.
- Reassess the environment as supported, governed, monitored, and production-ready only
  after hardening and self-service platform controls are in place.
- Keep the architecture document aligned with the reality that this is an enterprise
  platform backbone, not a demo accelerator.

Security/network hardening (private endpoints, restricted egress, production isolation)
should be assessed in the next architecture review, but is not promoted into this backlog
without explicit production requirements; current public-network posture is deliberate.