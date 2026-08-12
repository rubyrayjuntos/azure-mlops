## What this project is

`azure-mlops` is a live, deployed instance of Microsoft's MLOps v2 solution accelerator
(classical ML / aml-cli-v2 / terraform / GitHub Actions variant), running the taxi-fare
regression demo. Both Dev and Prod environments are actually deployed and exercised
end-to-end (infra -> train -> register -> batch endpoint), not just scaffolded.

## Read before touching anything

- `mem:architecture` — this repo is one of three related repos (a factory model), not
  standalone. Read this before editing any `.github/workflows/*.yml` or forking/re-syncing
  from upstream Microsoft repos.
- `mem:known_bugs` — 8 non-obvious bugs already found and fixed by actually running these
  pipelines (upstream Microsoft's own templates had never been executed before). Silently
  re-pulling from `Azure/mlops-templates` or `Azure/mlops-project-template` upstream will
  reintroduce them.
- `mem:deployment_state` — what's currently live in Azure, resource names, known gaps.
- `mem:follow_up_gaps` — prioritized post-delivery backlog for the next implementation
  plan; includes closure evidence and factory propagation constraints.
- `mem:tech_stack`, `mem:suggested_commands`, `mem:task_completion` — mechanics.

## Durable invariants

- mlops_version is aml-cli-v2 (CLI-driven, YAML pipeline specs), not Python SDK. Verify via
  `az ml job create --file ...` patterns before assuming otherwise.
- Orchestration is GitHub Actions only — no Azure DevOps involved, despite some upstream
  docs describing both paths.
- Full narrative history of how this state was reached: `docs/superpowers/plans/2026-08-09-mlops-factory-taxi-demo-dev-prod.md`
  and its ledger at `.superpowers/sdd/2026-08-09-mlops-factory-taxi-demo-dev-prod/progress.md`.
