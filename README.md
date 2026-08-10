# Azure MLOps (v2) — Taxi Fare Demo

A live, running instance of Microsoft's [MLOps v2 solution accelerator](https://github.com/Azure/mlops-v2) (classical ML / `aml-cli-v2` / Terraform / GitHub Actions), deployed end-to-end in both Dev and Prod:

| Stage | Dev | Prod |
|---|---|---|
| Infrastructure | ✅ `rg-azmlops-0001dev` | ✅ `rg-azmlops-0001prod` |
| Model training + registration | ✅ `taxi-model` | ✅ `taxi-model` |
| Batch endpoint | ✅ `taxi-gha-bep-azmlops-0001dev` | ✅ `taxi-gha-bep-azmlops-0001prod` |
| Online endpoint | not deployed (optional, deferred) | not deployed (optional, deferred) |

This isn't a scaffold — the pipelines have actually been run, and both environments have a registered model and a working batch endpoint.

## Architecture: this is one part of a three-repo factory

This repo is a generated **project instance**, not the whole accelerator. The full setup:

1. [`Azure/mlops-v2`](https://github.com/Azure/mlops-v2) — the project generator (`sparse_checkout.sh`). Used locally to scaffold new projects; not forked.
2. [`rubyrayjuntos/mlops-templates`](https://github.com/rubyrayjuntos/mlops-templates) — forked shared pipeline library (register environment/dataset, create compute/endpoint/deployment, run pipeline, allocate traffic). Referenced by this repo's workflows via `@main`.
3. [`rubyrayjuntos/mlops-project-template`](https://github.com/rubyrayjuntos/mlops-project-template) — forked project scaffold. `sparse_checkout.sh` copies from here to generate new projects like this one.
4. **This repo** — the generated instance: `infrastructure/` (Terraform), `data-science/` and `mlops/` (the taxi-fare training/scoring code), `.github/workflows/` (the deploy pipelines).

One shared Azure AD service principal, granted `Owner` at the subscription level once, authenticates every project via GitHub OIDC — no client secrets stored anywhere.

## Spinning up a new project from this factory

```bash
# 1. Configure and run sparse_checkout.sh from ~/mlops-root/mlops-v2, pointing
#    project_template_github_url at rubyrayjuntos/mlops-project-template
bash ~/mlops-root/mlops-v2/sparse_checkout.sh

# 2. Wire up OIDC auth for the new repo (adds a federated credential + the 3 GitHub secrets)
~/mlops-root/scripts/onboard-project.sh <github-org> <new-repo-name> dev
~/mlops-root/scripts/onboard-project.sh <github-org> <new-repo-name> main

# 3. Edit config-infra-dev.yml / config-infra-prod.yml in the new repo for a unique
#    namespace + postfix (Azure global-uniqueness constraints), then run
#    tf-gha-deploy-infra.yml from the Actions tab.
```

## Running the pipelines here

From the Actions tab, or via `gh`:

```bash
gh workflow run tf-gha-deploy-infra.yml --ref dev -f action=apply
gh workflow run deploy-model-training-pipeline-classical.yml --ref dev -f skip_compute_creation=true
gh workflow run deploy-batch-endpoint-pipeline-classical.yml --ref dev
```

Swap `--ref dev` for `--ref main` to target Prod. `skip_compute_creation=true` is recommended for training — the infra deploy already provisions `cpu-cluster`.

## Fixes applied on top of upstream Microsoft templates

Several real bugs surfaced only by actually running these pipelines (they were never executed before this project did). All are fixed in both this repo and the `mlops-project-template`/`mlops-templates` forks, so future projects inherit them — see [`docs/superpowers/plans/2026-08-09-mlops-factory-taxi-demo-dev-prod.md`](docs/superpowers/plans/2026-08-09-mlops-factory-taxi-demo-dev-prod.md) for the full write-up, or the Serena project memories (`known_bugs`) for the terse version. Highlights:

- GitHub Actions caps a called reusable workflow's OIDC permission to `id-token: none` unless the *caller* explicitly grants `id-token: write` — several pipelines were missing this and would fail instantly if run.
- Two pipeline files still used the deprecated `AZURE_CREDENTIALS` secret pattern, incompatible with the OIDC-only reusable workflows they call.
- `train-conda.yml` was missing `setuptools`, breaking `mlflow`'s import of `pkg_resources`.
- The batch-scoring compute cluster's hardcoded VM size and tier weren't valid/available in this subscription.

## Known limitation

Local `az ml batch-endpoint invoke` testing hits a `Tenant mismatch` error on some machines, unrelated to the deployed endpoints themselves (independently confirmed healthy via `az ml batch-endpoint show`). Use the Azure ML Studio endpoint **Test** tab as a workaround.

---

[Upstream accelerator README](https://github.com/Azure/mlops-v2/blob/main/README.md)
