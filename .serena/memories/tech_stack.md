- mlops_version: aml-cli-v2 — CLI-driven (`az ml job create --file ...`), YAML pipeline
  specs (`$schema: .../pipelineJob.schema.json`). NOT Python SDK. Verify against actual
  `run-pipeline.yml` content before assuming otherwise if this ever seems ambiguous.
- Infra: Terraform (`azurerm` provider), state in Azure Storage backend
  (`stazmlops0001<env>tf` / `rg-azmlops-0001<env>-tf`).
- Orchestration: GitHub Actions only. No Azure DevOps in this deployment, despite some
  upstream docs describing both paths as options.
- Training env: conda (`data-science/environment/train-conda.yml`), mlflow==2.9.2,
  scikit-learn, pandas. See `mem:known_bugs` item 5 for the setuptools pin requirement.
- `actionlint` is not on PATH by default in this environment — download from
  https://github.com/rhysd/actionlint/releases if missing before linting workflow YAML.
