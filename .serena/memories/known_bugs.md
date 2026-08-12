16 non-obvious bugs found only by actually executing these pipelines (none visible from
reading the code; nobody had run Microsoft's own templates before this project did).
Fixed in both `rubyrayjuntos/mlops-project-template` (nested source paths — see
`mem:architecture`'s two-copy trap) and this repo. Re-syncing from `Azure/mlops-templates`
or `Azure/mlops-project-template` upstream without reapplying these will reintroduce them.

1. **Reusable-workflow permission capping**: GitHub caps a called reusable workflow's OIDC
   token permission to `id-token: none` unless the CALLING job explicitly declares
   `permissions: id-token: write` — regardless of what the callee requests internally.
   Every job calling an Azure-OIDC reusable workflow needs this block explicitly, in every
   one of: `tf-gha-deploy-infra.yml`, `deploy-model-training-pipeline-classical.yml`,
   `deploy-batch-endpoint-pipeline-classical.yml`, `deploy-online-endpoint-pipeline-classical.yml`.

2. **Boolean coercion**: `${{ x == true }}` where `x` is a job-output string always
   evaluates false under GitHub Actions' type coercion (job outputs are always strings).
   Must compare against the string `'true'`.

3. **Deprecated `AZURE_CREDENTIALS`/`creds:` secret**: the batch and online endpoint
   pipelines, at their true sparse-checkout source paths, passed
   `secrets: creds: ${{secrets.AZURE_CREDENTIALS}}` to reusable workflows that only accept
   the 3-secret OIDC pattern. Fails GitHub's workflow-file validation before any job runs.

4. **Two copies of every workflow file in `mlops-project-template`**: top-level
   `.github/workflows/*.yml` is the template repo's own CI, never shipped by
   `sparse_checkout.sh`. Real sources: `infrastructure/terraform/github-actions/*.yml` and
   `classical/aml-cli-v2/mlops/github-actions/*.yml`. Editing the top-level copies has zero
   effect on generated projects.

5. **`train-conda.yml` missing `setuptools`**: `mlflow==2.9.2` imports `pkg_resources`
   (from `setuptools`) at runtime; newer Python/pip doesn't auto-bundle it. Fix must be
   `setuptools<70` — an UNPINNED `setuptools` resolves to whatever's latest (84.0.0 as of
   2026-08), which has dropped `pkg_resources` entirely and reproduces the identical
   failure. Confirmed via the AML job's image-build log.

6. **`batch-cluster`'s hardcoded VM size `STANDARD_D4S_V5` is invalid** — not in this
   subscription/region's supported-size list at all (checked via the actual
   `InvalidPropertyValue` error's returned list). Use `STANDARD_D4S_V3` (proven, matches
   `cpu-cluster`).

7. **`batch-cluster`'s `low_priority` tier hits 0 quota** in this subscription
   (`ClusterMinNodesExceedCoreQuota`). Switched to `dedicated` (ample headroom: 65 vCPU
   limit on Dedicated DSv3, only 4 in use by `cpu-cluster`). Trade-off: loses low-priority
   cost savings — revisit if/when a low-priority quota increase is granted.

8. **AML registers a compute entity even when provisioning fails**, left in `Failed` state.
   The create-compute step's "does it exist" check finds the broken entity and skips
   recreation, so the NEXT step (deployment) fails against unusable compute. Must
   `az ml compute delete --name <name> --resource-group <rg> --workspace-name <ws> --yes`
   before retrying after any compute-creation failure.

9. **Online custom scoring needs a compatible inference-server/MLflow pair**: omitting
   `azureml-inference-server-http` lets the image build but the user container exits 100 at
   startup. Current `azureml-inference-server-http==1.4.1` cannot coexist with
   `mlflow==2.9.2` (`gunicorn>=23` versus `<22`) and fails image dependency resolution.
   For Python 3.11 with MLflow 2.9.2, pin `azureml-inference-server-http==1.2.0` (uses
   `gunicorn==20.1.0`) and `setuptools<70` (see item 5). Confirmed from AML's retained
   `aml-environment-image-build` log and a full Python 3.11-targeted pip resolver run.

10. **Managed online endpoints can nest the MLflow artifact below `AZUREML_MODEL_DIR`**:
   for registered `taxi-model` version 4, AML set the variable to the version directory
   but mounted `MLmodel` under its `taxi-model/` child. Loading the variable directly
   crashes scorer initialization. Online `score.py` resolves either a direct `MLmodel`
   or exactly one immediate child containing it and fails on missing/ambiguous layouts.

11. **AML batch parallel-run coordination needs Azure Table data-plane RBAC**: blob roles
   and management-plane `Contributor` are insufficient. The compute UAI must have
   `Storage Table Data Contributor` on the workspace storage account or the parallel-run
   master fails before user code with `AuthorizationPermissionMismatch` while querying
   its coordination table. The role belongs in the AML workspace Terraform module and
   its existing RBAC propagation barrier.

12. **The active batch environment also needs `setuptools<70`**: its `conda.yml` pinned
   `pip=25.0` and `mlflow==2.9.2` but omitted setuptools, so every worker failed while
   importing `mlflow` with `No module named 'pkg_resources'`. Add the same compatibility
   pin used by training and online; unpinned modern setuptools can still omit
   `pkg_resources` (see item 5).

13. **AML batch parallel-run coordination also needs Azure Queue data-plane RBAC**:
   Table access alone lets coordination state initialize, but task scheduling still fails
   at `azure.storage.queue.QueueClient.send_message` with
   `AuthorizationPermissionMismatch`. The compute UAI also needs `Storage Queue Data
   Contributor` on workspace storage, included in the Terraform propagation barrier.

14. **Batch deployments can also nest the MLflow artifact below `AZUREML_MODEL_DIR`**:
   model v4 mounted `MLmodel` under an immediate child of the provided directory, exactly
   as the online deployment did. Batch and online scorers both need the direct-or-single-
   child resolver described in item 10.

15. **Invoke batch inference with the feature-only asset, not the training asset**:
   `azureml:taxi-data:1` contains the `cost` label and an exported `Unnamed: 0` index, so
   scikit-learn correctly rejects both as unseen feature names. The registered
   `azureml:taxi-batch:1` URI-file asset has the model's exact 20-feature contract and is
   the input specified by the original deployment plan.

16. **A new `workflow_dispatch` file is not dispatchable until it exists on GitHub's
   default branch**: keeping `monitor-and-retrain-classical.yml` only on `dev` makes
   `gh workflow run ... --ref dev` return HTTP 404 because GitHub has not registered the
   workflow. Merge the workflow file through the normal release path to `main` before
   Task 13's manual dispatch; do not treat the 404 as an Azure or authentication failure.
