# MLOps v2 Factory Completion + Taxi Demo (Dev & Prod) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the three-repo MLOps v2 accelerator (`mlops-v2`, `mlops-templates`, `mlops-project-template` — all forked to `rubyrayjuntos`) actually produce a working factory, then run the taxi-fare demo end-to-end (register environment/dataset, train, register model, deploy batch endpoint, optionally online endpoint) in both Dev and Prod on `rubyrayjuntos/azure-mlops`.

**Architecture:** One Azure AD app/service principal (`4a03064f-784b-4f7b-a429-46fad02549b5`) used for every project, granted `Owner` at the subscription level once. Every new project repo needs only a federated-credential addition (per branch) because GitHub OIDC subject matching is exact-string — everything else about the identity is shared. `mlops-templates` (shared pipeline library) and `mlops-project-template` (project scaffold) are forked once; new projects are created by running `sparse_checkout.sh` against the project-template fork. `azure-mlops` is the first project instance and becomes the reference example once these fixes land.

**Tech Stack:** GitHub Actions (OIDC to Azure AD), Terraform (`azurerm` provider), Azure ML CLI v2 (`az ml`), Bash.

## Global Constraints

- Confirmed via `run-pipeline.yml`'s `az ml job create --file ...` and `pipeline.yml`'s `$schema: .../pipelineJob.schema.json`: this project uses **aml-cli-v2**, not the Python SDK. No changes needed to confirm this — already the case throughout.
- References from one of our own forks to another of our own forks (`uses: rubyrayjuntos/mlops-templates/...`) use **`@main`**, not a pinned SHA. Rationale (per user direction): these are forks we fully control; pinning adds coordination overhead (bumping a SHA in every consumer every time the shared library is fixed) that works against the reuse goal of the factory model, without a real security benefit since nobody but us can push to our own fork's `main`.
- References to genuine third-party marketplace actions (`azure/login`, `azure/CLI`, `actions/checkout`, `hashicorp/setup-terraform`) stay **pinned to a commit SHA**. These are external and a floating tag is a real supply-chain risk regardless of whose fork calls them.
- The GitHub Actions service principal is granted **`Owner`** at the subscription level, not `Contributor` as the official deploy guide states. Verified reason: Terraform's `azurerm_role_assignment` resources in `infrastructure/terraform/modules/aml-workspace/main.tf:126-146`, and the workflow's own `az role assignment create` calls, both require `Microsoft.Authorization/roleAssignments/write`, which `Contributor` does not grant. `Owner` is the minimum role that makes the existing scripts work as-is without removing their self-granting behavior.
- `enable_private_endpoints` defaults to **`false`** going forward, matching the documented default — the `true` default in current code is an explicit "temporarily set to true for v1.2.0 testing" override we are reverting, to avoid provisioning VNets/private endpoints/DNS zones for a dev/demo setup.
- Training pipeline runs with `skip_compute_creation: true` (an existing `workflow_dispatch` input) — Terraform's `aml-workspace` module already provisions `cpu-cluster` during infra deploy; no code change needed for this. This applies to **training only** — the batch endpoint pipeline's own `batch-cluster` creation is left as designed (separate, standard AML practice for batch scoring compute).
- Scope covers **both Dev and Prod**, and **both batch and online endpoints** (batch is the required path; online endpoint tasks are marked optional/stretch and can be skipped without blocking completion).
- `azure-mlops` stays at `/home/rswan/azure-mlops` — not moved into the new `mlops-root` folder. Only the three factory-tool repos (`mlops-v2`, `mlops-templates`, `mlops-project-template`) get persistent local clones under `~/mlops-root/`.
- The two existing narrow `Owner` grants (on `rg-azmlops-0001dev` and `rg-azmlops-0001dev-tf`) are left in place, not removed. They become redundant once the subscription-level grant exists, but removing them adds risk for no benefit.
- **Critical structural fact, verified by diff:** `mlops-project-template`'s top-level `.github/workflows/*.yml` files (`tf-gha-deploy-infra.yml`, `deploy-*-pipeline-classical.yml`) are the template repo's own CI/test copies — `sparse_checkout.sh` never touches them. The files it actually ships to new projects live at `infrastructure/terraform/github-actions/tf-gha-deploy-infra.yml` and `classical/aml-cli-v2/mlops/github-actions/deploy-*-pipeline-classical.yml`. All fixes in Phase 2 target these **nested** paths, not the top-level ones (which were fixed by mistake in an earlier session pass and are lower priority to also fix — Task 9 covers this for consistency, but the nested paths are what matters).
- Known Azure identifiers used throughout: app id `4a03064f-784b-4f7b-a429-46fad02549b5`, SP object id `1f783207-33c9-40c5-ab2d-01a1f1510c8e`, subscription id `5b452321-32fd-4b1c-8bbf-6d69a5a587ad`, tenant id `90a7175b-82cd-4815-9050-8cbae3a1d234`, GitHub owner `rubyrayjuntos`, repo `azure-mlops` (numeric ids for the transitional OIDC subject: owner `204968804`, repo `1329194746`).

---

## Phase 0 — Factory Identity (one-time, unblocks every current and future project)

### Task 1: Grant the GitHub Actions service principal `Owner` at the subscription level

**Files:** None (Azure IAM only).

- [ ] **Step 1: Create the role assignment**

```bash
az role assignment create \
  --assignee-object-id 1f783207-33c9-40c5-ab2d-01a1f1510c8e \
  --assignee-principal-type ServicePrincipal \
  --role Owner \
  --scope /subscriptions/5b452321-32fd-4b1c-8bbf-6d69a5a587ad
```

- [ ] **Step 2: Verify**

```bash
az role assignment list --assignee 1f783207-33c9-40c5-ab2d-01a1f1510c8e --all -o table
```

Expected: a row showing `Owner` at scope `/subscriptions/5b452321-32fd-4b1c-8bbf-6d69a5a587ad`, in addition to the two existing narrower `Owner` rows on `rg-azmlops-0001dev` and `rg-azmlops-0001dev-tf`.

No commit — this is an Azure-side change only.

### Task 2: Add main-branch federated credentials for `azure-mlops` (unblocks Prod)

**Files:** None (Azure AD only).

- [ ] **Step 1: Add the plain-subject federated credential**

```bash
az ad app federated-credential create --id 4a03064f-784b-4f7b-a429-46fad02549b5 \
  --parameters '{"name":"github-azure-mlops-main-branch","issuer":"https://token.actions.githubusercontent.com","subject":"repo:rubyrayjuntos/azure-mlops:ref:refs/heads/main","audiences":["api://AzureADTokenExchange"]}'
```

- [ ] **Step 2: Add the numeric-ID transitional variant (matches the pattern already in place for the dev branch, in case GitHub's post-visibility-change subject format is still active)**

```bash
az ad app federated-credential create --id 4a03064f-784b-4f7b-a429-46fad02549b5 \
  --parameters '{"name":"github-azure-mlops-main-branch-transitional","issuer":"https://token.actions.githubusercontent.com","subject":"repo:rubyrayjuntos@204968804/azure-mlops@1329194746:ref:refs/heads/main","audiences":["api://AzureADTokenExchange"]}'
```

- [ ] **Step 3: Verify**

```bash
az ad app federated-credential list --id 4a03064f-784b-4f7b-a429-46fad02549b5 -o json | python3 -c "import json,sys; [print(x['name']) for x in json.load(sys.stdin)]"
```

Expected output includes all four: `github-azure-mlops-dev-branch`, `github-azure-mlops-dev-branch-transitional`, `github-azure-mlops-main-branch`, `github-azure-mlops-main-branch-transitional`.

No commit.

---

## Phase 1 — `mlops-templates` fork: consistent referencing + pinning

Work in a persistent local clone (created in Phase 4, Task 13) or the existing scratch clone — either way, push to `rubyrayjuntos/mlops-templates` `main` when done.

### Task 3: Relax the self-reference in `read-yaml.yml` from pinned SHA to `@main`

**Files:**
- Modify: `.github/workflows/read-yaml.yml`

- [ ] **Step 1: Edit the self-reference**

Change:
```yaml
        uses: rubyrayjuntos/mlops-templates/read_yaml_action@1002ad096f24223d871f298e20a3989da34ce291
```
to:
```yaml
        uses: rubyrayjuntos/mlops-templates/read_yaml_action@main
```

- [ ] **Step 2: Lint**

```bash
actionlint .github/workflows/read-yaml.yml
```
Expected: no output, exit 0.

- [ ] **Step 3: Commit and push**

```bash
git add .github/workflows/read-yaml.yml
git commit -m "Reference read_yaml_action by @main instead of a pinned SHA

Own-fork self-references don't need SHA pinning - we control main
directly, and pinning here just means every future fix to this
action requires a coordinated version bump in this same file."
git push origin main
```

### Task 4: Pin marketplace actions to SHA across the 7 dependency workflows

**Files:**
- Modify: `.github/workflows/register-environment.yml`
- Modify: `.github/workflows/register-dataset.yml`
- Modify: `.github/workflows/create-compute.yml`
- Modify: `.github/workflows/create-endpoint.yml`
- Modify: `.github/workflows/create-deployment.yml`
- Modify: `.github/workflows/run-pipeline.yml`
- Modify: `.github/workflows/allocate-traffic.yml`

(`tf-gha-install-terraform.yml` and `read-yaml.yml` are already pinned from the prior session pass — skip them. `connect-to-workspace.yml` is unused by the `classical/aml-cli-v2` project type — verified via grep, out of scope, skip it.)

- [ ] **Step 1: Apply the substitution to all 7 files**

```bash
cd .github/workflows
for f in register-environment.yml register-dataset.yml create-compute.yml create-endpoint.yml create-deployment.yml run-pipeline.yml allocate-traffic.yml; do
  sed -i \
    -e 's|uses: azure/login@v2|uses: azure/login@7184910d9eb2b1c5e48f7073824a90609bb9b6d6 # v2|' \
    -e 's|uses: actions/checkout@v4|uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4|' \
    "$f"
done
cd ../..
```

- [ ] **Step 2: Verify each file changed as expected**

```bash
grep -rn "azure/login@\|actions/checkout@" .github/workflows/register-environment.yml .github/workflows/register-dataset.yml .github/workflows/create-compute.yml .github/workflows/create-endpoint.yml .github/workflows/create-deployment.yml .github/workflows/run-pipeline.yml .github/workflows/allocate-traffic.yml
```
Expected: every line shows a 40-character SHA after `@`, none show a bare `@v2`/`@v4`.

- [ ] **Step 3: Lint**

```bash
actionlint
```
Expected: no output, exit 0.

- [ ] **Step 4: Commit and push**

```bash
git add .github/workflows/register-environment.yml .github/workflows/register-dataset.yml .github/workflows/create-compute.yml .github/workflows/create-endpoint.yml .github/workflows/create-deployment.yml .github/workflows/run-pipeline.yml .github/workflows/allocate-traffic.yml
git commit -m "Pin azure/login and actions/checkout to commit SHAs

Floating major-version tags on third-party actions mean a compromised
or retagged release silently changes what runs in every project that
depends on this shared library, including OIDC login."
git push origin main
```

---

## Phase 2 — `mlops-project-template` fork: fix the files `sparse_checkout.sh` actually ships

Work in a persistent local clone (Phase 4, Task 13) or the existing scratch clone, push to `rubyrayjuntos/mlops-project-template` `main`.

### Task 5: Fix the real infra-deploy source: `infrastructure/terraform/github-actions/tf-gha-deploy-infra.yml`

**Files:**
- Modify: `infrastructure/terraform/github-actions/tf-gha-deploy-infra.yml`

This file currently has none of last session's fixes (they went into the wrong, unused top-level copy). Bring over the `apply`/`destroy` input and `environment` output from the top-level copy (genuine improvements worth keeping), fix the boolean comparison, add `permissions:`, and point at the `mlops-templates` fork.

- [ ] **Step 1: Replace the file content**

Write the full corrected file:

```yaml
name: tf-gha-deploy-infra.yml

on:
  workflow_dispatch:
    inputs:
      action:
        description: 'Action to perform (apply or destroy)'
        required: true
        default: 'apply'
        type: choice
        options:
          - apply
          - destroy
env:
  config_env: "none"
jobs:
  set-env-branch:
    runs-on: ubuntu-latest
    outputs:
      config-file: ${{ steps.set-output-defaults.outputs.config-file }}
      environment: ${{ steps.set-output-defaults.outputs.environment }}
    steps:
      - id: set-prod-branch
        name: set-prod-branch
        if: ${{ github.ref == 'refs/heads/main'}}
        run: |
          echo "config_env=config-infra-prod.yml" >> $GITHUB_ENV
          echo "env_name=prod" >> $GITHUB_ENV
      - id: set-dev-branch
        name: setdevbranch
        if: ${{ github.ref != 'refs/heads/main'}}
        run: |
          echo "config_env=config-infra-dev.yml" >> $GITHUB_ENV
          echo "env_name=dev" >> $GITHUB_ENV
      - id: set-output-defaults
        name: set-output-defaults
        run: |
          echo "config-file=$config_env" >> $GITHUB_OUTPUT
          echo "environment=$env_name" >> $GITHUB_OUTPUT
  get-config:
    needs: set-env-branch
    permissions:
      contents: read
    uses: rubyrayjuntos/mlops-templates/.github/workflows/read-yaml.yml@main
    with:
      file_name: ${{ needs.set-env-branch.outputs.config-file}}
  test-terraform-state-deployment:
    needs: [get-config, set-env-branch]
    permissions:
      id-token: write
      contents: read
    uses: rubyrayjuntos/mlops-templates/.github/workflows/tf-gha-install-terraform.yml@main
    with:
      TFAction: ${{ github.event.inputs.action || 'apply' }}
      dply_environment: ${{ needs.set-env-branch.outputs.environment }}
      location: ${{ needs.get-config.outputs.location }}
      namespace: ${{ needs.get-config.outputs.namespace }}
      postfix: ${{ needs.get-config.outputs.postfix }}
      environment: ${{ needs.get-config.outputs.environment }}
      enable_aml_computecluster: ${{ needs.get-config.outputs.enable_aml_computecluster == 'true' }}
      enable_monitoring: ${{ needs.get-config.outputs.enable_monitoring == 'true' }}
      terraform_version: ${{ needs.get-config.outputs.terraform_version }}
      terraform_workingdir: ${{ needs.get-config.outputs.terraform_workingdir }}
      terraform_st_location: ${{ needs.get-config.outputs.terraform_st_location }}
      terraform_st_storage_account: ${{ needs.get-config.outputs.terraform_st_storage_account }}
      terraform_st_resource_group: ${{ needs.get-config.outputs.terraform_st_resource_group }}
      terraform_st_container_name: ${{ needs.get-config.outputs.terraform_st_container_name }}
      terraform_st_key: ${{ needs.get-config.outputs.terraform_st_key }}
      terraform_plan_location: ${{ needs.get-config.outputs.location }}
      terraform_plan_vnet: "TBD" # TBD
    secrets:
      AZURE_CLIENT_ID: ${{ secrets.AZURE_CLIENT_ID }}
      AZURE_TENANT_ID: ${{ secrets.AZURE_TENANT_ID }}
      AZURE_SUBSCRIPTION_ID: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
  deploy-azureml-resources:
    runs-on: ubuntu-latest
    steps:
      - id: deploy-aml-workspace
        name: deploy-aml-workspace
        run: echo "OK"
```

- [ ] **Step 2: Lint**

```bash
actionlint infrastructure/terraform/github-actions/tf-gha-deploy-infra.yml
```
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add infrastructure/terraform/github-actions/tf-gha-deploy-infra.yml
git commit -m "Fix the real sparse-checkout source for tf-gha-deploy-infra.yml

This nested copy (not .github/workflows/tf-gha-deploy-infra.yml,
which is this repo's own CI copy and is never shipped by
sparse_checkout.sh) had none of the fixes applied to the wrong file
in an earlier pass: missing permissions blocks (id-token capped to
none for nested OIDC login), boolean comparison against the bare
word true (always false under GitHub's type coercion), and an
unpinned reference to Azure/mlops-templates@main. Also adds the
apply/destroy action input this repo's top-level copy already had."
```

(Push happens after Task 8, once all three pipeline files in this phase are committed together — see Task 8's push step. Or push here immediately; either is fine, this task doesn't depend on the others.)

```bash
git push origin main
```

### Task 6: Fix `deploy-model-training-pipeline-classical.yml` (real nested source)

**Files:**
- Modify: `classical/aml-cli-v2/mlops/github-actions/deploy-model-training-pipeline-classical.yml`

Add `permissions:` to every job that calls a reusable workflow requiring OIDC, and point every `uses:` at the `mlops-templates` fork. Keep the existing `skip_environment_registration`/`skip_data_registration`/`skip_compute_creation` inputs and file paths exactly as they are — those are correct for the post-flatten project layout.

- [ ] **Step 1: Edit `get-config` job** — add:
```yaml
    permissions:
      contents: read
```
right after `needs: set-env-branch`, and change:
```yaml
    uses: Azure/mlops-templates/.github/workflows/read-yaml.yml@main
```
to:
```yaml
    uses: rubyrayjuntos/mlops-templates/.github/workflows/read-yaml.yml@main
```

- [ ] **Step 2: Edit `register-environment` job** — add:
```yaml
    permissions:
      id-token: write
      contents: read
```
right after the `if:` line, and change its `uses:` to `rubyrayjuntos/mlops-templates/.github/workflows/register-environment.yml@main`.

- [ ] **Step 3: Edit `register-dataset` job** — same pattern: add the `permissions:` block, change `uses:` to `rubyrayjuntos/mlops-templates/.github/workflows/register-dataset.yml@main`.

- [ ] **Step 4: Edit `create-compute` job** — same pattern: add the `permissions:` block, change `uses:` to `rubyrayjuntos/mlops-templates/.github/workflows/create-compute.yml@main`.

- [ ] **Step 5: Edit `run-model-training-pipeline` job** — same pattern: add the `permissions:` block, change `uses:` to `rubyrayjuntos/mlops-templates/.github/workflows/run-pipeline.yml@main`.

- [ ] **Step 6: Lint**

```bash
actionlint classical/aml-cli-v2/mlops/github-actions/deploy-model-training-pipeline-classical.yml
```
Expected: no output, exit 0.

- [ ] **Step 7: Commit**

```bash
git add classical/aml-cli-v2/mlops/github-actions/deploy-model-training-pipeline-classical.yml
git commit -m "Fix permission-capping bug and unpinned fork reference in training pipeline

Same class of bug fixed for tf-gha-deploy-infra.yml last session:
without a permissions block at the caller, GitHub caps nested reusable
workflow calls to id-token: none regardless of what those workflows
request internally. This pipeline has never been run, so the bug was
latent until now."
```

### Task 7: Fix `deploy-batch-endpoint-pipeline-classical.yml` (real nested source) — deprecated secret bug

**Files:**
- Modify: `classical/aml-cli-v2/mlops/github-actions/deploy-batch-endpoint-pipeline-classical.yml`

This file passes `secrets: creds: ${{secrets.AZURE_CREDENTIALS}}` to `create-compute.yml`, `create-endpoint.yml`, and `create-deployment.yml` — but verified those three reusable workflows only declare `AZURE_CLIENT_ID`/`AZURE_TENANT_ID`/`AZURE_SUBSCRIPTION_ID` as accepted secrets. As written, this workflow fails GitHub's workflow-file validation immediately (same `startup_failure` class as the very first bug this session, just never triggered because this pipeline has never run).

- [ ] **Step 1: Edit `get-config` job** — add `permissions: contents: read`, change `uses:` to `rubyrayjuntos/mlops-templates/.github/workflows/read-yaml.yml@main`.

- [ ] **Step 2: Edit `create-compute` job** — add:
```yaml
    permissions:
      id-token: write
      contents: read
```
change `uses:` to `rubyrayjuntos/mlops-templates/.github/workflows/create-compute.yml@main`, and replace:
```yaml
    secrets:
      creds: ${{secrets.AZURE_CREDENTIALS}}
```
with:
```yaml
    secrets:
      AZURE_CLIENT_ID: ${{ secrets.AZURE_CLIENT_ID }}
      AZURE_TENANT_ID: ${{ secrets.AZURE_TENANT_ID }}
      AZURE_SUBSCRIPTION_ID: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
```

- [ ] **Step 3: Edit `create-endpoint` job** — same `permissions:` addition, `uses:` → `rubyrayjuntos/mlops-templates/.github/workflows/create-endpoint.yml@main`, same `secrets:` replacement.

- [ ] **Step 4: Edit `create-deployment` job** — same `permissions:` addition, `uses:` → `rubyrayjuntos/mlops-templates/.github/workflows/create-deployment.yml@main`, same `secrets:` replacement.

- [ ] **Step 5: Lint**

```bash
actionlint classical/aml-cli-v2/mlops/github-actions/deploy-batch-endpoint-pipeline-classical.yml
```
Expected: no output, exit 0.

- [ ] **Step 6: Commit**

```bash
git add classical/aml-cli-v2/mlops/github-actions/deploy-batch-endpoint-pipeline-classical.yml
git commit -m "Fix deprecated AZURE_CREDENTIALS secret and permission-capping bug

This pipeline passed secrets.AZURE_CREDENTIALS as a 'creds' input to
three reusable workflows that don't declare 'creds' as an accepted
secret at all - they require AZURE_CLIENT_ID/TENANT_ID/SUBSCRIPTION_ID.
As written this fails GitHub's workflow-file validation before any
job runs. Never triggered before because this pipeline has never been
executed. Also adds the missing permissions blocks and points at the
mlops-templates fork instead of the unpinned upstream."
```

### Task 8: Fix `deploy-online-endpoint-pipeline-classical.yml` (real nested source) — same bug class

**Files:**
- Modify: `classical/aml-cli-v2/mlops/github-actions/deploy-online-endpoint-pipeline-classical.yml`

Same defects as Task 7: `get-config`, `create-endpoint`, `create-deployment`, and `allocate-traffic` jobs.

- [ ] **Step 1: Edit `get-config` job** — add `permissions: contents: read`, `uses:` → `rubyrayjuntos/mlops-templates/.github/workflows/read-yaml.yml@main`.

- [ ] **Step 2: Edit `create-endpoint` job** — add the `id-token: write`/`contents: read` permissions block, `uses:` → `rubyrayjuntos/mlops-templates/.github/workflows/create-endpoint.yml@main`, replace `secrets: creds: ...` with the 3-secret OIDC block.

- [ ] **Step 3: Edit `create-deployment` job** — same pattern, `uses:` → `rubyrayjuntos/mlops-templates/.github/workflows/create-deployment.yml@main`.

- [ ] **Step 4: Edit `allocate-traffic` job** — same pattern, `uses:` → `rubyrayjuntos/mlops-templates/.github/workflows/allocate-traffic.yml@main`.

- [ ] **Step 5: Lint**

```bash
actionlint classical/aml-cli-v2/mlops/github-actions/deploy-online-endpoint-pipeline-classical.yml
```
Expected: no output, exit 0.

- [ ] **Step 6: Commit and push everything from Phase 2**

```bash
git add classical/aml-cli-v2/mlops/github-actions/deploy-online-endpoint-pipeline-classical.yml
git commit -m "Fix deprecated AZURE_CREDENTIALS secret and permission-capping bug

Same defects as the batch endpoint pipeline: creds/AZURE_CREDENTIALS
passed to reusable workflows that only accept the OIDC 3-secret
pattern, and no permissions block on any job."
git push origin main
```

### Task 9: Verify `enable_private_endpoints` default and `terraform_workingdir` fixes are still correct on this fork

**Files:**
- Read: `infrastructure/terraform/variables.tf`
- Read: `config-infra-dev.yml`, `config-infra-prod.yml`

These were fixed correctly in an earlier session pass (top-level `infrastructure/` and root `config-infra-*.yml` genuinely are the real sparse-checkout sources — verified via `sparse_checkout.sh`'s cone-mode checkout, which always includes root files, and its explicit `infrastructure/$infrastructure_version` pattern). This task is a verification, not a new fix.

- [ ] **Step 1: Confirm**

```bash
grep -A3 'variable "enable_private_endpoints"' infrastructure/terraform/variables.tf
grep "terraform_workingdir" config-infra-dev.yml config-infra-prod.yml
```

Expected: `enable_private_endpoints` default is `false`; both config files show `terraform_workingdir: infrastructure`.

If either check fails, apply the fix now (`default = false` in `variables.tf`; `terraform_workingdir: infrastructure` in both config files), lint isn't applicable (not workflow YAML), commit and push.

---

## Phase 3 — Sync `azure-mlops` to the now-fixed factory output

### Task 10: Replace `azure-mlops`'s 4 workflow files with the fixed nested-source content

**Files:**
- Modify: `/home/rswan/azure-mlops/.github/workflows/tf-gha-deploy-infra.yml`
- Modify: `/home/rswan/azure-mlops/.github/workflows/deploy-model-training-pipeline-classical.yml`
- Modify: `/home/rswan/azure-mlops/.github/workflows/deploy-batch-endpoint-pipeline-classical.yml`
- Modify: `/home/rswan/azure-mlops/.github/workflows/deploy-online-endpoint-pipeline-classical.yml`

`azure-mlops`'s copies already have the post-flatten file paths (no `classical/aml-cli-v2/` prefix) — they're structurally identical to the nested source, just missing this phase's fixes. Copy the fixed nested-source content from the `mlops-project-template` fork directly.

- [ ] **Step 1: Copy the four fixed files over**

```bash
cd /home/rswan/azure-mlops
PROJECT_TEMPLATE=<path to your local mlops-project-template clone from Task 13>
cp "$PROJECT_TEMPLATE/infrastructure/terraform/github-actions/tf-gha-deploy-infra.yml" .github/workflows/tf-gha-deploy-infra.yml
cp "$PROJECT_TEMPLATE/classical/aml-cli-v2/mlops/github-actions/deploy-model-training-pipeline-classical.yml" .github/workflows/deploy-model-training-pipeline-classical.yml
cp "$PROJECT_TEMPLATE/classical/aml-cli-v2/mlops/github-actions/deploy-batch-endpoint-pipeline-classical.yml" .github/workflows/deploy-batch-endpoint-pipeline-classical.yml
cp "$PROJECT_TEMPLATE/classical/aml-cli-v2/mlops/github-actions/deploy-online-endpoint-pipeline-classical.yml" .github/workflows/deploy-online-endpoint-pipeline-classical.yml
```

- [ ] **Step 2: Diff against what was there before, sanity-check nothing azure-mlops-specific got lost**

```bash
git diff --stat .github/workflows/
git diff .github/workflows/
```
Expected: `tf-gha-deploy-infra.yml` gains the `action` input, `permissions:` blocks, fork references, boolean fix. The three pipeline files gain `permissions:` blocks and fork references; the batch/online ones also lose `AZURE_CREDENTIALS`/`creds:` in favor of the 3-secret pattern.

- [ ] **Step 3: Lint**

```bash
LINTER=$(command -v actionlint || echo /tmp/claude-1000/-home-rswan-azure-mlops/1a669a6d-32d1-40a8-8a14-e6afa3eb3ad8/scratchpad/actionlint)
"$LINTER"
```
Expected: no output, exit 0.

- [ ] **Step 4: Commit and push**

```bash
git add .github/workflows/
git commit -m "Sync workflow files with the fixed mlops-project-template fork

Brings azure-mlops's copies of all four workflows up to date with the
now-corrected sparse-checkout source: OIDC permissions blocks, the
fixed AZURE_CREDENTIALS bug in the two endpoint pipelines, the
apply/destroy input on the infra pipeline, and references to the
rubyrayjuntos/mlops-templates fork instead of unpinned upstream."
git push origin dev
```

### Task 11: Confirm `enable_private_endpoints` change and decide on re-applying to already-deployed Dev

**Files:**
- Read: `/home/rswan/azure-mlops/infrastructure/variables.tf`

The already-deployed Dev environment currently has private endpoints (created while the default was `true`). Flipping the default to `false` and re-running `terraform apply` would **destroy** those private endpoints, DNS zones, and the VNet — a real, visible infrastructure change to a working environment, not just a config edit.

- [ ] **Step 1: Sync the variable default**

```bash
cd /home/rswan/azure-mlops
grep -A3 'variable "enable_private_endpoints"' infrastructure/variables.tf
```
If it still shows `default = true`, change it to `default = false` (it may already be `false` from an earlier session pass — verify before editing).

- [ ] **Step 2: Commit if changed**

```bash
git add infrastructure/variables.tf
git commit -m "Default enable_private_endpoints to false, matching documented default"
git push origin dev
```

- [ ] **Step 3: Decide whether to re-apply to Dev now**

This step intentionally has no default action — confirm with the user before running `tf-gha-deploy-infra.yml` again against `dev` if the goal is to actually remove the already-created private endpoints from the live Dev environment. If skipped, the new default only affects Prod's first deploy (Task 15) and any future projects; Dev keeps its private endpoints until a deliberate re-apply.

---

## Phase 4 — Local factory folder + new-project onboarding automation

### Task 12: Create the persistent `mlops-root` folder with the three factory-tool clones

**Files:**
- Create: `~/mlops-root/` (directory)

- [ ] **Step 1: Create the folder and clone**

```bash
mkdir -p ~/mlops-root
cd ~/mlops-root
git clone https://github.com/Azure/mlops-v2.git
gh repo clone rubyrayjuntos/mlops-templates
gh repo clone rubyrayjuntos/mlops-project-template
```

- [ ] **Step 2: Verify**

```bash
ls ~/mlops-root
```
Expected: `mlops-v2`, `mlops-templates`, `mlops-project-template` — three directories, each a working git clone with `origin` pointing at your fork (for the latter two) or `Azure/mlops-v2` (for the generator).

This is where Phase 1 and Phase 2's edits should actually be made and pushed from, rather than the ephemeral scratch clones used during investigation.

### Task 13: Write a new-project onboarding script

**Files:**
- Create: `~/mlops-root/scripts/onboard-project.sh`

Automates the two genuinely manual steps left after `sparse_checkout.sh` runs: adding a federated credential for the new repo+branch, and setting the 3 GitHub secrets. Assumes Phase 0's subscription-level `Owner` grant already exists (one-time, not repeated here).

- [ ] **Step 1: Write the script**

```bash
mkdir -p ~/mlops-root/scripts
cat > ~/mlops-root/scripts/onboard-project.sh << 'SCRIPT_EOF'
#!/usr/bin/env bash
# Usage: onboard-project.sh <github-owner> <repo-name> <branch>
# Adds an OIDC federated credential for <repo-name>:<branch> to the shared
# GitHub Actions app registration, and sets the 3 Azure secrets on the repo.
# Run once per (repo, branch) pair - e.g. once for "dev", once for "main".
set -euo pipefail

OWNER="$1"
REPO="$2"
BRANCH="$3"
APP_ID="4a03064f-784b-4f7b-a429-46fad02549b5"

TENANT_ID=$(az account show --query tenantId -o tsv)
SUBSCRIPTION_ID=$(az account show --query id -o tsv)

az ad app federated-credential create --id "$APP_ID" --parameters "$(cat <<JSON
{
  "name": "github-${REPO}-${BRANCH}-branch",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:${OWNER}/${REPO}:ref:refs/heads/${BRANCH}",
  "audiences": ["api://AzureADTokenExchange"]
}
JSON
)"

gh secret set AZURE_CLIENT_ID --repo "${OWNER}/${REPO}" --body "$APP_ID"
gh secret set AZURE_TENANT_ID --repo "${OWNER}/${REPO}" --body "$TENANT_ID"
gh secret set AZURE_SUBSCRIPTION_ID --repo "${OWNER}/${REPO}" --body "$SUBSCRIPTION_ID"

echo "Onboarded ${OWNER}/${REPO}:${BRANCH} - federated credential added, 3 secrets set."
echo "Remaining manual step: edit config-infra-${BRANCH}.yml (or config-infra-dev.yml/config-infra-prod.yml) for a unique namespace/postfix."
SCRIPT_EOF
chmod +x ~/mlops-root/scripts/onboard-project.sh
```

- [ ] **Step 2: Verify it's idempotent-safe to inspect (dry check, don't run against a real new repo yet)**

```bash
bash -n ~/mlops-root/scripts/onboard-project.sh
```
Expected: no output (syntax valid).

- [ ] **Step 3: No commit needed** — this lives outside any of the three git repos, in the local `mlops-root` scratch/tooling folder. If you'd rather version it, `git init` a small `mlops-root/scripts` repo of your own; out of scope for this plan.

---

## Phase 5 — Run the taxi demo in Dev

Precondition: Phases 1–3 complete and pushed. Trigger everything via `gh workflow run` from `/home/rswan/azure-mlops`, on the `dev` branch (default for these commands).

### Task 14: Re-run infra deploy to pick up the fixed workflow (confirms Phase 2/3 fixes work)

- [ ] **Step 1: Trigger**

```bash
cd /home/rswan/azure-mlops
gh workflow run tf-gha-deploy-infra.yml --ref dev -f action=apply
```

- [ ] **Step 2: Watch to completion**

```bash
sleep 10
RUN_ID=$(gh run list --workflow=tf-gha-deploy-infra.yml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --exit-status
```
Expected: exits 0, all jobs succeeded.

### Task 15: Run the model training pipeline (Dev)

- [ ] **Step 1: Trigger with compute creation skipped**

```bash
gh workflow run deploy-model-training-pipeline-classical.yml --ref dev -f skip_compute_creation=true
```

- [ ] **Step 2: Watch to completion**

```bash
sleep 10
RUN_ID=$(gh run list --workflow=deploy-model-training-pipeline-classical.yml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --exit-status
```
Expected: exits 0.

- [ ] **Step 3: Verify a model was registered**

```bash
az ml model list --resource-group rg-azmlops-0001dev --workspace-name mlw-azmlops-0001dev -o table
```
Expected: at least one model listed (name from `mlops/azureml/train/pipeline.yml`'s register step).

### Task 16: Deploy and test the batch endpoint (Dev)

- [ ] **Step 1: Trigger**

```bash
gh workflow run deploy-batch-endpoint-pipeline-classical.yml --ref dev
```

- [ ] **Step 2: Watch to completion**

```bash
sleep 10
RUN_ID=$(gh run list --workflow=deploy-batch-endpoint-pipeline-classical.yml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --exit-status
```
Expected: exits 0.

- [ ] **Step 3: Test the batch endpoint per the deploy guide**

```bash
az ml data create --name taxi-batch --version 1 \
  --workspace-name mlw-azmlops-0001dev --resource-group rg-azmlops-0001dev \
  --path data/taxi-batch.csv --type uri_file

az ml batch-endpoint invoke --name taxi-gha-bep-azmlops-0001dev \
  --workspace-name mlw-azmlops-0001dev --resource-group rg-azmlops-0001dev \
  --input azureml:taxi-batch:1 --input-type uri_folder
```
Expected: a job ID is printed. Follow up with:
```bash
az ml job show --name <job-id> --workspace-name mlw-azmlops-0001dev --resource-group rg-azmlops-0001dev
```
Expected `status`: `Completed`.

### Task 17 (Optional/Stretch): Deploy and test the online endpoint (Dev)

- [ ] **Step 1: Trigger**

```bash
gh workflow run deploy-online-endpoint-pipeline-classical.yml --ref dev
```

- [ ] **Step 2: Watch to completion**

```bash
sleep 10
RUN_ID=$(gh run list --workflow=deploy-online-endpoint-pipeline-classical.yml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --exit-status
```

- [ ] **Step 3: Test using `data/test-request.json` (already present in the repo)**

```bash
SCORING_URI=$(az ml online-endpoint show --name taxi-gha-oep-azmlops-0001dev \
  --workspace-name mlw-azmlops-0001dev --resource-group rg-azmlops-0001dev \
  --query scoring_uri -o tsv)
KEY=$(az ml online-endpoint get-credentials --name taxi-gha-oep-azmlops-0001dev \
  --workspace-name mlw-azmlops-0001dev --resource-group rg-azmlops-0001dev \
  --query primaryKey -o tsv)
curl -X POST "$SCORING_URI" -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  --data @data/test-request.json
```
Expected: a JSON array of fare predictions.

---

## Phase 6 — Run the taxi demo in Prod

Precondition: Phase 0 complete (subscription `Owner` grant + main-branch federated credentials). This is the first-ever Prod deploy for this project.

### Task 18: Deploy Prod infrastructure

- [ ] **Step 1: Trigger on `main`**

```bash
gh workflow run tf-gha-deploy-infra.yml --ref main -f action=apply
```

- [ ] **Step 2: Watch to completion**

```bash
sleep 10
RUN_ID=$(gh run list --workflow=tf-gha-deploy-infra.yml --branch main --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --exit-status
```
Expected: exits 0. If it fails on Azure login with an OIDC subject mismatch, re-check Task 2's federated credentials — this is the same failure mode diagnosed for `dev` earlier, just for `main`.

- [ ] **Step 3: Verify the resource group exists**

```bash
az group exists --name rg-azmlops-0001prod
```
Expected: `true`.

### Task 19: Run the model training pipeline (Prod)

- [ ] **Step 1: Trigger**

```bash
gh workflow run deploy-model-training-pipeline-classical.yml --ref main -f skip_compute_creation=true
```

- [ ] **Step 2: Watch to completion**

```bash
sleep 10
RUN_ID=$(gh run list --workflow=deploy-model-training-pipeline-classical.yml --branch main --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --exit-status
```

- [ ] **Step 3: Verify**

```bash
az ml model list --resource-group rg-azmlops-0001prod --workspace-name mlw-azmlops-0001prod -o table
```
Expected: at least one model listed.

### Task 20: Deploy and test the batch endpoint (Prod)

- [ ] **Step 1: Trigger**

```bash
gh workflow run deploy-batch-endpoint-pipeline-classical.yml --ref main
```

- [ ] **Step 2: Watch to completion**

```bash
sleep 10
RUN_ID=$(gh run list --workflow=deploy-batch-endpoint-pipeline-classical.yml --branch main --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --exit-status
```

- [ ] **Step 3: Test**

```bash
az ml data create --name taxi-batch --version 1 \
  --workspace-name mlw-azmlops-0001prod --resource-group rg-azmlops-0001prod \
  --path data/taxi-batch.csv --type uri_file

az ml batch-endpoint invoke --name taxi-gha-bep-azmlops-0001prod \
  --workspace-name mlw-azmlops-0001prod --resource-group rg-azmlops-0001prod \
  --input azureml:taxi-batch:1 --input-type uri_folder
```
Expected: job created; follow up with `az ml job show` until `status` is `Completed`.

### Task 21 (Optional/Stretch): Deploy and test the online endpoint (Prod)

- [ ] **Step 1: Trigger**

```bash
gh workflow run deploy-online-endpoint-pipeline-classical.yml --ref main
```

- [ ] **Step 2: Watch and test**, same pattern as Task 17 but against `rg-azmlops-0001prod` / `mlw-azmlops-0001prod` / `taxi-gha-oep-azmlops-0001prod`.

---

## Self-Review Notes

- **Spec coverage:** Phase 0 covers the subscription-level identity correction and Prod OIDC unblock; Phase 1–2 cover every file in `mlops-templates` and `mlops-project-template` that this project's pipelines touch (verified by tracing every `uses:` in all 4 azure-mlops workflow files back to its real source, not assumed); Phase 3 syncs `azure-mlops` itself; Phase 4 covers the local folder structure and onboarding automation; Phases 5–6 run boxes ③–⑦ for both Dev and Prod, batch required, online marked optional per explicit scope-reduction permission.
- **Placeholder scan:** All commands use real, verified IDs (app id, SP object id, subscription id, tenant id, repo names). The one intentionally-open decision (Task 11, Step 3) is flagged as a decision point, not a placeholder — re-applying a destructive change to live infrastructure shouldn't be scripted as an unconditional step.
- **Type/name consistency:** Endpoint names (`taxi-gha-bep-<namespace><postfix><env>`, `taxi-gha-oep-...`) match the `format('taxi-gha-{0}', needs.get-config.outputs.bep)` pattern in the actual pipeline files, cross-checked against the `read-yaml` action's `bep`/`oep` output construction (`bep-${namespace}-${postfix}${environment}`).
