# Declarative Monitoring Storage + Endpoint RBAC Implementation Plan

> **For agentic workers:** Implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make monitoring storage and online-inference logging work in a newly generated environment without manual Azure changes, while reducing the online endpoint identity from storage-account-wide access to write access on the `monitoring` container only.

**Architecture:** Terraform owns the lifecycle of the existing workspace storage account's private `monitoring` blob container. The online endpoint remains system-assigned because Azure ML creates that identity with the endpoint, after the infrastructure deployment has completed. The shared `create-endpoint` workflow queries the principal immediately after endpoint creation or update and idempotently creates a container-scoped `Storage Blob Data Contributor` assignment before the scoring deployment starts. This avoids a Terraform dependency on an endpoint that Terraform does not own and makes endpoint identity replacement self-healing on the next deployment. Training and batch compute retain their existing user-assigned identity and storage permissions.

**Tech Stack:** Terraform `azurerm` 4.52, Azure ML CLI v2, Azure CLI RBAC commands, GitHub Actions reusable workflows, OIDC.

## Global Constraints

- **Do not move Azure ML endpoint ownership into Terraform.** Endpoint and deployment lifecycle currently belongs to the project workflow and AML YAML. Importing only the endpoint identity into infrastructure state would create split ownership of one resource.
- **Do not grant the endpoint at storage-account scope.** Use the container resource scope `${storage_account_id}/blobServices/default/containers/monitoring` and `Storage Blob Data Contributor` only. The scorer needs blob data-plane writes, not Reader plus Contributor and not management-plane Contributor.
- **Keep the endpoint system-assigned.** A dedicated user-assigned identity would solve creation ordering but would add another identity to every generated project and require dynamic identity injection into endpoint YAML. The endpoint workflow already has the exact post-creation boundary needed to authorize the system identity.
- **Make role creation idempotent and fail closed.** Do not hide arbitrary `az role assignment create` failures behind `|| true`. Query the exact principal, role, and scope first; create only when absent; fail if the principal or container cannot be resolved.
- **Handle propagation before deployment.** Role creation success does not mean the data-plane permission is usable. Use a bounded retry that performs an authenticated container access check as the endpoint identity where Azure supports it; if that is not possible from GitHub-hosted runners, poll the exact role assignment through ARM and retain the scorer's existing retry behavior as defense in depth. Never use an unbounded sleep.
- **Preserve existing batch/training permissions.** `uai-<prefix>-<postfix><env>` still needs Blob, Table, and Queue data-plane roles for AML jobs and ParallelRunStep. This plan changes only the online endpoint's grant.
- **Migrate before removing broad live grants.** First prove online logging through the narrow assignment. Remove the old storage-account-scoped endpoint grants only after that proof, and verify they are absent afterward.
- **Propagate factory changes.** Project infrastructure changes map to `/home/rswan/mlops-root/mlops-project-template/infrastructure/terraform/`. Project workflow changes map to `/home/rswan/mlops-root/mlops-project-template/classical/aml-cli-v2/mlops/github-actions/`. Shared workflow changes belong to `/home/rswan/mlops-root/mlops-templates/.github/workflows/` and must be consumed through the existing `@main` own-fork convention.
- **Dev first, then Prod.** Do not migrate Prod RBAC until Dev has written and retrieved a real online inference log with only the narrow grant.
- **No destructive infrastructure apply without plan review.** Container adoption must not replace the storage account, workspace, endpoint, or existing monitoring data.

---

## Phase 1 - Terraform-owned monitoring container

### Task 1: Add a private monitoring container to the storage module

**Files:**
- Modify: `infrastructure/modules/storage-account/main.tf`
- Modify: `infrastructure/modules/storage-account/outputs.tf`
- Mirror: `/home/rswan/mlops-root/mlops-project-template/infrastructure/terraform/modules/storage-account/main.tf`
- Mirror: `/home/rswan/mlops-root/mlops-project-template/infrastructure/terraform/modules/storage-account/outputs.tf`

- [ ] **Step 1: Add the container resource**

Add `azurerm_storage_container.monitoring` with:

```hcl
resource "azurerm_storage_container" "monitoring" {
  name                  = "monitoring"
  storage_account_id    = azurerm_storage_account.st.id
  container_access_type = "private"
}
```

Use `storage_account_id`, not the deprecated account-name argument. Do not encode the `monitoring/` virtual-directory prefix in infrastructure; blob producers already own the paths beneath the container.

- [ ] **Step 2: Export the canonical container resource ID**

Add an output named `monitoring_container_id` whose value is `azurerm_storage_container.monitoring.id`. This gives future role assignments and tests one canonical scope rather than reconstructing it in multiple Terraform modules.

- [ ] **Step 3: Mirror the module changes to the project-template nested source**

Confirm project and template files are byte-identical after the edit.

- [ ] **Step 4: Format and validate both Terraform trees**

```bash
terraform -chdir=infrastructure fmt -check -recursive
terraform -chdir=infrastructure init -backend=false
terraform -chdir=infrastructure validate

terraform -chdir=/home/rswan/mlops-root/mlops-project-template/infrastructure/terraform fmt -check -recursive
terraform -chdir=/home/rswan/mlops-root/mlops-project-template/infrastructure/terraform init -backend=false
terraform -chdir=/home/rswan/mlops-root/mlops-project-template/infrastructure/terraform validate
```

Expected: both validations pass and formatting produces no diff.

### Task 2: Adopt the existing Dev container into Terraform state

**Files:** None. This is a Terraform state migration.

- [ ] **Step 1: Initialize the Dev backend using the same backend values as the infrastructure workflow**

Use the values from `config-infra-dev.yml`. Do not create a second local state file and do not apply yet.

- [ ] **Step 2: Confirm the live container and state relationship**

```bash
terraform -chdir=infrastructure state show module.storage_account.azurerm_storage_container.monitoring
```

Expected before import: resource not found in state. Independently confirm the `monitoring` container exists in `stazmlops0001dev`.

- [ ] **Step 3: Import the existing container**

Import it at the exact module address using its ARM resource ID:

```text
/subscriptions/<subscription-id>/resourceGroups/rg-azmlops-0001dev/providers/Microsoft.Storage/storageAccounts/stazmlops0001dev/blobServices/default/containers/monitoring
```

- [ ] **Step 4: Run and review a saved Dev plan**

Use the same variable set as `tf-gha-deploy-infra.yml`. Expected result for this task: no container creation and no replacement or deletion of the storage account, workspace, or monitoring blobs. Stop if the plan proposes destructive changes.

- [ ] **Step 5: Apply only through the established infrastructure workflow**

Apply after reviewing the saved plan or the workflow's equivalent plan artifact. Verify Terraform state now owns the container and a second plan is empty.

---

## Phase 2 - Endpoint-owned RBAC automation

### Task 3: Extend the reusable endpoint workflow with optional blob-container authorization

**Files:**
- Modify: `/home/rswan/mlops-root/mlops-templates/.github/workflows/create-endpoint.yml`

- [ ] **Step 1: Add optional workflow inputs**

Add string inputs with empty defaults:

```yaml
storage_account_name:
  required: false
  type: string
  default: ""
storage_container_name:
  required: false
  type: string
  default: ""
```

Authorization runs only when both values are non-empty. Batch endpoint callers remain unchanged.

- [ ] **Step 2: Authorize the endpoint identity after create or update**

After `create-or-update-endpoint`, add a step that:

1. Rejects partial configuration where only one storage input is set.
2. Requires `endpoint_type == online` when storage authorization is requested.
3. Gets `identity.principal_id` from the just-created endpoint and fails if empty.
4. Resolves the storage account ARM ID with `az storage account show`.
5. Constructs `<storage-account-id>/blobServices/default/containers/<container-name>`.
6. Verifies the container exists through ARM.
7. Lists assignments for the exact principal, role, and scope.
8. Creates `Storage Blob Data Contributor` only when the exact assignment is absent, using `--assignee-object-id` and `--assignee-principal-type ServicePrincipal` to avoid directory lookup ambiguity.
9. Emits the endpoint principal ID and role-assignment scope to the job summary without exposing tokens or credentials.

Keep all shell values quoted. Use `set -euo pipefail` and argument arrays for Azure CLI commands.

- [ ] **Step 3: Add a bounded propagation check**

Poll for the exact ARM role assignment for at most five minutes with capped intervals. Fail the workflow if it never becomes observable. Do not use a fixed multi-minute sleep.

- [ ] **Step 4: Validate the reusable workflow**

```bash
actionlint /home/rswan/mlops-root/mlops-templates/.github/workflows/create-endpoint.yml
```

Add or run a shell-level test for all branches that can be exercised without Azure:
- no storage inputs: authorization step skips;
- one input only: fails;
- batch endpoint plus storage inputs: fails;
- exact assignment already exists: does not create a duplicate.

- [ ] **Step 5: Commit and push the shared workflow independently**

Push this change to `rubyrayjuntos/mlops-templates` `main` before changing project callers, because callers use `@main`.

### Task 4: Pass the monitoring container from online endpoint callers

**Files:**
- Modify: `.github/workflows/deploy-online-endpoint-pipeline-classical.yml`
- Mirror: `/home/rswan/mlops-root/mlops-project-template/classical/aml-cli-v2/mlops/github-actions/deploy-online-endpoint-pipeline-classical.yml`

- [ ] **Step 1: Pass storage authorization inputs to `create-endpoint`**

Set:

```yaml
storage_account_name: ${{ needs.get-config.outputs.storage_account }}
storage_container_name: monitoring
```

Do not move authorization after `create-deployment`: the deployment can receive traffic and invoke `score.py`, so permission must be ready first.

- [ ] **Step 2: Mirror and compare the project-template workflow**

The project and nested template copy must be byte-identical.

- [ ] **Step 3: Lint both workflows**

```bash
actionlint .github/workflows/deploy-online-endpoint-pipeline-classical.yml
actionlint /home/rswan/mlops-root/mlops-project-template/classical/aml-cli-v2/mlops/github-actions/deploy-online-endpoint-pipeline-classical.yml
```

- [ ] **Step 4: Commit and push the project-template change independently**

Push the nested source change before generating any clean-room project used for final proof.

---

## Phase 3 - Dev migration and live proof

### Task 5: Add the narrow Dev assignment and prove online logging

**Files:** None beyond prior tasks. This is a live workflow validation.

- [ ] **Step 1: Dispatch the Dev online deployment workflow**

Record the GitHub run ID. Verify `create-endpoint` reports either creating or finding exactly one container-scoped contributor assignment for the endpoint principal.

- [ ] **Step 2: Verify Azure state independently**

Assert all of the following:
- endpoint provisioning state is `Succeeded`;
- endpoint identity type is `system_assigned`;
- exactly one `Storage Blob Data Contributor` assignment exists for its principal at the `monitoring` container scope;
- the endpoint has not gained management-plane Contributor or Owner permissions from this change.

- [ ] **Step 3: Invoke the endpoint and verify the resulting blob**

Invoke with `data/taxi-request.json`. Record the prediction, locate the newly written Parquet blob under `inference-log/online/`, download it, and assert it contains the request features, prediction, timestamp, and endpoint metadata expected by `score.py`.

- [ ] **Step 4: Re-run deployment to prove idempotency**

The second run must succeed without adding another equivalent role assignment. Invoke again and verify a second log is written.

### Task 6: Remove obsolete broad Dev assignments

**Files:** None. This is a controlled Azure RBAC migration.

- [ ] **Step 1: Capture the exact old assignments**

Export assignment IDs for the endpoint principal's storage-account-scoped `Storage Blob Data Reader` and `Storage Blob Data Contributor` roles. Confirm they are manual/unmanaged and that no other principal is selected.

- [ ] **Step 2: Remove only those assignment IDs**

Delete by role-assignment ID, not by broad assignee/role filters. This preserves unrelated grants.

- [ ] **Step 3: Re-prove scoring after removal**

Invoke the endpoint and verify a new online Parquet log. Confirm the narrow container assignment remains and no storage-account-scoped data role remains for this endpoint principal.

- [ ] **Step 4: Re-run the monitor workflow**

Record the run ID and confirm it can still read baseline plus online and batch logs. Retraining behavior must remain controlled only by the existing drift result.

---

## Phase 4 - Clean-environment factory proof and Prod rollout

### Task 7: Prove a newly generated Dev project needs no manual storage setup

**Files:** Generated test project or disposable environment configuration; do not modify the reference project to fake this proof.

- [ ] **Step 1: Generate a fresh classical AML CLI v2 project from `mlops-project-template`**

Use a unique postfix and the factory's normal sparse-checkout path. Confirm the generated project contains the monitoring container resource and online caller inputs.

- [ ] **Step 2: Deploy infrastructure from empty state**

Verify Terraform creates the private `monitoring` container. No manual `az storage container create` or role-assignment command is allowed.

- [ ] **Step 3: Train and deploy batch plus online paths**

Verify baseline, batch inference, and online inference blobs are written. Assert the online endpoint has only the container-scoped contributor assignment introduced by this plan.

- [ ] **Step 4: Tear-down decision remains explicit**

Do not delete the proof environment without user approval. Record resource names, run IDs, and estimated ongoing cost so cleanup can be decided safely.

### Task 8: Roll the same migration through Prod

**Files:** None beyond prior committed changes.

- [ ] **Step 1: Review the Prod Terraform plan**

If the Prod container already exists, import it before apply exactly as in Dev. Stop on any unrelated replacement or deletion.

- [ ] **Step 2: Deploy the Prod online endpoint workflow**

Verify the container-scoped assignment, invoke the endpoint, and inspect the resulting log.

- [ ] **Step 3: Remove obsolete broad Prod grants only after proof**

Delete exact assignment IDs and re-run the smoke test.

---

## Phase 5 - Documentation and closure

### Task 9: Record ownership, evidence, and rollback

**Files:**
- Modify: `README.md`
- Modify: `.serena/memories/follow_up_gaps.md`
- Update the active execution ledger used for this plan.

- [ ] **Step 1: Document the ownership boundary**

State that Terraform owns the container, endpoint deployment owns the endpoint system identity, and the shared endpoint workflow owns the identity's container-scoped RBAC assignment.

- [ ] **Step 2: Record live evidence**

Capture Terraform plan/apply identifiers, GitHub run IDs, endpoint principal IDs, exact RBAC scopes, and representative blob paths for Dev, clean environment, and Prod.

- [ ] **Step 3: Document rollback**

Rollback order:
1. restore the previous workflow caller if authorization automation is defective;
2. temporarily restore the endpoint's account-scoped contributor assignment only if scoring is broken;
3. never destroy the monitoring container as rollback because it contains baseline and inference evidence;
4. fix and re-run the narrow assignment workflow, then remove the temporary broad grant.

- [ ] **Step 4: Close the P0 backlog item only when all done evidence exists**

Required evidence:
- clean Terraform deployment creates the private container;
- online endpoint deployment creates one narrow role assignment automatically;
- online and batch logs are written successfully;
- monitor workflow reads them successfully;
- repeat deployment is idempotent;
- no manual account-scoped endpoint storage assignments remain;
- project, project-template, and shared-template changes are committed and pushed.
