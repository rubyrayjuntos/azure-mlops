Snapshot as of 2026-08-10. Re-verify with `az` before trusting for anything consequential —
this is a point-in-time record, not a live view.

## Dev
- `rg-azmlops-0001dev`, workspace `mlw-azmlops-0001dev`.
- `cpu-cluster` (training, STANDARD_D4S_V3, dedicated) and `batch-cluster` (batch scoring,
  STANDARD_D4S_V3, dedicated tier — see `mem:known_bugs` item 7 for why not low_priority).
- Model `taxi-model` registered. Batch endpoint `taxi-gha-bep-azmlops-0001dev`, provisioning
  state Succeeded.
- No private endpoints (deliberately torn down — see below).

## Prod
- `rg-azmlops-0001prod`, workspace `mlw-azmlops-0001prod`. Same shape as Dev. First-ever
  prod deploy landed clean on the first attempt because Dev had already surfaced every
  environment bug in `mem:known_bugs`.

## Design decisions baked into current state
- `enable_private_endpoints` defaults to `false` (Terraform variable). Upstream had it
  hardcoded `true` with a "temporarily set for v1.2.0 testing" comment; reverted
  deliberately to avoid VNet/private-endpoint/DNS-zone sprawl for dev/demo use. Re-enabling
  it and re-applying to an environment that already has resources is destructive (tears
  down and recreates ~29 resources) — always run `terraform plan` first and confirm before
  `apply` if flipping this on an existing deployment.
- Online endpoints were not deployed in either environment — explicitly deferred as
  optional scope, not a gap.

## Known gap (not a defect in what's deployed)
Local `az ml batch-endpoint invoke` CLI testing on this dev machine hits
`Tenant mismatch: Token tenant does not match resource tenant`, persisting across a full
`az account clear` + fresh interactive `az login` + MSAL token cache file deletion +
another fresh login. Root cause unresolved. The deployed endpoints are independently
confirmed healthy via `az ml batch-endpoint show --query provisioning_state` in both
environments — use Azure ML Studio's endpoint Test tab for local inference verification
instead of the CLI on this machine.
