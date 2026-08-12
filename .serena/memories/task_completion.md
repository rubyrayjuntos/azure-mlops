A change to this project's pipeline/infra code is not done until:

1. Every changed `.github/workflows/*.yml` passes `actionlint` clean (see
  `mem:suggested_commands`).
2. If the changed workflow file also exists in `mlops-project-template` (true for nearly
   all of them — see `mem:architecture`'s propagation rule), the same fix is applied to the
   fork's nested source path AND this repo's copy, both committed and pushed.
3. For anything touching Azure resources: don't trust a green GitHub Actions checkmark
   alone. Independently verify against live Azure state — `az resource list`,
   `az ml model list`, `az ml batch-endpoint show --query provisioning_state`, etc. — since
   a job can report success while the underlying resource ends up in a bad state (see
   `mem:known_bugs` item 8).
4. For Terraform variable changes affecting already-deployed infrastructure: run
   `terraform plan` (never a blind `apply`) and confirm the change set before applying,
   especially anything touching `enable_private_endpoints` (see `mem:deployment_state`).
