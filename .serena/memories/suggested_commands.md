## Lint a workflow file before committing
`actionlint <path/to/file.yml>` (or bare `actionlint` for the whole repo). Expect no
output, exit 0.

## Trigger and watch a GitHub Actions pipeline
```
gh workflow run <name>.yml --ref <dev|main> [-f key=value ...]
gh run list --workflow=<name>.yml --branch <branch> --limit 1 --json databaseId,status,conclusion
gh run view <run-id> --json status,conclusion,jobs
```

## Pull a failed GH Actions job's raw log
`gh api repos/rubyrayjuntos/azure-mlops/actions/jobs/<job-id>/logs`
(get `<job-id>` from `gh api repos/.../actions/runs/<run-id>/jobs`)

## Diagnose a failed AML pipeline step (the GH Actions wrapper log only shows polling
## output like "Running" / "Failed" — the real error is in the AML job's own logs)
1. Find the failed child step:
   `az ml job list --parent-job-name <run-name> --resource-group <rg> --workspace-name <ws>`
2. Your own user likely lacks data-plane storage access (only the shared service principal
   has it via Terraform). Grant temporarily:
   `az role assignment create --assignee $(az ad signed-in-user show --query id -o tsv) --role "Storage Blob Data Reader" --scope <storage-account-resource-id>`
   Wait ~20-100s for RBAC propagation (poll, don't assume instant).
3. `az ml job download -n <child-job-name> --resource-group <rg> --workspace-name <ws> --download-path <dir>`
4. Read `<dir>/artifacts/user_logs/std_log.txt` for the actual Python traceback.
5. Remove the temporary role assignment afterward (`az role assignment delete ...`).

## Recover from a compute-creation failure
`az ml compute delete --name <name> --resource-group <rg> --workspace-name <ws> --yes`
then wait for it to actually disappear (`az ml compute show` starts erroring) before
retrying — see `mem:known_bugs` item 8.
