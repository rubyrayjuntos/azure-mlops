# Read-only MLOps visibility MVP

This directory provides a local, read-only visibility slice for the current Azure ML project. It correlates recent GitHub Actions runs with recent Azure ML jobs in a normalized JSON contract and renders a small dashboard. It does not dispatch workflows, deploy resources, or mutate Azure state.

## Run it

From the repository root, with authenticated `gh` and `az` CLIs:

```bash
python3 visibility/collect_status.py \
  --resource-group rg-azmlops-0001dev \
  --workspace mlw-azmlops-0001dev \
  --storage-account stazmlops0001dev
python3 visibility/server.py
```

Open <http://127.0.0.1:8765/dashboard.html>. Port 8765 is used by default to avoid common local services on port 8080. Use `--port 8766` if needed. The collector uses only `gh run list` and `az ml job list`. If either CLI or authentication is unavailable, the dashboard still starts and reports the warning.

The collector and server are separate commands: run the collector whenever you want a fresh snapshot, then keep the server terminal open while viewing the dashboard. If the browser says it cannot load the page, restart `python3 visibility/server.py` and revisit the URL.

The generated `visibility/status.json` is local runtime state and is ignored by Git. The public contract is implemented in `status.py`; a future managed service can replace `collect_status.py` without changing the dashboard contract.

## Current boundary

The collector reports GitHub workflow jobs and steps, Azure ML parent jobs and child-job stages, model registry versions/tags, MLflow evaluation metrics, endpoint provisioning state, and compute posture. If the local environment has the monitoring dependencies, `--storage-account` also runs the read-only drift check; otherwise it falls back to a `MONITORING_STATUS` found in a GitHub monitor-run log. It remains snapshot-based: live log streaming and automatic refresh are not yet implemented. No claims of live Azure validation are made by this local dashboard.

## Changelog

| Version | Created | Modified | Author | Notes |
|---|---|---|---|---|
| 0.1.0 | 2026-08-10 | 2026-08-10 | Codex | Added read-only status contract, CLI collector, local dashboard, and test. |
