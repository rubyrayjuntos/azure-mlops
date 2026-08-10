# Monitor + Retrain + Champion/Challenger Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two gaps Microsoft's own accelerator explicitly leaves open (model monitoring, automated retraining) by finishing wiring that's already half-built in this codebase, without adopting a new platform or provisioning new billed infrastructure.

**Architecture:** Inference scripts (batch + online) log predictions and inputs to the existing storage account. A baseline reference snapshot is captured from training data every time a model is promoted (the promotion gate already exists in `evaluate.py`/`register.py` — this plan doesn't touch that logic, only adds a snapshot step alongside it). A scheduled workflow compares recent inference logs against the current baseline using standard two-sample statistical tests (no new database), and triggers retraining as a reusable-workflow call when drift is detected. A successful retrain that results in promotion automatically triggers redeployment, closing the loop.

**Tech Stack:** Python (scipy for drift tests, `azure-storage-blob`/`azure-identity` for logging), GitHub Actions reusable workflows (`workflow_call`), existing OIDC identity model — no new Azure resources, no new secrets.

## Global Constraints

- **No ADX/Kusto.** Confirmed by reading `infrastructure/modules/data-explorer/main.tf`: the accelerator's built-in `enable_monitoring` Terraform flag provisions a real, billed `Standard_D11_v2` Kusto cluster. Explicitly rejected — this plan uses the existing storage account instead, for both inference logs and the baseline snapshot.
- **No new secrets or PATs.** All new cross-workflow triggering uses `workflow_call` (reusable-workflow) references to workflow files already in this repo, matching the existing OIDC-only posture. The default `GITHUB_TOKEN` cannot trigger a separate workflow's `workflow_dispatch` via the API by design (to prevent infinite loops) — `workflow_call` sidesteps this entirely and needs no token.
- **No new RBAC grants needed.** Verified live: the compute clusters (`cpu-cluster`, `batch-cluster`) run under `uai-azmlops-0001dev`, which already has `Storage Blob Data Contributor` on `stazmlops0001<env>` (granted by Terraform's `aml-workspace` module for a different reason, but the grant is broad enough to cover this use case too).
- **`mlops/azureml/deploy/batch/conda.yml` is the environment `batch-deployment.yml` actually uses** — confirmed by reading `batch-deployment.yml`'s inline `environment:` block. `batch-conda.yml`, `batch-env.yml`, and `batch-environment.yml` in the same directory are unused leftover files from an earlier template iteration; do not confuse them with the real one, and do not modify them as part of this plan.
- **`online-deployment.yml` currently has no `code_configuration` or `environment` block at all** — it relies on Azure ML's automatic no-code MLflow deployment. That's why `score.py` was empty and unused, not a bug in isolation. This plan adds both blocks so the custom scoring script (with logging) actually runs.
- **`prep.py`'s train/val/test split is unseeded** (`np.random.rand(len(data))` with no seed) — every pipeline run produces a different split, so the champion/challenger comparison in `evaluate.py` is currently noisy: two runs on identical code can produce different scores purely from split randomness. Fixed in Task 1 (`np.random.seed(42)`) because the drift baseline and the promotion gate both depend on comparisons being meaningful, not incidental scope creep.
- **Existing champion/challenger and promotion-gate logic in `evaluate.py`/`register.py` is not touched.** It already does the right thing: every new model is scored against every previously-registered version, and only registered if it beats the incumbent. This plan only adds a baseline-snapshot step alongside that existing gate.
- **Propagation rule (established convention, see `mem:architecture` in Serena):** every change under `mlops/` or `.github/workflows/` in this repo must also land in the `mlops-project-template` fork's nested source (`classical/aml-cli-v2/mlops/...` and `classical/aml-cli-v2/mlops/github-actions/...`), so future projects generated from the factory inherit this loop too.
- **Scope: Dev first, Prod after Dev is verified working**, matching the pattern from the original deployment plan.
- Blob layout in the existing storage account:
  - `monitoring/inference-log/<endpoint-type>/<YYYY>/<MM>/<DD>/<uuid>.parquet` — one file per batch run or per online request batch.
  - `monitoring/baseline/reference.json` — overwritten each time a model is promoted; holds per-feature summary stats from that promotion's test set.

---

## Phase 1 — Reproducibility and baseline snapshot capability

### Task 1: Seed the train/val/test split

**Files:**
- Modify: `data-science/src/prep.py`

- [ ] **Step 1: Add a fixed seed before the split**

In `prep.py`, right before `random_data = np.random.rand(len(data))`, add:
```python
    np.random.seed(42)
```

- [ ] **Step 2: Commit**

```bash
cd /home/rswan/azure-mlops
git add data-science/src/prep.py
git commit -m "Seed the train/val/test split for reproducible evaluation

evaluate.py's champion/challenger comparison and this plan's drift
baseline both depend on comparisons being meaningful across runs.
An unseeded split made every run's test set different, adding pure
randomness to what should be a signal."
```

(Push happens together with Task 3's pipeline.yml change, since both are needed before the pipeline can run meaningfully — or push now independently, either is fine; this commit is self-contained.)

```bash
git push origin dev
```

### Task 2: Baseline snapshot script

**Files:**
- Create: `data-science/src/snapshot_baseline.py`

Writes per-feature reference statistics from the test set to blob storage, but only when the model was actually promoted (mirrors `register.py`'s own `deploy_flag` check — same input, same gate, no new signal needed).

- [ ] **Step 1: Write the script**

```python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Snapshots per-feature reference statistics from the test set to blob
storage, for later drift comparison against production inference data.
Only runs when the model was promoted (same deploy_flag gate as register.py).
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

NUMERIC_COLS = [
    "distance",
    "dropoff_latitude",
    "dropoff_longitude",
    "passengers",
    "pickup_latitude",
    "pickup_longitude",
    "pickup_weekday",
    "pickup_month",
    "pickup_monthday",
    "pickup_hour",
    "pickup_minute",
    "pickup_second",
    "dropoff_weekday",
    "dropoff_month",
    "dropoff_monthday",
    "dropoff_hour",
    "dropoff_minute",
    "dropoff_second",
]
CAT_NOM_COLS = ["store_forward", "vendor"]


def parse_args():
    parser = argparse.ArgumentParser("snapshot_baseline")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test dataset (parquet dir)")
    parser.add_argument("--evaluation_output", type=str, required=True, help="Path containing the deploy_flag file")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--storage_account", type=str, required=True, help="Storage account name, e.g. stazmlops0001dev")
    parser.add_argument("--container", type=str, default="azureml-blobstore", help="Blob container to write to")
    return parser.parse_args()


def compute_reference_stats(df: pd.DataFrame) -> dict:
    stats = {"numeric": {}, "categorical": {}}
    for col in NUMERIC_COLS:
        if col in df.columns:
            stats["numeric"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "values_sample": df[col].sample(min(500, len(df)), random_state=42).tolist(),
            }
    for col in CAT_NOM_COLS:
        if col in df.columns:
            counts = df[col].value_counts(normalize=True)
            stats["categorical"][col] = counts.to_dict()
    return stats


def main(args):
    deploy_flag_path = Path(args.evaluation_output) / "deploy_flag"
    with open(deploy_flag_path, "rb") as f:
        deploy_flag = int(f.read())

    if deploy_flag != 1:
        print("Model was not promoted - skipping baseline snapshot.")
        return

    test_data = pd.read_parquet(Path(args.test_data))
    stats = compute_reference_stats(test_data)
    payload = {
        "model_name": args.model_name,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(test_data),
        "stats": stats,
    }

    credential = DefaultAzureCredential()
    account_url = f"https://{args.storage_account}.blob.core.windows.net"
    blob_service = BlobServiceClient(account_url=account_url, credential=credential)
    blob_client = blob_service.get_blob_client(container=args.container, blob="monitoring/baseline/reference.json")
    blob_client.upload_blob(json.dumps(payload, indent=2), overwrite=True)
    print(f"Baseline snapshot written to {args.container}/monitoring/baseline/reference.json "
          f"({payload['row_count']} rows, captured {payload['captured_at']}).")


if __name__ == "__main__":
    main(parse_args())
```

- [ ] **Step 2: Add dependencies to the training environment**

`data-science/environment/train-conda.yml` already has `azure-identity==1.19.0`. Add `azure-storage-blob` right after it:

```yaml
      - azure-identity==1.19.0
      - azure-storage-blob==12.19.0
```

- [ ] **Step 3: Validate YAML and Python syntax**

```bash
cd /home/rswan/azure-mlops
python3 -m py_compile data-science/src/snapshot_baseline.py && echo "syntax OK"
python3 -c "import yaml; yaml.safe_load(open('data-science/environment/train-conda.yml'))" && echo "YAML valid"
```

- [ ] **Step 4: Commit**

```bash
git add data-science/src/snapshot_baseline.py data-science/environment/train-conda.yml
git commit -m "Add baseline-snapshot script for drift comparison

Writes per-feature reference stats (mean/std for numeric columns,
frequency distribution for categorical columns) from the test set to
blob storage, whenever a model is promoted. This becomes the
comparison point for the drift check in a later task."
git push origin dev
```

### Task 3: Wire the snapshot into the training pipeline

**Files:**
- Modify: `mlops/azureml/train/pipeline.yml`

**Interfaces:**
- Consumes: `snapshot_baseline.py`'s CLI args from Task 2.
- Produces: nothing new for later tasks to consume (this job's effect is the side-effect blob write).

- [ ] **Step 1: Add the `snapshot_baseline` job**

Add this job to `pipeline.yml`, after `register_model`:

```yaml
  snapshot_baseline:
    name: snapshot_baseline
    display_name: snapshot-baseline
    code: ../../../data-science/src
    command: >-
      python snapshot_baseline.py
      --test_data ${{inputs.test_data}}
      --evaluation_output ${{inputs.evaluation_output}}
      --model_name ${{inputs.model_name}}
      --storage_account ${{inputs.storage_account}}
    environment: azureml:taxi-train-env@latest
    inputs:
      test_data: ${{parent.jobs.prep_data.outputs.test_data}}
      evaluation_output: ${{parent.jobs.evaluate_model.outputs.evaluation_output}}
      model_name: "taxi-model"
      storage_account: ${{parent.inputs.storage_account}}
    outputs: {}
```

- [ ] **Step 2: Add the `storage_account` pipeline input**

In the `inputs:` block near the top of the file (alongside `input`, `enable_monitoring`, `table_name`), add:

```yaml
  storage_account:
    type: string
```

- [ ] **Step 3: Pass `storage_account` from the caller**

In `.github/workflows/deploy-model-training-pipeline-classical.yml`'s `run-model-training-pipeline` job (which calls `run-pipeline.yml`), the `parameters-file` mechanism doesn't currently pass extra `-var`-style overrides — check how `run-pipeline.yml` invokes `az ml job create` (it does `az ml job create --file <parameters-file> ...` with no `--set` overrides currently). Add a `--set inputs.storage_account=<value>` override, OR simplest: since `get-config`'s `read-yaml` output already exposes a storage-account-shaped value indirectly via naming convention, just pass it explicitly. Confirm the exact mechanism by reading `~/mlops-root/mlops-templates/.github/workflows/run-pipeline.yml`'s `az ml job create` invocation before choosing `--set` vs. editing the reusable workflow to accept a new input — if the reusable workflow needs a new input added, that's a `mlops-templates` fork change (add an optional `job-inputs` string input, defaulted to empty, appended to the `az ml job create` command as `--set <value>` when non-empty), not a `pipeline.yml`-only change. Do this investigation as part of this step, before writing the fix — do not guess the mechanism.

- [ ] **Step 4: Lint and validate**

```bash
cd /home/rswan/azure-mlops
python3 -c "import yaml; yaml.safe_load(open('mlops/azureml/train/pipeline.yml'))" && echo "pipeline.yml valid"
```

- [ ] **Step 5: Commit**

```bash
git add mlops/azureml/train/pipeline.yml .github/workflows/deploy-model-training-pipeline-classical.yml
git commit -m "Wire snapshot_baseline into the training pipeline

Runs after register_model, using the same test_data and deploy_flag
already computed by evaluate_model - no duplicate evaluation logic."
git push origin dev
```

---

## Phase 2 — Inference logging (batch + online)

### Task 4: Log inference data from the batch scorer

**Files:**
- Modify: `mlops/azureml/deploy/batch/code/batch_driver.py`
- Modify: `mlops/azureml/deploy/batch/conda.yml`

- [ ] **Step 1: Add logging to `batch_driver.py`**

Add near the top imports:
```python
import json
import uuid
from datetime import datetime, timezone
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
```

In `init()`, after the model loads successfully, add:
```python
    global blob_service, storage_account, container
    storage_account = os.environ.get("MONITORING_STORAGE_ACCOUNT")
    container = os.environ.get("MONITORING_CONTAINER", "azureml-blobstore")
    if storage_account:
        credential = DefaultAzureCredential()
        blob_service = BlobServiceClient(
            account_url=f"https://{storage_account}.blob.core.windows.net",
            credential=credential,
        )
    else:
        blob_service = None
        logger.warning("MONITORING_STORAGE_ACCOUNT not set - inference logging disabled.")
```

In `run()`, after computing `predictions` for each file (inside the `try` block, after the `for pred in predictions:` loop, before moving to the next file), add:
```python
            if blob_service is not None:
                log_df = data.copy()
                log_df["prediction"] = predictions
                log_df["logged_at"] = datetime.now(timezone.utc).isoformat()
                now = datetime.now(timezone.utc)
                blob_path = (
                    f"monitoring/inference-log/batch/{now:%Y}/{now:%m}/{now:%d}/{uuid.uuid4()}.parquet"
                )
                blob_client = blob_service.get_blob_client(container=container, blob=blob_path)
                blob_client.upload_blob(log_df.to_parquet(index=False), overwrite=True)
```

- [ ] **Step 2: Add dependencies to `conda.yml`** (the file `batch-deployment.yml` actually uses — see Global Constraints)

```yaml
      - azure-identity==1.19.0
      - azure-storage-blob==12.19.0
```

- [ ] **Step 3: Pass the storage account name to the deployment environment**

In `mlops/azureml/deploy/batch/batch-deployment.yml`, add an `environment_variables:` block:
```yaml
environment_variables:
  MONITORING_STORAGE_ACCOUNT: "REPLACE_AT_DEPLOY_TIME"
```
Then in `.github/workflows/deploy-batch-endpoint-pipeline-classical.yml`'s `create-deployment` job, check whether `create-deployment.yml` (the reusable workflow) supports injecting environment variable overrides at deploy time via `--set environment_variables.MONITORING_STORAGE_ACCOUNT=<value>` (the AML CLI supports `--set` overrides on `az ml batch-deployment create`). If the reusable workflow doesn't currently expose a way to pass `--set` overrides, that's a small addition needed in `mlops-templates`'s `create-deployment.yml` (an optional `extra_args` string input, appended verbatim to the `az ml batch-deployment create`/`az ml online-deployment create` command when non-empty) — confirm by reading the current reusable workflow before deciding the exact mechanism, same as Task 3 Step 3.

- [ ] **Step 4: Validate**

```bash
cd /home/rswan/azure-mlops
python3 -m py_compile mlops/azureml/deploy/batch/code/batch_driver.py && echo "syntax OK"
python3 -c "import yaml; yaml.safe_load(open('mlops/azureml/deploy/batch/conda.yml'))" && echo "conda.yml valid"
python3 -c "import yaml; yaml.safe_load(open('mlops/azureml/deploy/batch/batch-deployment.yml'))" && echo "batch-deployment.yml valid"
```

- [ ] **Step 5: Commit**

```bash
git add mlops/azureml/deploy/batch/code/batch_driver.py mlops/azureml/deploy/batch/conda.yml mlops/azureml/deploy/batch/batch-deployment.yml .github/workflows/deploy-batch-endpoint-pipeline-classical.yml
git commit -m "Log batch inference inputs+predictions to blob storage

Feeds the drift-check script added later in this plan. Uses the
compute's existing user-assigned identity (already has Storage Blob
Data Contributor on the storage account - verified, no new RBAC
grant needed) via DefaultAzureCredential, not a connection string."
git push origin dev
```

### Task 5: Implement `score.py` and wire the online deployment to use it

**Files:**
- Create: `mlops/azureml/deploy/online/score.py`
- Create: `mlops/azureml/deploy/online/online-conda.yml`
- Modify: `mlops/azureml/deploy/online/online-deployment.yml`

- [ ] **Step 1: Write `score.py`** (mirrors `batch_driver.py`'s pattern, adapted for online's request/response shape)

```python
import os
import json
import uuid
import logging
from datetime import datetime, timezone

import mlflow
import pandas as pd

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init():
    global model, blob_service, storage_account, container

    model_dir = os.environ.get("AZUREML_MODEL_DIR")
    logger.info(f"AZUREML_MODEL_DIR: {model_dir}")
    model_path = model_dir if model_dir else "./model"

    try:
        model = mlflow.pyfunc.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise

    storage_account = os.environ.get("MONITORING_STORAGE_ACCOUNT")
    container = os.environ.get("MONITORING_CONTAINER", "azureml-blobstore")
    if storage_account:
        credential = DefaultAzureCredential()
        blob_service = BlobServiceClient(
            account_url=f"https://{storage_account}.blob.core.windows.net",
            credential=credential,
        )
    else:
        blob_service = None
        logger.warning("MONITORING_STORAGE_ACCOUNT not set - inference logging disabled.")


def run(raw_data):
    logger.info("Received scoring request")
    body = json.loads(raw_data)
    input_data = body["input_data"]
    df = pd.DataFrame(data=input_data["data"], columns=input_data["columns"])

    predictions = model.predict(df)

    if blob_service is not None:
        log_df = df.copy()
        log_df["prediction"] = predictions
        log_df["logged_at"] = datetime.now(timezone.utc).isoformat()
        now = datetime.now(timezone.utc)
        blob_path = f"monitoring/inference-log/online/{now:%Y}/{now:%m}/{now:%d}/{uuid.uuid4()}.parquet"
        try:
            blob_client = blob_service.get_blob_client(container=container, blob=blob_path)
            blob_client.upload_blob(log_df.to_parquet(index=False), overwrite=True)
        except Exception as e:
            # Logging failure must never break a live scoring request.
            logger.error(f"Inference logging failed (non-fatal): {str(e)}")

    return json.dumps(predictions.tolist())
```

- [ ] **Step 2: Create `online-conda.yml`** (mirrors `batch/conda.yml` plus the same two new packages)

```yaml
name: taxi-online-env
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip=25.0
  - pip:
      - mlflow==2.9.2
      - scikit-learn==1.5.2
      - numpy==1.26.4
      - cloudpickle==3.1.0
      - pyarrow==14.0.2
      - scipy==1.14.0
      - pandas>=1.3.0
      - azure-identity==1.19.0
      - azure-storage-blob==12.19.0
```

- [ ] **Step 3: Add `code_configuration` and `environment` to `online-deployment.yml`**

Replace the file's content with:
```yaml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: blue
endpoint_name: taxi-fare-online
model: azureml:taxi-model@latest
code_configuration:
  code: .
  scoring_script: score.py
environment:
  image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04
  conda_file: online-conda.yml
environment_variables:
  MONITORING_STORAGE_ACCOUNT: "REPLACE_AT_DEPLOY_TIME"
instance_type: Standard_DS3_v2
instance_count: 1
egress_public_network_access: enabled
```

- [ ] **Step 4: Wire the storage account name at deploy time**, same investigation as Task 4 Step 3 — this affects `.github/workflows/deploy-online-endpoint-pipeline-classical.yml`'s `create-deployment` job. Reuse whatever mechanism (`extra_args` or similar) gets added to `mlops-templates`'s `create-deployment.yml` in Task 4; don't design a second, different mechanism for online.

- [ ] **Step 5: Validate**

```bash
cd /home/rswan/azure-mlops
python3 -m py_compile mlops/azureml/deploy/online/score.py && echo "syntax OK"
python3 -c "import yaml; yaml.safe_load(open('mlops/azureml/deploy/online/online-conda.yml'))" && echo "online-conda.yml valid"
python3 -c "import yaml; yaml.safe_load(open('mlops/azureml/deploy/online/online-deployment.yml'))" && echo "online-deployment.yml valid"
```

- [ ] **Step 6: Commit**

```bash
git add mlops/azureml/deploy/online/score.py mlops/azureml/deploy/online/online-conda.yml mlops/azureml/deploy/online/online-deployment.yml .github/workflows/deploy-online-endpoint-pipeline-classical.yml
git commit -m "Implement score.py and wire the online deployment to use it

online-deployment.yml previously had no code_configuration or
environment block at all - it silently relied on Azure ML's implicit
no-code MLflow deployment, which is why score.py was an empty stub.
Adding both blocks so the custom scorer (with the same inference
logging as the batch path) actually runs."
git push origin dev
```

---

## Phase 3 — Drift detection

### Task 6: Drift-check script

**Files:**
- Create: `data-science/src/check_drift.py`

**Interfaces:**
- Consumes: `monitoring/baseline/reference.json` (Task 2's output shape) and `monitoring/inference-log/**/*.parquet` (Tasks 4-5's output shape).
- Produces: exit code 0 always; prints `DRIFT_DETECTED=true` or `DRIFT_DETECTED=false` as the last line of stdout, for the calling workflow to parse.

- [ ] **Step 1: Write the script**

```python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Compares recent production inference data against the stored training
baseline. Prints DRIFT_DETECTED=true/false as the last stdout line.

Numeric features: two-sample Kolmogorov-Smirnov test (scipy.stats.ks_2samp).
Categorical features: chi-square goodness-of-fit against the baseline's
frequency distribution (scipy.stats.chisquare).

A feature is flagged as drifted if its test's p-value is below --p-threshold
(default 0.05). Overall DRIFT_DETECTED=true if at least --min-drifted-features
(default 1) features are flagged.
"""
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from io import BytesIO

import pandas as pd
from scipy.stats import ks_2samp, chisquare

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

NUMERIC_COLS = [
    "distance", "dropoff_latitude", "dropoff_longitude", "passengers",
    "pickup_latitude", "pickup_longitude", "pickup_weekday", "pickup_month",
    "pickup_monthday", "pickup_hour", "pickup_minute", "pickup_second",
    "dropoff_weekday", "dropoff_month", "dropoff_monthday",
    "dropoff_hour", "dropoff_minute", "dropoff_second",
]
CAT_NOM_COLS = ["store_forward", "vendor"]


def parse_args():
    p = argparse.ArgumentParser("check_drift")
    p.add_argument("--storage_account", type=str, required=True)
    p.add_argument("--container", type=str, default="azureml-blobstore")
    p.add_argument("--lookback_days", type=int, default=7)
    p.add_argument("--p_threshold", type=float, default=0.05)
    p.add_argument("--min_drifted_features", type=int, default=1)
    p.add_argument("--report_output", type=str, default=None, help="Optional path to write a JSON report")
    return p.parse_args()


def load_baseline(blob_service, container):
    client = blob_service.get_blob_client(container=container, blob="monitoring/baseline/reference.json")
    if not client.exists():
        return None
    return json.loads(client.download_blob().readall())


def load_recent_inference_data(blob_service, container, lookback_days):
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    prefix = "monitoring/inference-log/"
    container_client = blob_service.get_container_client(container)
    frames = []
    for blob in container_client.list_blobs(name_starts_with=prefix):
        if blob.last_modified is not None and blob.last_modified < cutoff:
            continue
        data = container_client.download_blob(blob.name).readall()
        frames.append(pd.read_parquet(BytesIO(data)))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def main(args):
    credential = DefaultAzureCredential()
    blob_service = BlobServiceClient(
        account_url=f"https://{args.storage_account}.blob.core.windows.net",
        credential=credential,
    )

    baseline = load_baseline(blob_service, args.container)
    if baseline is None:
        print("No baseline found yet - nothing to compare against. Treating as no drift.")
        print("DRIFT_DETECTED=false")
        return

    recent = load_recent_inference_data(blob_service, args.container, args.lookback_days)
    if recent is None or len(recent) == 0:
        print(f"No inference data logged in the last {args.lookback_days} days. Treating as no drift.")
        print("DRIFT_DETECTED=false")
        return

    drifted_features = []
    report = {"numeric": {}, "categorical": {}}

    for col, ref in baseline["stats"]["numeric"].items():
        if col not in recent.columns:
            continue
        stat, p_value = ks_2samp(ref["values_sample"], recent[col].dropna())
        report["numeric"][col] = {"statistic": float(stat), "p_value": float(p_value)}
        if p_value < args.p_threshold:
            drifted_features.append(col)

    for col, ref_dist in baseline["stats"]["categorical"].items():
        if col not in recent.columns:
            continue
        observed_counts = recent[col].value_counts()
        categories = list(ref_dist.keys())
        expected = [ref_dist[c] * len(recent) for c in categories]
        observed = [observed_counts.get(c, 0) for c in categories]
        if sum(expected) == 0:
            continue
        stat, p_value = chisquare(observed, f_exp=expected)
        report["categorical"][col] = {"statistic": float(stat), "p_value": float(p_value)}
        if p_value < args.p_threshold:
            drifted_features.append(col)

    report["drifted_features"] = drifted_features
    report["inference_rows_compared"] = len(recent)
    report["baseline_captured_at"] = baseline["captured_at"]

    if args.report_output:
        with open(args.report_output, "w") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    drift_detected = len(drifted_features) >= args.min_drifted_features
    print(f"DRIFT_DETECTED={'true' if drift_detected else 'false'}")


if __name__ == "__main__":
    sys.exit(main(parse_args()) or 0)
```

- [ ] **Step 2: Validate**

```bash
cd /home/rswan/azure-mlops
python3 -m py_compile data-science/src/check_drift.py && echo "syntax OK"
```

- [ ] **Step 3: Commit**

```bash
git add data-science/src/check_drift.py
git commit -m "Add drift-check script (KS test + chi-square, no ADX)

Compares recent inference logs against the stored baseline snapshot.
Prints DRIFT_DETECTED=true/false as the last stdout line for the
calling workflow to parse. Missing baseline or no recent inference
data both resolve to no-drift rather than failing, since both are
expected states early in a project's life."
git push origin dev
```

---

## Phase 4 — Workflow orchestration

### Task 7: Investigate and extend `run-pipeline.yml` and `create-deployment.yml` for parameter passthrough

This task resolves the two "confirm by reading, don't guess" steps flagged in Tasks 3 and 4/5.

**Files:**
- Modify: `~/mlops-root/mlops-templates/.github/workflows/run-pipeline.yml`
- Modify: `~/mlops-root/mlops-templates/.github/workflows/create-deployment.yml`

- [ ] **Step 1: Read both files' current `az ml job create` / `az ml *-deployment create` invocations**

```bash
cat ~/mlops-root/mlops-templates/.github/workflows/run-pipeline.yml
cat ~/mlops-root/mlops-templates/.github/workflows/create-deployment.yml
```

- [ ] **Step 2: Add an optional passthrough input to each**

For `run-pipeline.yml`: add an optional `job-inputs` string input (default `""`) to `on.workflow_call.inputs`, and append `${{ inputs.job-inputs != '' && format('--set {0}', inputs.job-inputs) || '' }}` (or the shell-equivalent conditional, matching the file's existing style) to the `az ml job create` command line.

For `create-deployment.yml`: add an optional `extra_args` string input (default `""`) the same way, appended to whichever `az ml *-deployment create` command the file runs.

- [ ] **Step 3: Lint**

```bash
cd ~/mlops-root/mlops-templates
actionlint .github/workflows/run-pipeline.yml .github/workflows/create-deployment.yml
```

- [ ] **Step 4: Commit and push**

```bash
git add .github/workflows/run-pipeline.yml .github/workflows/create-deployment.yml
git commit -m "Add optional parameter/arg passthrough for job and deployment creation

Needed so callers can inject the storage-account name (for inference
logging and baseline snapshotting) without hardcoding it into the
reusable workflow itself. Defaults to empty/no-op for existing callers."
git push origin main
```

- [ ] **Step 5: Go back and finish Task 3 Step 3, Task 4 Step 3, and Task 5 Step 4** using the mechanism just added (`job-inputs` / `extra_args`), now that it exists. Pass `inputs.storage_account=azureml:...` (or the correct AML CLI `--set` key path for a pipeline job input) and `environment_variables.MONITORING_STORAGE_ACCOUNT=<storage-account-name>` respectively, sourced from `needs.get-config.outputs` (the storage account name is derivable from the existing `resource_group`/`aml_workspace` naming convention, or add a `storage_account` output to `read-yaml.yml`/`read_yaml_action` if not already exposed — check `read_yaml_action/index.js`'s outputs first).

### Task 8: Add `workflow_call` trigger to the training pipeline, alongside the existing `workflow_dispatch`

**Files:**
- Modify: `classical/aml-cli-v2/mlops/github-actions/deploy-model-training-pipeline-classical.yml` (in `~/mlops-root/mlops-project-template`)
- Modify: `.github/workflows/deploy-model-training-pipeline-classical.yml` (in `azure-mlops`)

- [ ] **Step 1: Add `workflow_call` alongside `workflow_dispatch`**

```yaml
on:
  workflow_dispatch:
    inputs:
      skip_environment_registration:
        description: 'Skip environment registration'
        required: false
        type: boolean
        default: false
      skip_data_registration:
        description: 'Skip data registration'
        required: false
        type: boolean
        default: false
      skip_compute_creation:
        description: 'Skip compute creation'
        required: false
        type: boolean
        default: false
  workflow_call:
    outputs:
      promoted:
        description: "Whether a new model version was registered in this run"
        value: ${{ jobs.check-promotion.outputs.promoted }}
```

(`workflow_call` doesn't need its own `inputs:` here since this workflow doesn't take any beyond the optional skip flags, which only make sense from a human `workflow_dispatch` — leave `skip_*` unset/default when called via `workflow_call`.)

- [ ] **Step 2: Add the `check-promotion` job**

Add after `run-model-training-pipeline`:
```yaml
  check-promotion:
    needs: [get-config, run-model-training-pipeline]
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    outputs:
      promoted: ${{ steps.check.outputs.promoted }}
    steps:
      - name: azure-login
        uses: azure/login@7184910d9eb2b1c5e48f7073824a90609bb9b6d6 # v2
        with:
          client-id: ${{ secrets.AZURE_CLIENT_ID }}
          tenant-id: ${{ secrets.AZURE_TENANT_ID }}
          subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
      - id: check
        run: |
          az extension add -n ml -y
          RUN_NAME=$(az ml job list --resource-group ${{ needs.get-config.outputs.resource_group }} --workspace-name ${{ needs.get-config.outputs.aml_workspace }} --query "sort_by(@, &creation_context.created_at)[-1].name" -o tsv)
          CHILD_NAME=$(az ml job list --parent-job-name "$RUN_NAME" --resource-group ${{ needs.get-config.outputs.resource_group }} --workspace-name ${{ needs.get-config.outputs.aml_workspace }} --query "[?display_name=='register-model'].name" -o tsv)
          rm -rf /tmp/promotion-check && mkdir -p /tmp/promotion-check
          az ml job download -n "$CHILD_NAME" --resource-group ${{ needs.get-config.outputs.resource_group }} --workspace-name ${{ needs.get-config.outputs.aml_workspace }} --download-path /tmp/promotion-check --output-name model_info_output_path || true
          if [ -f /tmp/promotion-check/named-outputs/model_info_output_path/model_info.json ]; then
            echo "promoted=true" >> "$GITHUB_OUTPUT"
          else
            echo "promoted=false" >> "$GITHUB_OUTPUT"
          fi
```

- [ ] **Step 3: Add the conditional redeploy jobs**

```yaml
  redeploy-batch:
    needs: check-promotion
    if: needs.check-promotion.outputs.promoted == 'true'
    uses: ./.github/workflows/deploy-batch-endpoint-pipeline-classical.yml
    secrets: inherit

  redeploy-online:
    needs: check-promotion
    if: needs.check-promotion.outputs.promoted == 'true'
    uses: ./.github/workflows/deploy-online-endpoint-pipeline-classical.yml
    secrets: inherit
```

- [ ] **Step 4: Lint**

```bash
cd ~/mlops-root/mlops-project-template
actionlint classical/aml-cli-v2/mlops/github-actions/deploy-model-training-pipeline-classical.yml
```

- [ ] **Step 5: Commit both repos**

```bash
cd ~/mlops-root/mlops-project-template
git add classical/aml-cli-v2/mlops/github-actions/deploy-model-training-pipeline-classical.yml
git commit -m "Detect promotion and auto-redeploy on success; add workflow_call trigger

Lets the training pipeline be invoked as a reusable workflow (for the
scheduled monitor-and-retrain workflow added later in this plan), and
closes the loop: a promoted model now automatically triggers a batch
and online redeploy instead of requiring a manual follow-up run."
git push origin main
```

```bash
cd /home/rswan/azure-mlops
cp ~/mlops-root/mlops-project-template/classical/aml-cli-v2/mlops/github-actions/deploy-model-training-pipeline-classical.yml .github/workflows/deploy-model-training-pipeline-classical.yml
actionlint .github/workflows/deploy-model-training-pipeline-classical.yml
git add .github/workflows/deploy-model-training-pipeline-classical.yml
git commit -m "Sync promotion-detection and auto-redeploy from the fork"
git push origin dev
```

### Task 9: Add `workflow_call` trigger to the batch and online deploy pipelines

**Files:**
- Modify: `classical/aml-cli-v2/mlops/github-actions/deploy-batch-endpoint-pipeline-classical.yml` and `deploy-online-endpoint-pipeline-classical.yml` (fork)
- Modify: same two files in `azure-mlops`

- [ ] **Step 1: Add `workflow_call:` alongside `workflow_dispatch:`** in both files (each currently has only `on: workflow_dispatch:` with no inputs):

```yaml
on:
  workflow_dispatch:
  workflow_call:
```

- [ ] **Step 2: Lint, commit, push — fork first, then sync to azure-mlops**, same pattern as every prior propagation task in this plan and the original deployment plan.

```bash
cd ~/mlops-root/mlops-project-template
actionlint classical/aml-cli-v2/mlops/github-actions/deploy-batch-endpoint-pipeline-classical.yml classical/aml-cli-v2/mlops/github-actions/deploy-online-endpoint-pipeline-classical.yml
git add classical/aml-cli-v2/mlops/github-actions/deploy-batch-endpoint-pipeline-classical.yml classical/aml-cli-v2/mlops/github-actions/deploy-online-endpoint-pipeline-classical.yml
git commit -m "Add workflow_call trigger so these can be invoked as reusable workflows"
git push origin main

cd /home/rswan/azure-mlops
cp ~/mlops-root/mlops-project-template/classical/aml-cli-v2/mlops/github-actions/deploy-batch-endpoint-pipeline-classical.yml .github/workflows/deploy-batch-endpoint-pipeline-classical.yml
cp ~/mlops-root/mlops-project-template/classical/aml-cli-v2/mlops/github-actions/deploy-online-endpoint-pipeline-classical.yml .github/workflows/deploy-online-endpoint-pipeline-classical.yml
actionlint .github/workflows/deploy-batch-endpoint-pipeline-classical.yml .github/workflows/deploy-online-endpoint-pipeline-classical.yml
git add .github/workflows/deploy-batch-endpoint-pipeline-classical.yml .github/workflows/deploy-online-endpoint-pipeline-classical.yml
git commit -m "Sync workflow_call trigger from the fork"
git push origin dev
```

### Task 10: Scheduled monitor-and-retrain workflow

**Files:**
- Create: `classical/aml-cli-v2/mlops/github-actions/monitor-and-retrain-classical.yml` (fork)
- Create: `.github/workflows/monitor-and-retrain-classical.yml` (azure-mlops)

- [ ] **Step 1: Write the workflow**

```yaml
name: monitor-and-retrain

on:
  workflow_dispatch:
  schedule:
    - cron: "0 6 * * 1"  # Mondays 06:00 UTC - adjust to your actual data velocity

jobs:
  set-env-branch:
    runs-on: ubuntu-latest
    outputs:
      config-file: ${{ steps.set-output-defaults.outputs.config-file }}
    steps:
      - id: set-prod-branch
        if: ${{ github.ref == 'refs/heads/main'}}
        run: echo "config_env=config-infra-prod.yml" >> $GITHUB_ENV
      - id: set-dev-branch
        if: ${{ github.ref != 'refs/heads/main'}}
        run: echo "config_env=config-infra-dev.yml" >> $GITHUB_ENV
      - id: set-output-defaults
        run: echo "config-file=$config_env" >> $GITHUB_OUTPUT

  get-config:
    needs: set-env-branch
    permissions:
      contents: read
    uses: rubyrayjuntos/mlops-templates/.github/workflows/read-yaml.yml@main
    with:
      file_name: ${{ needs.set-env-branch.outputs.config-file }}

  check-drift:
    needs: get-config
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    outputs:
      drift_detected: ${{ steps.drift.outputs.drift_detected }}
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4
      - uses: azure/login@7184910d9eb2b1c5e48f7073824a90609bb9b6d6 # v2
        with:
          client-id: ${{ secrets.AZURE_CLIENT_ID }}
          tenant-id: ${{ secrets.AZURE_TENANT_ID }}
          subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
      - name: Set up Python
        run: |
          python3 -m pip install --quiet scipy pandas pyarrow azure-identity azure-storage-blob
      - id: drift
        run: |
          # STORAGE_ACCOUNT: confirm the exact get-config output name from Task 7's investigation
          OUTPUT=$(python3 data-science/src/check_drift.py --storage_account ${{ needs.get-config.outputs.storage_account }})
          echo "$OUTPUT"
          echo "drift_detected=$(echo "$OUTPUT" | tail -1 | cut -d= -f2)" >> "$GITHUB_OUTPUT"

  retrain:
    needs: check-drift
    if: needs.check-drift.outputs.drift_detected == 'true'
    uses: ./.github/workflows/deploy-model-training-pipeline-classical.yml
    secrets: inherit
```

- [ ] **Step 2: Lint**

```bash
cd ~/mlops-root/mlops-project-template
actionlint classical/aml-cli-v2/mlops/github-actions/monitor-and-retrain-classical.yml
```

- [ ] **Step 3: Commit fork, then sync to azure-mlops**

```bash
cd ~/mlops-root/mlops-project-template
git add classical/aml-cli-v2/mlops/github-actions/monitor-and-retrain-classical.yml
git commit -m "Add scheduled drift-check + auto-retrain workflow

Closes the loop: check-drift runs on a schedule, and only invokes the
training pipeline (as a reusable workflow, no new secrets) when
drift crosses the threshold. A promoted retrain then auto-redeploys
via Task 8's check-promotion job."
git push origin main

cd /home/rswan/azure-mlops
cp ~/mlops-root/mlops-project-template/classical/aml-cli-v2/mlops/github-actions/monitor-and-retrain-classical.yml .github/workflows/monitor-and-retrain-classical.yml
actionlint .github/workflows/monitor-and-retrain-classical.yml
git add .github/workflows/monitor-and-retrain-classical.yml
git commit -m "Sync monitor-and-retrain workflow from the fork"
git push origin dev
```

---

## Phase 5 — Bootstrap and verify (Dev)

### Task 11: Run training once to establish the seeded baseline

- [ ] **Step 1: Trigger**

```bash
cd /home/rswan/azure-mlops
gh workflow run deploy-model-training-pipeline-classical.yml --ref dev -f skip_compute_creation=true
```

- [ ] **Step 2: Watch to completion, then verify the baseline was written**

```bash
az storage blob exists --account-name stazmlops0001dev --container-name azureml-blobstore --name monitoring/baseline/reference.json --auth-mode login
```
Expected: `"exists": true`.

- [ ] **Step 3: Verify `check-promotion` correctly detected the promotion and triggered redeploy** — check the run's job list for `redeploy-batch`/`redeploy-online` and their conclusions.

### Task 12: Deploy the (now-fixed) online endpoint and verify inference logging on both paths

- [ ] **Step 1: If `redeploy-online` didn't already deploy it in Task 11, trigger manually**

```bash
gh workflow run deploy-online-endpoint-pipeline-classical.yml --ref dev
```

- [ ] **Step 2: Verify it's healthy**

```bash
az ml online-endpoint show --name taxi-gha-oep-azmlops-0001dev --resource-group rg-azmlops-0001dev --workspace-name mlw-azmlops-0001dev --query provisioning_state -o tsv
```
Expected: `Succeeded`.

- [ ] **Step 3: Invoke it and confirm a log blob appears** (use Studio's Test tab if the CLI hits the tenant-mismatch issue noted in `mem:deployment_state`)

```bash
az storage blob list --account-name stazmlops0001dev --container-name azureml-blobstore --prefix monitoring/inference-log/online/ --auth-mode login -o table
```
Expected: at least one blob, timestamped after the test invocation.

- [ ] **Step 4: Re-run the batch endpoint pipeline's existing invoke test (from the original deployment plan) and confirm the batch-side log blob appears too**

```bash
az storage blob list --account-name stazmlops0001dev --container-name azureml-blobstore --prefix monitoring/inference-log/batch/ --auth-mode login -o table
```

### Task 13: Manually trigger the monitor-and-retrain workflow once and verify the drift report

- [ ] **Step 1: Trigger**

```bash
gh workflow run monitor-and-retrain-classical.yml --ref dev
```

- [ ] **Step 2: Watch to completion, read the `check-drift` job's log output**

Expected on this first real run (baseline exists from Task 11, some inference data exists from Task 12): a JSON report with `numeric`/`categorical` p-values and a `DRIFT_DETECTED=` line. With so little inference data logged so far, drift is not expected to trigger — that's fine, this task verifies the mechanism runs end-to-end and produces a sane report, not that it correctly detects real drift (there's no real drift yet to detect).

---

## Self-Review Notes

- **Spec coverage:** monitoring (inference logging both paths + drift comparison), automated retraining (scheduled trigger, reusable-workflow call, no new secrets), and champion/challenger (confirmed already existing, not rebuilt) are all addressed. The `online/score.py` empty-file bug and the unseeded train/test split are both fixed as prerequisites the rest of the plan depends on, not scope creep.
- **Placeholder scan:** two steps (Task 3 Step 3, Task 4/5's storage-account passthrough) are deliberately marked "investigate then implement" rather than given a guessed mechanism, because the exact `run-pipeline.yml`/`create-deployment.yml` invocation syntax wasn't read during planning — Task 7 resolves both before they're needed. This is flagged explicitly rather than silently guessed, per the plan's own standard for verified-not-assumed facts.
- **Type/interface consistency:** `check_drift.py`'s baseline JSON schema (`stats.numeric.<col>.values_sample`, `stats.categorical.<col>`) matches exactly what `snapshot_baseline.py` writes. The `DRIFT_DETECTED=` stdout contract is used identically by `monitor-and-retrain`'s `check-drift` job.
- **Cost/scope discipline:** no new Azure resources anywhere in this plan — confirmed by grepping for the absence of any new Terraform files or `az ... create` calls against anything other than blobs (which live in the already-provisioned storage account).
