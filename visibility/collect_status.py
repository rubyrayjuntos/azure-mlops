"""Collect read-only status from GitHub Actions and Azure ML CLI.

The collector deliberately shells out only to fixed, read-only commands. It
does not dispatch workflows, create resources, or change Azure state.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import re
import urllib.parse
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from .status import Stage, TrainingRun, VisibilitySnapshot, utc_now
    from .project_state import derive_project_state
except ImportError:  # Supports `python visibility/collect_status.py` from repo root.
    from status import Stage, TrainingRun, VisibilitySnapshot, utc_now
    from project_state import derive_project_state


def run_json(command: list[str]) -> tuple[Any, str | None]:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return json.loads(result.stdout or "[]"), None
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        return [], f"Could not read {' '.join(command[:2])}: {exc}"


def run_text(command: list[str]) -> tuple[str, str | None]:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return result.stdout.strip(), None
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return "", f"Could not read {' '.join(command[:2])}: {exc}"


def monitoring_dependencies_available() -> bool:
    try:
        return all(importlib.util.find_spec(module) for module in ("azure.identity", "azure.storage.blob", "scipy", "pandas"))
    except ModuleNotFoundError:
        return False


def child_stages(parent_name: str, args: argparse.Namespace) -> tuple[list[Stage], str | None]:
    children, warning = run_json(
        [
            "az", "ml", "job", "list", "--parent-job-name", parent_name,
            "--resource-group", args.resource_group, "--workspace-name", args.workspace, "-o", "json",
        ]
    )
    if warning:
        return [], warning
    stages = [
        Stage(name=job.get("display_name") or job.get("name", "Azure ML child job"), status=(job.get("status") or "Unknown").lower())
        for job in children
    ]
    return stages, None


def workflow_jobs(run_id: int) -> tuple[list[dict[str, Any]], str | None]:
    jobs, warning = run_json(["gh", "run", "view", str(run_id), "--json", "jobs"])
    if warning:
        return [], warning
    return jobs.get("jobs", []) if isinstance(jobs, dict) else [], None


def monitoring_status_from_workflows(workflows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for workflow in workflows:
        name = workflow.get("workflowName") or workflow.get("displayTitle") or ""
        if "monitor" not in name.lower() or not workflow.get("databaseId"):
            continue
        log_text, warning = run_text(["gh", "run", "view", str(workflow["databaseId"]), "--log"])
        if warning:
            continue
        matches = re.findall(r"MONITORING_STATUS=(NOT_READY|INSUFFICIENT_DATA|HEALTHY|DRIFT_DETECTED)", log_text)
        if matches:
            return {"status": matches[-1], "source": "GitHub Actions monitor log", "workflow_url": workflow.get("url")}
    return None


def mlflow_run(run_id: str, args: argparse.Namespace, token: str) -> tuple[dict[str, Any], str | None]:
    base = f"https://{args.location}.api.azureml.ms/mlflow/v1.0/subscriptions/{args.subscription}/resourceGroups/{args.resource_group}/providers/Microsoft.MachineLearningServices/workspaces/{args.workspace}"
    url = f"{base}/api/2.0/mlflow/runs/get?{urllib.parse.urlencode({'run_id': run_id})}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response).get("run", {}), None
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"Could not read MLflow run {run_id}: {exc}"


def collect_search_service_metadata(args: argparse.Namespace) -> tuple[dict[str, Any], str | None]:
    services, warning = run_json(["az", "search", "service", "list", "--resource-group", args.resource_group, "-o", "json"])
    if warning:
        return {}, warning
    if not services:
        return {}, None

    service = services[0] if isinstance(services, list) else {}
    service_name = service.get("name") or "azure-search"
    metadata: dict[str, Any] = {
        "service": service_name,
        "status": str(service.get("status") or "unknown"),
        "indexes": [],
        "index_count": 0,
        "document_count": 0,
    }

    keys, warning = run_json(["az", "search", "admin-key", "show", "--resource-group", args.resource_group, "--service-name", service_name, "-o", "json"])
    if warning or not isinstance(keys, dict):
        return metadata, None

    key = keys.get("primaryKey")
    if not key:
        return metadata, None

    url = f"https://{service_name}.search.windows.net/indexes?api-version=2024-07-01"
    request = urllib.request.Request(url, headers={"api-key": key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except (OSError, json.JSONDecodeError):
        return metadata, None

    indexes = payload.get("value", []) if isinstance(payload, dict) else []
    metadata["index_count"] = len(indexes)
    metadata["indexes"] = indexes
    for index in indexes:
        idx_name = index.get("name")
        if not idx_name:
            continue
        stats_url = f"https://{service_name}.search.windows.net/indexes/{urllib.parse.quote(idx_name)}/stats?api-version=2024-07-01"
        stats_request = urllib.request.Request(stats_url, headers={"api-key": key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(stats_request, timeout=15) as stats_response:
                stats_payload = json.load(stats_response)
        except (OSError, json.JSONDecodeError):
            continue
        document_count = int(stats_payload.get("documentCount", 0) if isinstance(stats_payload, dict) else 0)
        metadata["document_count"] += document_count
        metadata["indexes"].append({
            "name": idx_name,
            "document_count": document_count,
            "status": index.get("status"),
        })
    return metadata, None


def collect_rag_evidence(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    evidence: dict[str, Any] = {}
    warnings: list[str] = []

    search_metadata, search_warning = collect_search_service_metadata(args)
    if search_warning:
        warnings.append(search_warning)
    if search_metadata.get("service"):
        evidence["retrieval"] = {
            **evidence.get("retrieval", {}),
            "status": "ready" if str(search_metadata.get("status", "")).lower() in {"running", "online", "succeeded", "success", "ready", "healthy"} else str(search_metadata.get("status", "unknown")),
            "index": os.getenv("AZURE_SEARCH_INDEX", "knowledge-base"),
            "service": search_metadata.get("service"),
            "index_count": search_metadata.get("index_count", 0),
            "document_count": search_metadata.get("document_count", 0),
            "source": "Azure AI Search",
            "live_metadata": search_metadata,
        }
        evidence["grounding"] = {
            **evidence.get("grounding", {}),
            "status": "ready" if str(search_metadata.get("status", "")).lower() in {"running", "online", "succeeded", "success", "ready", "healthy"} else str(search_metadata.get("status", "unknown")),
            "documents": int(search_metadata.get("document_count", evidence.get("grounding", {}).get("documents", 0))),
            "source": "Azure AI Search",
            "service": search_metadata.get("service"),
        }

    health_script = Path("rag/healthcheck.py")
    if health_script.exists():
        result = subprocess.run(["python3", str(health_script)], check=False, capture_output=True, text=True)
        if result.returncode == 0:
            try:
                payload = json.loads(result.stdout or "{}")
            except json.JSONDecodeError as exc:
                warnings.append(f"Could not parse rag health output: {exc}")
            else:
                status = str(payload.get("status", "error"))
                ready = status == "ready"
                evidence["retrieval"] = {
                    "status": "ready" if ready else status,
                    "index": payload.get("index", os.getenv("AZURE_SEARCH_INDEX", "knowledge-base")),
                    "source": "rag/healthcheck.py",
                    "documents": int(payload.get("documents", 0)),
                }
                evidence["grounding"] = {
                    "status": "ready" if ready else status,
                    "documents": int(payload.get("documents", 0)),
                    "source": "rag/healthcheck.py",
                }
                evidence["ai_runtime"] = {
                    "status": "ready" if ready else "not_ready",
                    "agent": "grounding-slice",
                    "source": "rag/healthcheck.py",
                    "model": payload.get("model"),
                    "latency_ms": payload.get("latency_ms"),
                }
        elif result.stderr.strip():
            warnings.append(f"RAG health check failed: {result.stderr.strip()}")

    quality_script = Path("rag/quality.py")
    if quality_script.exists():
        result = subprocess.run(["python3", "-c", "from rag.quality import assess_retrieval_quality; import json; print(json.dumps(assess_retrieval_quality()))"], check=False, capture_output=True, text=True)
        if result.returncode == 0:
            try:
                payload = json.loads(result.stdout or "{}")
            except json.JSONDecodeError as exc:
                warnings.append(f"Could not parse rag quality output: {exc}")
            else:
                evidence.setdefault("retrieval", {})
                evidence["retrieval"].update({
                    "quality": payload,
                    "coverage": payload.get("coverage"),
                    "health": payload.get("health"),
                })
                evidence.setdefault("grounding", {})
                evidence["grounding"].update({
                    "quality": payload,
                    "coverage": payload.get("coverage"),
                    "health": payload.get("health"),
                })

    search_services, warning = run_json(["az", "search", "service", "list", "--resource-group", args.resource_group, "-o", "json"])
    if warning:
        warnings.append(warning)
    elif search_services:
        service = search_services[0]
        service_name = service.get("name") or "azure-search"
        evidence["retrieval"] = {
            **evidence.get("retrieval", {}),
            "status": "ready" if str(service.get("status", "")).lower() in {"running", "online", "succeeded", "success", "ready", "healthy"} else str(service.get("status", "unknown")),
            "index": os.getenv("AZURE_SEARCH_INDEX", "knowledge-base"),
            "service": service_name,
            "source": "Azure AI Search",
        }
        evidence["grounding"] = {
            **evidence.get("grounding", {}),
            "status": evidence.get("grounding", {}).get("status", "ready"),
            "source": "Azure AI Search",
        }
        evidence["ai_runtime"] = {
            **evidence.get("ai_runtime", {}),
            "status": evidence.get("ai_runtime", {}).get("status", "ready"),
            "agent": "grounding-slice",
            "source": "Azure AI Search",
        }

    ai_services, warning = run_json(["az", "cognitiveservices", "account", "list", "--resource-group", args.resource_group, "-o", "json"])
    if warning:
        warnings.append(warning)
    elif ai_services:
        evidence["ai_runtime"] = {
            **evidence.get("ai_runtime", {}),
            "status": "ready",
            "agent": "foundry-grounding-slice",
            "source": "Azure AI Services",
            "model": os.getenv("AZURE_OPENAI_MODEL") or os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("FOUNDRY_MODEL") or os.getenv("MODEL_NAME") or evidence.get("ai_runtime", {}).get("model"),
            "latency_ms": os.getenv("AI_RUNTIME_LATENCY_MS") or evidence.get("ai_runtime", {}).get("latency_ms"),
            "services": ai_services,
        }

    return evidence, warnings


def collect_azure_evidence(args: argparse.Namespace, training_runs: list[TrainingRun]) -> tuple[dict[str, Any], list[str]]:
    evidence: dict[str, Any] = {"models": [], "endpoints": [], "computes": [], "metrics": {}, "monitoring": {"status": "not collected"}}
    warnings: list[str] = []
    models, warning = run_json(["az", "ml", "model", "list", "--name", args.model_name, "--resource-group", args.resource_group, "--workspace-name", args.workspace, "-o", "json"])
    if warning:
        warnings.append(warning)
    else:
        evidence["models"] = models

    online, warning = run_json(["az", "ml", "online-endpoint", "list", "--resource-group", args.resource_group, "--workspace-name", args.workspace, "-o", "json"])
    if warning:
        warnings.append(warning)
    else:
        evidence["endpoints"].extend({**endpoint, "endpoint_type": "online"} for endpoint in online)

    batch, warning = run_json(["az", "ml", "batch-endpoint", "list", "--resource-group", args.resource_group, "--workspace-name", args.workspace, "-o", "json"])
    if warning:
        warnings.append(warning)
    else:
        evidence["endpoints"].extend({**endpoint, "endpoint_type": "batch"} for endpoint in batch)

    computes, warning = run_json(["az", "ml", "compute", "list", "--resource-group", args.resource_group, "--workspace-name", args.workspace, "-o", "json"])
    if warning:
        warnings.append(warning)
    else:
        evidence["computes"] = computes

    token, warning = run_text(["az", "account", "get-access-token", "--resource", "https://ml.azure.com", "--query", "accessToken", "-o", "tsv"])
    if warning:
        warnings.append(warning)
    elif isinstance(token, str):
        models_by_created = sorted(models, key=lambda model: model.get("creation_context", {}).get("created_at", ""), reverse=True)
        if models_by_created:
            model = models_by_created[0]
            try:
                model_json = json.loads(model.get("properties", {}).get("model_json", "{}"))
                register_run_id = model_json.get("run_id")
                register_run, warning = mlflow_run(register_run_id, args, token)
                if warning:
                    warnings.append(warning)
                tags = {tag.get("key"): tag.get("value") for tag in register_run.get("data", {}).get("tags", [])}
                root_run_id = tags.get("mlflow.rootRunId")
                if root_run_id:
                    children, child_warning = run_json(["az", "ml", "job", "list", "--parent-job-name", root_run_id, "--resource-group", args.resource_group, "--workspace-name", args.workspace, "-o", "json"])
                    if child_warning:
                        warnings.append(child_warning)
                    else:
                        evaluate = next((job for job in children if job.get("display_name") == "evaluate_model"), None)
                        if evaluate:
                            evaluate_run, metric_warning = mlflow_run(evaluate["name"], args, token)
                            if metric_warning:
                                warnings.append(metric_warning)
                            evidence["metrics"] = {m["key"]: m["value"] for m in evaluate_run.get("data", {}).get("metrics", [])}
                evidence["model_registry"] = {"registered_versions": [m.get("version") for m in models_by_created], "latest_registered_version": model.get("version"), "champion_version": next((m.get("version") for m in models if (m.get("tags") or {}).get("lifecycle") == "champion"), None)}
            except (TypeError, json.JSONDecodeError) as exc:
                warnings.append(f"Could not interpret model registry lineage: {exc}")
    rag_evidence, rag_warnings = collect_rag_evidence(args)
    warnings.extend(rag_warnings)
    evidence.update(rag_evidence)

    if args.storage_account and monitoring_dependencies_available():
        result = subprocess.run(
            ["python3", "data-science/src/check_drift.py", "--storage_account", args.storage_account],
            check=False, capture_output=True, text=True,
        )
        status_line = next((line for line in result.stdout.splitlines() if line.startswith("MONITORING_STATUS=")), None)
        if status_line:
            evidence["monitoring"] = {"status": status_line.split("=", 1)[1], "detail": result.stdout.strip().splitlines()[-2:]}
        elif result.returncode != 0:
            warnings.append(f"Monitoring check did not return a status: {result.stderr.strip() or 'unknown error'}")
    return evidence, warnings


def collect(args: argparse.Namespace) -> VisibilitySnapshot:
    warnings: list[str] = []
    workflow_runs, warning = run_json(
        [
            "gh", "run", "list", "--limit", str(args.limit), "--json",
            "databaseId,displayTitle,status,conclusion,workflowName,createdAt,updatedAt,url",
        ]
    )
    if warning:
        warnings.append(warning)

    for workflow in workflow_runs:
        run_id = workflow.get("databaseId")
        if not run_id:
            continue
        jobs_for_run, job_warning = workflow_jobs(run_id)
        workflow["jobs"] = jobs_for_run
        if job_warning:
            warnings.append(job_warning)

    jobs, warning = run_json(
        [
            "az", "ml", "job", "list", "--resource-group", args.resource_group,
            "--workspace-name", args.workspace, "--max-results", str(args.limit), "-o", "json",
        ]
    )
    if warning:
        warnings.append(warning)

    training_runs = []
    for job in jobs:
        if args.experiment and job.get("experiment_name") != args.experiment:
            continue
        status = (job.get("status") or job.get("properties", {}).get("status") or "Unknown").lower()
        stages, child_warning = child_stages(str(job.get("name", "unknown")), args)
        if child_warning:
            warnings.append(child_warning)
        if not stages:
            stages = [Stage(name="Azure ML job", status=status)]
        training_runs.append(
            TrainingRun(
                run_id=str(job.get("name", "unknown")),
                display_name=job.get("display_name") or job.get("displayName") or "Azure ML job",
                status=status,
                experiment=job.get("experiment_name", ""),
                model_name=args.model_name,
                url=job.get("studio_url") or job.get("services", {}).get("Studio", {}).get("endpoint"),
                started_at=job.get("creation_context", {}).get("created_at"),
                updated_at=job.get("creation_context", {}).get("last_modified_at"),
                stages=stages,
            )
        )

    evidence, evidence_warnings = collect_azure_evidence(args, training_runs)
    warnings.extend(evidence_warnings)
    workflow_monitoring = monitoring_status_from_workflows(workflow_runs)
    if workflow_monitoring:
        evidence["monitoring"] = workflow_monitoring

    return VisibilitySnapshot(
        project=args.project,
        environment=args.environment,
        generated_at=utc_now(),
        workflow_runs=workflow_runs,
        training_runs=training_runs,
        warnings=warnings,
        project_state=derive_project_state({"workflow_runs": workflow_runs, "training_runs": [asdict(run) for run in training_runs], "evidence": evidence}),
        evidence=evidence,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect read-only MLOps visibility state")
    parser.add_argument("--project", default="azure-mlops")
    parser.add_argument("--environment", default="dev")
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--subscription", default="5b452321-32fd-4b1c-8bbf-6d69a5a587ad")
    parser.add_argument("--location", default="eastus")
    parser.add_argument("--model-name", default="taxi-model")
    parser.add_argument("--experiment", default="taxi-fare-training")
    parser.add_argument("--storage-account", default=None, help="Optional storage account for the read-only drift check")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("visibility/status.json"))
    args = parser.parse_args()
    snapshot = collect(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot.to_dict(), indent=2) + "\n")
    print(f"Wrote read-only visibility snapshot to {args.output}")


if __name__ == "__main__":
    main()
