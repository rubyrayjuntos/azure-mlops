"""Translate provider evidence into MLOps lifecycle meaning for the dashboard."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _parse_time(value: str | None) -> datetime:
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return datetime.min


def _state_from_status(status: str | None) -> str:
    value = (status or "").lower()
    if value in {"in_progress", "running", "queued", "preparing"}:
        return "running"
    if value in {"success", "completed"}:
        return "live"
    if value in {"failure", "failed", "startup_failure", "cancelled"}:
        return "failed"
    return "unknown"


def _workflow_category(workflow: str) -> str:
    value = workflow.lower()
    if "monitor" in value:
        return "monitoring"
    if "train" in value or "retrain" in value:
        return "training"
    if "online" in value:
        return "online_deployment"
    if "batch" in value:
        return "batch_deployment"
    if "infra" in value or "terraform" in value:
        return "infrastructure"
    return "workflow"


def derive_project_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Build a stable semantic view from GitHub and Azure evidence.

    This is intentionally a projection, not a new source of truth. Provider
    records remain available in the technical drill-down and can be revisited
    as the state rules mature.
    """
    nodes = {
        "data": {"label": "Data preparation", "state": "idle", "detail": "No active data run"},
        "training": {"label": "Training", "state": "idle", "detail": "No active training run"},
        "evaluation": {"label": "Evaluation", "state": "idle", "detail": "Awaiting a training run"},
        "champion_model": {"label": "Champion model", "state": "unknown", "detail": "Model registry evidence not collected"},
        "batch_deployment": {"label": "Batch deployment", "state": "unknown", "detail": "No recent batch deployment evidence"},
        "online_deployment": {"label": "Online deployment", "state": "unknown", "detail": "No recent online deployment evidence"},
        "monitoring": {"label": "Monitoring", "state": "unknown", "detail": "No recent monitoring evidence"},
        "retraining": {"label": "Retraining", "state": "idle", "detail": "No retraining currently running"},
        "retrieval": {"label": "Retrieval", "state": "unknown", "detail": "No retrieval evidence collected"},
        "grounding": {"label": "Grounding", "state": "unknown", "detail": "No grounding evidence collected"},
        "ai_runtime": {"label": "AI runtime", "state": "unknown", "detail": "No Foundry or agent evidence collected"},
    }
    evidence = snapshot.get("evidence", {})
    timeline: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []

    training_runs = sorted(snapshot.get("training_runs", []), key=lambda r: _parse_time(r.get("started_at")), reverse=True)
    latest_training = training_runs[0] if training_runs else None
    if latest_training:
        run_status = _state_from_status(latest_training.get("status"))
        nodes["training"].update(state=run_status, detail=f"{latest_training.get('display_name', 'Training run')} · {latest_training.get('status', 'unknown')}")
        stages = latest_training.get("stages", [])
        by_name = {stage.get("name"): stage for stage in stages}
        if by_name.get("prep_data"):
            nodes["data"].update(state=_state_from_status(by_name["prep_data"].get("status")), detail=by_name["prep_data"].get("status", "unknown"))
        if by_name.get("evaluate_model"):
            nodes["evaluation"].update(state=_state_from_status(by_name["evaluate_model"].get("status")), detail=by_name["evaluate_model"].get("status", "unknown"))
        if by_name.get("register_model") and _state_from_status(by_name["register_model"].get("status")) == "live":
            nodes["champion_model"].update(state="live", detail="Model registration completed")
        for stage in stages:
            timeline.append({"at": latest_training.get("updated_at") or latest_training.get("started_at"), "title": stage.get("name", "Azure ML stage"), "detail": stage.get("status", "unknown"), "state": _state_from_status(stage.get("status")), "source": "Azure ML"})

    seen_categories: set[str] = set()
    for workflow in sorted(snapshot.get("workflow_runs", []), key=lambda w: _parse_time(w.get("updatedAt") or w.get("createdAt")), reverse=True):
        category = _workflow_category(workflow.get("workflowName") or workflow.get("displayTitle") or "")
        state = _state_from_status(workflow.get("status") if workflow.get("status") != "completed" else workflow.get("conclusion"))
        if category in nodes and state != "unknown" and category not in seen_categories:
            nodes[category].update(state=state, detail=f"{workflow.get('workflowName', 'Workflow')} · {workflow.get('conclusion') or workflow.get('status')}")
            seen_categories.add(category)
        if state == "running":
            active.append({"type": category, "title": workflow.get("workflowName") or workflow.get("displayTitle"), "detail": "Workflow is in progress", "url": workflow.get("url")})
        timeline.append({"at": workflow.get("updatedAt") or workflow.get("createdAt"), "title": workflow.get("workflowName") or workflow.get("displayTitle"), "detail": workflow.get("conclusion") or workflow.get("status"), "state": state, "source": "GitHub Actions", "url": workflow.get("url")})

    registry = evidence.get("model_registry", {})
    if registry.get("champion_version"):
        nodes["champion_model"].update(state="live", detail=f"Champion version {registry['champion_version']}")
    elif registry.get("registered_versions"):
        nodes["champion_model"].update(state="unknown", detail="Registered models found; champion tag not set")

    retrieval_health = evidence.get("retrieval")
    if retrieval_health and retrieval_health.get("status") not in {None, "not collected"}:
        quality = retrieval_health.get("quality") or {}
        coverage = quality.get("coverage") or retrieval_health.get("coverage")
        health = quality.get("health") or retrieval_health.get("health")
        detail = f"Index {retrieval_health.get('index', 'unknown')} · {retrieval_health.get('status')}"
        if coverage and health:
            detail = f"{detail} · coverage={coverage} health={health}"
        nodes["retrieval"].update(state="live" if str(retrieval_health.get("status", "")).lower() in {"ready", "healthy", "online", "active"} else "failed", detail=detail)

    grounding_health = evidence.get("grounding")
    if grounding_health and grounding_health.get("status") not in {None, "not collected"}:
        quality = grounding_health.get("quality") or {}
        coverage = quality.get("coverage") or grounding_health.get("coverage")
        health = quality.get("health") or grounding_health.get("health")
        detail = f"{grounding_health.get('documents', 0)} docs · {grounding_health.get('status')}"
        if coverage and health:
            detail = f"{detail} · coverage={coverage} health={health}"
        nodes["grounding"].update(state="live" if str(grounding_health.get("status", "")).lower() in {"ready", "healthy", "online", "active", "degraded"} else "failed", detail=detail)

    ai_runtime_health = evidence.get("ai_runtime")
    if ai_runtime_health and ai_runtime_health.get("status") not in {None, "not collected"}:
        model = ai_runtime_health.get("model") or ai_runtime_health.get("deployment")
        latency = ai_runtime_health.get("latency_ms")
        detail = f"{ai_runtime_health.get('agent', 'agent')} · {ai_runtime_health.get('status')}"
        if model:
            detail = f"{detail} · model={model}"
        if latency is not None:
            detail = f"{detail} · latency_ms={latency}"
        nodes["ai_runtime"].update(state="live" if str(ai_runtime_health.get("status", "")).lower() in {"ready", "healthy", "online", "active"} else "failed", detail=detail)

    endpoints = evidence.get("endpoints", [])
    for endpoint in endpoints:
        endpoint_type = endpoint.get("endpoint_type")
        key = "online_deployment" if endpoint_type == "online" else "batch_deployment"
        provisioning = str(endpoint.get("provisioning_state", "")).lower()
        nodes[key].update(state="live" if "succeed" in provisioning else ("failed" if "fail" in provisioning else "running"), detail=f"{endpoint.get('name', 'Endpoint')} · {endpoint.get('provisioning_state', 'unknown')}")
    if evidence.get("monitoring", {}).get("status") not in {None, "not collected"}:
        monitoring_status = evidence["monitoring"]["status"]
        nodes["monitoring"].update(state="failed" if monitoring_status == "DRIFT_DETECTED" else "live", detail=monitoring_status)
    if evidence.get("metrics") and nodes["evaluation"]["state"] in {"idle", "unknown"}:
        nodes["evaluation"].update(state="live", detail="Evaluation metrics collected from MLflow")

    retrieval = evidence.get("retrieval", {})
    if retrieval.get("status") not in {None, "not collected"}:
        quality = retrieval.get("quality") or {}
        coverage = quality.get("coverage") or retrieval.get("coverage")
        health = quality.get("health") or retrieval.get("health")
        detail = f"Index {retrieval.get('index', 'unknown')} · {retrieval.get('status')}"
        if coverage and health:
            detail = f"{detail} · coverage={coverage} health={health}"
        nodes["retrieval"].update(state="live" if str(retrieval.get("status", "")).lower() in {"ready", "healthy", "online", "active"} else "failed", detail=detail)
    grounding = evidence.get("grounding", {})
    if grounding.get("status") not in {None, "not collected"}:
        quality = grounding.get("quality") or {}
        coverage = quality.get("coverage") or grounding.get("coverage")
        health = quality.get("health") or grounding.get("health")
        detail = f"{grounding.get('documents', 0)} docs · {grounding.get('status')}"
        if coverage and health:
            detail = f"{detail} · coverage={coverage} health={health}"
        nodes["grounding"].update(state="live" if str(grounding.get("status", "")).lower() in {"ready", "healthy", "online", "active", "degraded"} else "failed", detail=detail)
    runtime = evidence.get("ai_runtime", {})
    if runtime.get("status") not in {None, "not collected"}:
        model = runtime.get("model") or runtime.get("deployment")
        latency = runtime.get("latency_ms")
        detail = f"{runtime.get('agent', 'agent')} · {runtime.get('status')}"
        if model:
            detail = f"{detail} · model={model}"
        if latency is not None:
            detail = f"{detail} · latency_ms={latency}"
        nodes["ai_runtime"].update(state="live" if str(runtime.get("status", "")).lower() in {"ready", "healthy", "online", "active"} else "failed", detail=detail)

    if active:
        current = active[0]
        project_state = current["type"] + "_in_progress"
        active_activity = current
    elif any(node["state"] == "failed" for node in nodes.values()):
        failed_key = next(key for key, node in nodes.items() if node["state"] == "failed")
        project_state = failed_key + "_failed"
        active_activity = {"type": failed_key, "title": nodes[failed_key]["label"] + " needs attention", "detail": nodes[failed_key]["detail"]}
    elif latest_training and _state_from_status(latest_training.get("status")) == "failed":
        project_state = "training_failed"
        active_activity = {"type": "training", "title": "Latest training run failed", "detail": latest_training.get("run_id")}
    elif latest_training:
        project_state = "ready_or_monitoring"
        active_activity = {"type": "idle", "title": "No active lifecycle operation", "detail": "Latest training run is complete"}
    else:
        project_state = "awaiting_activity"
        active_activity = {"type": "idle", "title": "No lifecycle activity found", "detail": "Refresh the visibility snapshot"}

    timeline.sort(key=lambda event: _parse_time(event.get("at")), reverse=True)
    failed_workflows = [w for w in snapshot.get("workflow_runs", []) if _state_from_status(w.get("status") if w.get("status") != "completed" else w.get("conclusion")) == "failed"]
    deployment_workflows = [
        w for w in snapshot.get("workflow_runs", [])
        if _workflow_category(w.get("workflowName") or w.get("displayTitle") or "") in {"batch_deployment", "online_deployment"}
    ]
    latest_deployment = sorted(deployment_workflows, key=lambda w: _parse_time(w.get("updatedAt") or w.get("createdAt")), reverse=True)
    governance = {
        "performance": {
            "state": nodes["evaluation"]["state"],
            "model": registry.get("champion_version") or "Champion state not tagged",
            "latest_registered": registry.get("latest_registered_version", "Not collected"),
            "metrics": evidence.get("metrics", {}) or (latest_training.get("metrics", {}) if latest_training else {}),
            "detail": nodes["evaluation"]["detail"],
        },
        "data_quality": {
            "state": nodes["data"]["state"],
            "detail": nodes["data"]["detail"],
            "source": "Azure ML prep_data stage",
        },
        "deployments": {
            "active": sum(1 for endpoint in endpoints if "succeed" not in str(endpoint.get("provisioning_state", "")).lower() and "fail" not in str(endpoint.get("provisioning_state", "")).lower()),
            "successful": sum(1 for endpoint in endpoints if "succeed" in str(endpoint.get("provisioning_state", "")).lower()),
            "failed": sum(1 for endpoint in endpoints if "fail" in str(endpoint.get("provisioning_state", "")).lower()),
            "latest": (latest_deployment[0].get("workflowName") if latest_deployment else (endpoints[0].get("name") if endpoints else "No deployment evidence")),
        },
        "alerts": [
            {"severity": "critical", "title": w.get("workflowName") or w.get("displayTitle"), "detail": w.get("conclusion") or w.get("status"), "url": w.get("url")}
            for w in failed_workflows[:8]
        ],
        "azure": {
            "training_jobs": len(training_runs),
            "latest_training": latest_training.get("status") if latest_training else "not collected",
            "endpoints": len(endpoints),
            "monitoring": evidence.get("monitoring", {}).get("status", "not collected"),
        },
    }
    summary_parts = []
    retrieval_state = nodes["retrieval"]["state"]
    grounding_state = nodes["grounding"]["state"]
    runtime_state = nodes["ai_runtime"]["state"]
    if retrieval_state == "live":
        summary_parts.append("retrieval ready")
    elif retrieval_state == "failed":
        summary_parts.append("retrieval degraded")
    else:
        summary_parts.append("retrieval unknown")

    if grounding_state == "live":
        summary_parts.append("grounding healthy")
    elif grounding_state == "failed":
        summary_parts.append("grounding degraded")
    else:
        summary_parts.append("grounding unknown")

    if runtime_state == "live":
        summary_parts.append("AI runtime online")
    elif runtime_state == "failed":
        summary_parts.append("AI runtime offline")
    else:
        summary_parts.append("AI runtime unknown")

    control_plane = {
        "views": [
            "project",
            "training",
            "deployments",
            "monitoring",
            "retrieval",
            "grounding",
            "ai_runtime",
        ],
        "topology": {
            "mlops": ["training", "evaluation", "champion_model", "batch_deployment", "online_deployment", "monitoring"],
            "llmops": ["retrieval", "grounding", "ai_runtime"],
        },
        "summary": "; ".join(summary_parts),
    }

    return {
        "project_state": project_state,
        "active_activity": active_activity,
        "nodes": nodes,
        "timeline": timeline[:40],
        "governance": governance,
        "control_plane": control_plane,
    }
