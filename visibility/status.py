"""Normalized, source-neutral status contract for the visibility dashboard."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Stage:
    name: str
    status: str
    detail: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    url: str | None = None


@dataclass
class TrainingRun:
    run_id: str
    display_name: str
    status: str
    experiment: str = ""
    model_name: str = ""
    model_version: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    stages: list[Stage] = field(default_factory=list)
    url: str | None = None
    started_at: str | None = None
    updated_at: str | None = None


@dataclass
class VisibilitySnapshot:
    project: str
    environment: str
    generated_at: str
    workflow_runs: list[dict[str, Any]] = field(default_factory=list)
    training_runs: list[TrainingRun] = field(default_factory=list)
    endpoints: list[dict[str, Any]] = field(default_factory=list)
    monitoring: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    project_state: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def snapshot_from_dict(payload: dict[str, Any]) -> VisibilitySnapshot:
    """Parse the public contract while tolerating missing optional sections."""
    training_runs = []
    for raw_run in payload.get("training_runs", []):
        raw_run = dict(raw_run)
        stages = [Stage(**stage) for stage in raw_run.pop("stages", [])]
        training_runs.append(TrainingRun(stages=stages, **raw_run))
    return VisibilitySnapshot(
        project=payload.get("project", "unknown"),
        environment=payload.get("environment", "unknown"),
        generated_at=payload.get("generated_at", utc_now()),
        workflow_runs=payload.get("workflow_runs", []),
        training_runs=training_runs,
        endpoints=payload.get("endpoints", []),
        monitoring=payload.get("monitoring", {}),
        warnings=payload.get("warnings", []),
        project_state=payload.get("project_state", {}),
        evidence=payload.get("evidence", {}),
    )
