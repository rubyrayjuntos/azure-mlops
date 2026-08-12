import unittest

from status import snapshot_from_dict
from project_state import derive_project_state


class StatusContractTests(unittest.TestCase):
    def test_optional_sections_default_without_losing_training_stages(self):
        snapshot = snapshot_from_dict(
            {
                "project": "demo",
                "environment": "dev",
                "training_runs": [
                    {"run_id": "job-1", "display_name": "train", "status": "running", "stages": [{"name": "train", "status": "running"}]}
                ],
            }
        )
        self.assertEqual(snapshot.project, "demo")
        self.assertEqual(snapshot.training_runs[0].stages[0].name, "train")
        self.assertEqual(snapshot.workflow_runs, [])

    def test_project_state_surfaces_active_deployment_and_pipeline_nodes(self):
        state = derive_project_state(
            {
                "workflow_runs": [
                    {
                        "workflowName": "deploy-online-endpoint-pipeline",
                        "status": "in_progress",
                        "createdAt": "2026-08-10T10:00:00Z",
                    }
                ],
                "training_runs": [
                    {
                        "run_id": "job-1",
                        "display_name": "training",
                        "status": "completed",
                        "started_at": "2026-08-10T09:00:00Z",
                        "stages": [
                            {"name": "prep_data", "status": "completed"},
                            {"name": "evaluate_model", "status": "completed"},
                            {"name": "register_model", "status": "completed"},
                        ],
                    }
                ],
                "evidence": {"endpoints": [{"endpoint_type": "online", "name": "online", "provisioning_state": "Creating"}]},
            }
        )
        self.assertEqual(state["project_state"], "online_deployment_in_progress")
        self.assertEqual(state["nodes"]["online_deployment"]["state"], "running")
        self.assertEqual(state["nodes"]["champion_model"]["state"], "live")
        self.assertEqual(state["governance"]["deployments"]["active"], 1)
        self.assertGreater(len(state["timeline"]), 0)

    def test_project_state_exposes_retrieval_and_grounding_control_plane(self):
        state = derive_project_state(
            {
                "workflow_runs": [],
                "training_runs": [],
                "evidence": {
                    "retrieval": {"status": "ready", "index": "knowledge-base"},
                    "grounding": {"status": "healthy", "documents": 142},
                    "ai_runtime": {"status": "online", "agent": "support-agent"},
                },
            }
        )
        self.assertEqual(state["nodes"]["retrieval"]["state"], "live")
        self.assertEqual(state["nodes"]["grounding"]["state"], "live")
        self.assertEqual(state["nodes"]["ai_runtime"]["state"], "live")
        self.assertIn("retrieval", state["control_plane"]["views"])
        self.assertIn("ai_runtime", state["control_plane"]["views"])

    def test_project_state_includes_retrieval_quality_health_in_node_detail(self):
        state = derive_project_state(
            {
                "workflow_runs": [],
                "training_runs": [],
                "evidence": {
                    "retrieval": {"status": "ready", "index": "knowledge-base", "quality": {"coverage": "strong", "health": "ready"}},
                    "grounding": {"status": "degraded", "documents": 23, "quality": {"coverage": "adequate", "health": "review"}},
                    "ai_runtime": {"status": "online", "agent": "support-agent"},
                },
            }
        )
        self.assertIn("strong", state["nodes"]["retrieval"]["detail"])
        self.assertIn("review", state["nodes"]["grounding"]["detail"])

    def test_project_state_surfaces_foundry_runtime_metadata(self):
        state = derive_project_state(
            {
                "workflow_runs": [],
                "training_runs": [],
                "evidence": {
                    "retrieval": {"status": "ready", "index": "knowledge-base", "quality": {"coverage": "strong", "health": "ready"}},
                    "grounding": {"status": "ready", "documents": 142, "quality": {"coverage": "strong", "health": "ready"}},
                    "ai_runtime": {"status": "ready", "agent": "foundry-grounding-slice", "model": "gpt-4o-mini", "latency_ms": 420},
                },
            }
        )
        self.assertIn("gpt-4o-mini", state["nodes"]["ai_runtime"]["detail"])
        self.assertIn("420", state["nodes"]["ai_runtime"]["detail"])

    def test_project_state_emits_operator_summary_for_runtime_control_plane(self):
        state = derive_project_state(
            {
                "workflow_runs": [],
                "training_runs": [],
                "evidence": {
                    "retrieval": {"status": "ready", "index": "knowledge-base", "quality": {"coverage": "strong", "health": "ready"}},
                    "grounding": {"status": "degraded", "documents": 88, "quality": {"coverage": "adequate", "health": "review"}},
                    "ai_runtime": {"status": "ready", "agent": "foundry-grounding-slice", "model": "gpt-4o-mini", "latency_ms": 420},
                },
            }
        )
        self.assertIn("retrieval", state["control_plane"]["summary"].lower())
        self.assertIn("grounding", state["control_plane"]["summary"].lower())
        self.assertIn("ai runtime", state["control_plane"]["summary"].lower())


if __name__ == "__main__":
    unittest.main()
