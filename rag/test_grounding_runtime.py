import os
import unittest
from unittest.mock import patch

from rag.grounding import grounded_answer
from rag.index_documents import build_document
from rag.query_documents import search


class GroundingRuntimeTests(unittest.TestCase):
    def test_grounded_answer_returns_citations_for_retrieved_context(self):
        result = grounded_answer(
            "How does the platform combine Azure ML and Foundry grounding?",
            live_results=[
                {
                    "title": "Platform architecture",
                    "source": "internal-platform",
                    "content": "This platform combines Azure ML for training and model lifecycle governance with Azure AI Search and Foundry for retrieval and grounding.",
                }
            ],
        )

        self.assertEqual(result["status"], "ready")
        self.assertIn("Azure ML", result["answer"])
        self.assertIn("Foundry", result["answer"])
        self.assertEqual(result["citations"][0]["source"], "internal-platform")

    def test_grounded_answer_returns_empty_status_when_no_context_is_found(self):
        result = grounded_answer("What is the answer to an unknown question?", live_results=[])

        self.assertEqual(result["status"], "not_found")
        self.assertIn("No grounded context", result["answer"])
        self.assertEqual(result["citations"], [])

    def test_search_raises_when_azure_search_is_not_configured(self):
        with patch.dict(os.environ, {"AZURE_SEARCH_ENDPOINT": "", "AZURE_SEARCH_KEY": ""}, clear=False):
            with self.assertRaises(RuntimeError) as context:
                search("What does the platform combine?", top_k=2)
        self.assertIn("AZURE_SEARCH_ENDPOINT", str(context.exception))

    def test_build_document_uses_safe_key_for_azure_search(self):
        doc = build_document("Platform architecture", "internal-platform", "content", [0.01] * 1536)
        self.assertNotIn(":", doc["id"])
        self.assertTrue(doc["id"].startswith("internal-platform-"))


if __name__ == "__main__":
    unittest.main()
