"""Tests for run-aware TrackSight dashboard navigation."""

import tempfile
import unittest
from pathlib import Path

import streamlit_app as dashboard


class DashboardNavigationTests(unittest.TestCase):
    def _artifacts(self, directory):
        directory = Path(directory)
        return {
            "annotated_video": directory / "annotated.mp4",
            "marshal_briefing_json": directory / "briefing.json",
            "marshal_evidence_package": directory / "briefing.zip",
            "human_review_decisions_json": directory / "decisions.json",
            "race_control_action_log_json": directory / "actions.json",
            "review_corrections_json": directory / "corrections.json",
            "second_opinions_json": directory / "opinions.json",
        }

    def test_camera_only_run_hides_event_and_spatial_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = self._artifacts(directory)
            artifacts["annotated_video"].touch()
            views = dashboard.get_available_review_views(
                artifacts,
                review_events=[],
                risk_events=[],
                run_summary={"spatial_analysis_enabled": False},
                safety_zone_evidence=None,
                safety_candidate_evidence=None,
                available_run_count=1,
            )

        self.assertEqual(
            views,
            [
                "Overview",
                "Camera video",
                "Evidence quality",
                "Camera trust monitor",
                "Operational coverage",
                "Evaluation and limits",
                "Provenance",
                "Evidence integrity",
            ],
        )

    def test_event_run_exposes_event_review_without_unavailable_policy_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = self._artifacts(directory)
            artifacts["annotated_video"].touch()
            views = dashboard.get_available_review_views(
                artifacts,
                review_events=[{"track_ids": [3, 4]}],
                risk_events=[],
                run_summary={"spatial_analysis_enabled": True},
                safety_zone_evidence=None,
                safety_candidate_evidence=None,
                available_run_count=2,
            )

        self.assertIn("Evidence queue", views)
        self.assertIn("Live race-control simulation", views)
        self.assertIn("Session comparison", views)
        self.assertNotIn("Safety zones", views)
        self.assertNotIn("Safety candidates", views)
        self.assertNotIn("Marshal briefing", views)


if __name__ == "__main__":
    unittest.main()
