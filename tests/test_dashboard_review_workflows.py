"""View-contract checks for TrackSight's decision and correction workflows."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import streamlit_app as dashboard


class _FakeContainer:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _FakeStreamlit:
    """Small deterministic Streamlit surface for reviewer-facing dashboard views."""

    def __init__(self):
        self.headers = []
        self.errors = []
        self.infos = []
        self.dataframes = []
        self.downloads = []
        self.session_state = {}
        self.column_config = SimpleNamespace(
            NumberColumn=lambda *args, **kwargs: (args, kwargs),
        )

    def header(self, value, **_):
        self.headers.append(value)

    def subheader(self, value, **_):
        self.headers.append(value)

    def warning(self, *_args, **_kwargs):
        pass

    def caption(self, *_args, **_kwargs):
        pass

    def container(self, **_kwargs):
        return _FakeContainer()

    def form(self, *_args, **_kwargs):
        return _FakeContainer()

    def metric(self, *_args, **_kwargs):
        pass

    def dataframe(self, dataframe, **_kwargs):
        self.dataframes.append(dataframe)

    def download_button(self, label, **_kwargs):
        self.downloads.append(label)

    def selectbox(self, _label, options, **_kwargs):
        return list(options)[0]

    def text_input(self, *_args, **_kwargs):
        return ""

    def text_area(self, *_args, **_kwargs):
        return ""

    def form_submit_button(self, *_args, **_kwargs):
        return False

    def info(self, value, **_kwargs):
        self.infos.append(value)

    def error(self, value, **_kwargs):
        self.errors.append(value)


class DashboardReviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.artifacts = {
            "run_name": "dashboard_review_workflow",
            "annotated_video": Path("missing_annotated.mp4"),
            "human_review_decisions_json": Path("missing_human_review_decisions.json"),
            "review_corrections_json": Path("missing_reviewer_corrections.json"),
        }
        self.review_events = [{
            "track_ids": [7, 11],
            "closest_frame": 98,
            "closest_time_seconds": 4.0874,
            "closest_normalized_distance": 0.171,
            "review_clip_path": None,
        }]
        self.risk_events = [{
            "track_ids": [7, 11],
            "highest_risk_frame": 99,
            "highest_risk_time_seconds": 4.1291,
            "predicted_separation_vehicle_widths": 0.1513,
            "review_clip_path": None,
        }]
        self.run_summary = {
            "spatial_analysis_enabled": True,
            "track_quality": {
                "7": {"trusted_for_safety_analysis": True},
                "11": {"trusted_for_safety_analysis": True},
            },
            "run_provenance": {"calibration_profile": "demo_calibration.json"},
        }

    def test_command_center_shows_auditable_review_state_when_replay_is_missing(self):
        fake_st = _FakeStreamlit()

        with patch.object(dashboard, "st", fake_st):
            dashboard.render_safety_command_center(
                self.artifacts,
                self.review_events,
                self.risk_events,
                self.run_summary,
                True,
            )

        self.assertEqual(
            fake_st.headers,
            [
                "Safety command center",
                "Why this recommendation",
                "Replay and evidence",
                "Evidence correlation",
            ],
        )
        self.assertEqual(len(fake_st.dataframes), 3)
        self.assertIn("Download after-action report", fake_st.downloads)
        self.assertTrue(any("annotated video is unavailable" in value for value in fake_st.errors))

    def test_reviewer_corrections_stay_separate_until_a_human_submits_feedback(self):
        fake_st = _FakeStreamlit()

        with patch.object(dashboard, "st", fake_st):
            dashboard.render_reviewer_corrections(
                self.artifacts,
                self.review_events,
                self.risk_events,
            )

        self.assertEqual(fake_st.headers, ["Reviewer corrections"])
        self.assertTrue(any("No reviewer correction" in value for value in fake_st.infos))
        self.assertIn("Download correction handoff", fake_st.downloads)


if __name__ == "__main__":
    unittest.main()
