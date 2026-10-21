"""Dashboard contracts for TrackSight's auditable-operations views."""

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
    def __init__(self):
        self.headers = []
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

    def info(self, value, **_kwargs):
        self.infos.append(value)

    def error(self, *_args, **_kwargs):
        pass

    def write(self, *_args, **_kwargs):
        pass

    def container(self, **_kwargs):
        return _FakeContainer()

    def metric(self, *_args, **_kwargs):
        pass

    def dataframe(self, dataframe, **_kwargs):
        self.dataframes.append(dataframe)

    def altair_chart(self, *_args, **_kwargs):
        pass

    def download_button(self, label, **_kwargs):
        self.downloads.append(label)

    def button(self, *_args, **_kwargs):
        return False

    def selectbox(self, _label, options, **_kwargs):
        return list(options)[0]


class DashboardAuditWorkflowTests(unittest.TestCase):
    def test_audit_workflows_render_with_missing_optional_human_records(self):
        fake_st = _FakeStreamlit()
        artifacts = {
            "run_name": "audit_workflow_smoke",
            "annotated_video": Path("missing.mp4"),
            "trajectory_json": Path("missing_trajectories.json"),
            "run_summary_json": Path("missing_run_summary.json"),
            "review_evidence_json": Path("missing_review.json"),
            "risk_evidence_json": Path("missing_risk.json"),
            "human_review_decisions_json": Path("missing_human_review.json"),
            "race_control_action_log_json": Path("missing_actions.json"),
            "review_corrections_json": Path("missing_corrections.json"),
            "second_opinions_json": Path("missing_opinions.json"),
        }
        review_events = [{
            "track_ids": [7, 11],
            "closest_frame": 98,
            "closest_time_seconds": 4.08,
            "closest_normalized_distance": 0.17,
            "review_clip_path": None,
        }]
        risk_events = [{
            "track_ids": [7, 11],
            "highest_risk_frame": 99,
            "highest_risk_time_seconds": 4.12,
            "predicted_separation_vehicle_widths": 0.15,
            "review_clip_path": None,
        }]
        run_summary = {
            "spatial_analysis_enabled": True,
            "processed_frame_count": 2,
            "frames_per_second": 25.0,
            "track_quality": {
                "7": {"trusted_for_safety_analysis": True},
                "11": {"trusted_for_safety_analysis": True},
            },
        }
        trajectories = {
            "7": [{"frame_index": 0, "confidence": 0.8}],
            "11": [{"frame_index": 0, "confidence": 0.9}],
        }

        with patch.object(dashboard, "st", fake_st):
            dashboard.render_decision_ledger(artifacts, review_events, risk_events)
            dashboard.render_evidence_integrity(artifacts)
            dashboard.render_operational_coverage(trajectories, run_summary)
            dashboard.render_independent_second_review(artifacts, review_events, risk_events)
            dashboard.render_grounded_case_assistant(
                artifacts, review_events, risk_events, run_summary
            )

        self.assertEqual(
            fake_st.headers,
            [
                "Decision ledger",
                "Evidence integrity",
                "Operational coverage",
                "Independent second review",
                "Grounded case assistant",
                "Grounded answer",
            ],
        )
        self.assertIn("Download integrity manifest", fake_st.downloads)
        self.assertGreaterEqual(len(fake_st.dataframes), 2)
        self.assertTrue(any("primary human-review decision" in value for value in fake_st.infos))


if __name__ == "__main__":
    unittest.main()
