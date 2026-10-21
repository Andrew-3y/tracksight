"""Deterministic view-level checks for TrackSight's optional dashboard extensions."""

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
    """Minimal Streamlit surface used to verify extension-view output contracts."""

    def __init__(self):
        self.headers = []
        self.infos = []
        self.dataframes = []
        self.downloads = []
        self.column_config = SimpleNamespace(
            NumberColumn=lambda *args, **kwargs: (args, kwargs),
            JsonColumn=lambda *args, **kwargs: (args, kwargs),
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

    def metric(self, *_args, **_kwargs):
        pass

    def dataframe(self, dataframe, **_kwargs):
        self.dataframes.append(dataframe)

    def download_button(self, label, **_kwargs):
        self.downloads.append(label)

    def file_uploader(self, *_args, **_kwargs):
        return None

    def info(self, value, **_kwargs):
        self.infos.append(value)


class DashboardExtensionViewTests(unittest.TestCase):
    def test_extension_views_render_safe_data_and_empty_states(self):
        fake_st = _FakeStreamlit()
        artifacts = {
            "run_name": "dashboard_extension_smoke",
            "annotated_video": Path("missing.mp4"),
            "human_review_decisions_json": Path("missing_human_review_decisions.json"),
        }
        review_events = [{
            "track_ids": [7, 11],
            "closest_frame": 98,
            "closest_time_seconds": 4.0874,
            "closest_normalized_distance": 0.171,
            "review_clip_path": None,
        }]
        risk_events = [{
            "track_ids": [3, 4],
            "highest_risk_frame": 26,
            "highest_risk_time_seconds": 1.0844,
            "predicted_separation_vehicle_widths": 0.1172,
            "review_clip_path": None,
        }]
        run_summary = {
            "run_provenance": {
                "source_video": "C:/private/demo.mp4",
                "camera_mode": "fixed",
            }
        }

        with patch.object(dashboard, "st", fake_st):
            dashboard.render_learning_queue(
                artifacts, review_events, risk_events, None
            )
            dashboard.render_telemetry_correlation(
                artifacts, review_events, risk_events
            )
            dashboard.render_multi_camera_review(
                artifacts, review_events, risk_events
            )
            dashboard.render_race_control_export(
                artifacts, review_events, risk_events, run_summary
            )

        self.assertEqual(
            fake_st.headers,
            [
                "Learning queue",
                "Telemetry correlation",
                "Multi-camera review",
                "Race-control export",
                "Timeline preview",
            ],
        )
        self.assertEqual(len(fake_st.dataframes), 2)
        self.assertTrue(any("No telemetry file" in value for value in fake_st.infos))
        self.assertTrue(any("No camera manifest" in value for value in fake_st.infos))
        self.assertIn("Download labelling manifest", fake_st.downloads)
        self.assertIn("Download race-control timeline", fake_st.downloads)


if __name__ == "__main__":
    unittest.main()
