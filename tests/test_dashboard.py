import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from streamlit_app import (
    build_trajectory_dataframe,
    format_table_values,
    get_artifact_display_name,
    get_default_evidence_event_index,
    is_reviewable_run_summary,
)


class DashboardRunCompatibilityTests(unittest.TestCase):
    def test_accepts_a_complete_older_run_schema(self):
        summary = {
            "all_observations": 1,
            "all_track_segments": 1,
            "frames_per_second": 25.0,
            "observations_per_frame": 1.0,
            "observed_close_approach_event_count": 0,
            "processed_frame_count": 1,
            "risk_event_count": 0,
            "spatial_analysis_enabled": False,
            "spatial_analysis_status": "disabled_without_camera_calibration",
            "track_quality": {},
            "trusted_track_segments": 0,
            "pipeline_version": "an-older-label",
        }

        self.assertTrue(is_reviewable_run_summary(summary))

    def test_rejects_a_summary_without_track_quality_controls(self):
        summary = {
            "all_observations": 1,
            "all_track_segments": 1,
        }

        self.assertFalse(is_reviewable_run_summary(summary))


    def test_formats_mixed_values_as_display_text(self):
        formatted = format_table_values(
            {
                "Count": 5,
                "Rate": 6.5,
                "Missing": None,
                "Name": "clean TCR demo",
            }
        )

        self.assertEqual(
            formatted,
            {
                "Count": "5",
                "Rate": "6.5",
                "Missing": "none",
                "Name": "clean TCR demo",
            },
        )

    def test_formats_provenance_artifacts_as_compact_names(self):
        self.assertEqual(
            get_artifact_display_name(r"C:\\example\\models\\vehicle.pt"),
            "vehicle.pt",
        )
        self.assertEqual(get_artifact_display_name(None), "none")

    def test_prefers_an_event_with_a_saved_clip(self):
        with TemporaryDirectory() as temporary_directory:
            clip_path = Path(temporary_directory) / "review.mp4"
            clip_path.touch()
            events = pd.DataFrame(
                [{"clip": None}, {"clip": str(clip_path)}],
                index=[10, 11],
            )

            self.assertEqual(get_default_evidence_event_index(events), 11)

    def test_splits_a_path_after_a_large_motion_gap(self):
        trajectories = {
            "1": [
                {
                    "frame_index": 0,
                    "time_seconds": 0.0,
                    "confidence": 0.9,
                    "motion": {},
                    "track_position": (10.0, 20.0),
                },
                {
                    "frame_index": 1,
                    "time_seconds": 0.04,
                    "confidence": 0.9,
                    "motion": {},
                    "track_position": (12.0, 20.0),
                },
                {
                    "frame_index": 3,
                    "time_seconds": 0.12,
                    "confidence": 0.9,
                    "motion": {"status": "gap_too_large"},
                    "track_position": (18.0, 20.0),
                },
            ]
        }

        dataframe = build_trajectory_dataframe(trajectories)

        self.assertEqual(dataframe["frame"].tolist(), [0, 1, 3])
        self.assertEqual(dataframe["trajectory_segment"].tolist(), ["1-0", "1-0", "1-1"])

if __name__ == "__main__":
    unittest.main()
