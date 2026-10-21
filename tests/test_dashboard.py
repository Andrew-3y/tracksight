import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from streamlit_app import (
    build_race_control_timeline_dataframe,
    build_trajectory_dataframe,
    build_human_review_queue,
    filter_race_control_timeline,
    format_table_values,
    get_artifact_display_name,
    get_run_artifacts,
    get_default_evidence_event_index,
    get_source_evidence_event,
    get_human_review_event_summary,
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

    def test_finds_the_optional_marshal_briefing_artifact(self):
        trajectory_path = Path("outputs/tracking/demo_trajectories.json")

        artifacts = get_run_artifacts(trajectory_path)

        self.assertEqual(
            artifacts["marshal_briefing_json"],
            Path("outputs/tracking/demo_marshal_briefing.json"),
        )
        self.assertEqual(
            artifacts["safety_zone_evidence_json"],
            Path("outputs/tracking/demo_safety_zone_evidence.json"),
        )
        self.assertEqual(
            artifacts["safety_candidate_evidence_json"],
            Path("outputs/tracking/demo_safety_candidates.json"),
        )

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

    def test_timeline_keeps_evidence_types_separate_and_filters_by_vehicle(self):
        timeline = build_race_control_timeline_dataframe(
            review_events=[
                {
                    "track_ids": [2, 3],
                    "closest_frame": 10,
                    "closest_time_seconds": 0.4,
                    "closest_normalized_distance": 0.5,
                    "review_clip_path": "observed.mp4",
                }
            ],
            risk_events=[
                {
                    "track_ids": [1, 2],
                    "highest_risk_frame": 20,
                    "highest_risk_time_seconds": 0.8,
                    "predicted_separation_vehicle_widths": 0.3,
                    "review_clip_path": "risk.mp4",
                }
            ],
        )

        filtered = filter_race_control_timeline(
            timeline,
            selected_evidence_types=["Observed proximity"],
            selected_track_ids=[3],
            selected_time_range=(0.0, 1.0),
        )

        self.assertEqual(timeline["evidence_type"].tolist(), ["Predicted risk", "Observed proximity"])
        self.assertEqual(filtered["event_key"].tolist(), ["observed-0"])
        self.assertEqual(
            filtered.iloc[0]["measurement_basis"],
            "average detected-box diagonals",
        )

    def test_returns_the_persisted_record_for_a_timeline_event_key(self):
        predicted_event = {"track_ids": [1, 2], "highest_risk_frame": 20}
        observed_event = {"track_ids": [2, 3], "closest_frame": 10}

        self.assertIs(
            get_source_evidence_event("predicted-0", [observed_event], [predicted_event]),
            predicted_event,
        )
        self.assertIs(
            get_source_evidence_event("observed-0", [observed_event], [predicted_event]),
            observed_event,
        )

    def test_review_queue_does_not_reuse_a_decision_for_changed_evidence(self):
        timeline = build_race_control_timeline_dataframe(
            review_events=[],
            risk_events=[
                {
                    "track_ids": [1, 2],
                    "highest_risk_frame": 20,
                    "highest_risk_time_seconds": 0.8,
                    "predicted_separation_vehicle_widths": 0.3,
                    "review_clip_path": "risk.mp4",
                }
            ],
        )
        event = timeline.iloc[0]
        summary = get_human_review_event_summary(event)
        decisions = {
            "decisions": {
                event["review_event_id"]: {
                    "event": summary,
                    "outcome": "Cleared",
                    "reviewer_name": "Andre",
                    "updated_at": "2026-10-18T01:00:00+00:00",
                }
            }
        }

        reviewed_queue = build_human_review_queue(timeline, decisions)
        changed_timeline = timeline.copy()
        changed_timeline.loc[0, "measurement_value"] = 0.4
        changed_queue = build_human_review_queue(changed_timeline, decisions)

        self.assertEqual(reviewed_queue.iloc[0]["review_status"], "Cleared")
        self.assertEqual(changed_queue.iloc[0]["review_status"], "Unreviewed")

if __name__ == "__main__":
    unittest.main()
