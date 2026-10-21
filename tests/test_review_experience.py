import unittest

from src.review_experience import (
    build_evidence_coverage,
    build_incident_replay_context_points,
    build_incident_replay_points,
    build_incident_replay_summary,
)


class IncidentReplayTests(unittest.TestCase):
    def test_replay_keeps_only_selected_tracks_and_splits_missing_frame_gaps(self):
        trajectories = {
            "1": [
                {"frame_index": 9, "time_seconds": 0.36, "center": [10, 20], "confidence": 0.9},
                {"frame_index": 10, "time_seconds": 0.4, "center": [12, 20], "confidence": 0.9},
                {"frame_index": 12, "time_seconds": 0.48, "center": [16, 20], "confidence": 0.9},
            ],
            "2": [
                {"frame_index": 10, "time_seconds": 0.4, "center": [30, 20], "confidence": 0.8, "track_position": [3, 2]},
            ],
            "3": [
                {"frame_index": 10, "time_seconds": 0.4, "center": [50, 20], "confidence": 0.8},
            ],
        }

        points = build_incident_replay_points(
            trajectories,
            track_ids=[1, 2],
            event_time_seconds=0.4,
            event_frame=10,
            pre_seconds=0.1,
            post_seconds=0.1,
        )

        self.assertEqual([point["track_id"] for point in points], [1, 1, 1, 2])
        self.assertEqual([point["replay_segment"] for point in points[:3]], ["1-0", "1-0", "1-1"])
        self.assertTrue(points[1]["is_event_frame"])
        self.assertEqual(points[3]["track_x"], 3.0)

    def test_context_snapshot_uses_only_exact_projected_event_positions(self):
        trajectories = {
            "1": [
                {"frame_index": 9, "track_position": [1, 1]},
                {"frame_index": 10, "track_position": [2, 2]},
            ],
            "2": [{"frame_index": 10, "track_position": [3, 2]}],
            "3": [{"frame_index": 11, "track_position": [4, 2]}],
            "4": [{"frame_index": 10}],
        }

        context_points = build_incident_replay_context_points(
            trajectories,
            track_ids=[1, 2],
            event_frame=10,
        )

        self.assertEqual([point["track_id"] for point in context_points], [1, 2])
        self.assertTrue(all(point["is_selected"] for point in context_points))
        self.assertEqual(context_points[0]["track_x"], 2.0)

    def test_replay_summary_reports_missing_vehicle_and_gaps(self):
        points = [
            {"track_id": 1, "replay_segment": "1-0", "track_x": 1.0, "track_y": 2.0},
            {"track_id": 1, "replay_segment": "1-1", "track_x": 2.0, "track_y": 2.0},
        ]

        summary = build_incident_replay_summary(points, [1, 2])

        self.assertEqual(summary["missing_track_ids"], [2])
        self.assertTrue(summary["contains_observation_gaps"])
        self.assertTrue(summary["has_calibrated_positions"])


class EvidenceCoverageTests(unittest.TestCase):
    def test_coverage_reports_only_trusted_real_observation_frames(self):
        run_summary = {
            "processed_frame_count": 10,
            "all_track_segments": 2,
            "spatial_analysis_enabled": True,
            "track_quality": {
                "1": {"trusted_for_safety_analysis": True},
                "2": {"trusted_for_safety_analysis": False},
            },
        }
        trajectories = {
            "1": [
                {"frame_index": 0, "track_position": [1, 2]},
                {"frame_index": 2, "track_position": [2, 2]},
            ],
            "2": [
                {"frame_index": 1, "track_position": [3, 2]},
            ],
        }

        coverage = build_evidence_coverage(run_summary, trajectories)

        self.assertEqual(coverage["trusted_track_segments"], 1)
        self.assertEqual(coverage["untrusted_track_segments"], 1)
        self.assertEqual(coverage["trusted_observation_frames"], 2)
        self.assertEqual(coverage["projected_trusted_frames"], 2)
        self.assertEqual(coverage["trusted_frame_ratio"], 0.2)
        self.assertTrue(any("Estimated tracker positions" in reason for reason in coverage["exclusion_reasons"]))

    def test_coverage_explains_missing_calibration_without_calling_it_accuracy(self):
        coverage = build_evidence_coverage(
            {
                "processed_frame_count": 5,
                "all_track_segments": 1,
                "spatial_analysis_enabled": False,
                "track_quality": {"1": {"trusted_for_safety_analysis": True}},
            },
            {"1": [{"frame_index": 0}]},
        )

        self.assertFalse(coverage["spatial_analysis_enabled"])
        self.assertTrue(any("no usable camera calibration" in reason for reason in coverage["exclusion_reasons"]))


if __name__ == "__main__":
    unittest.main()
