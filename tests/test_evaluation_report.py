import unittest

from src.evaluation_report import build_evaluation_report


class EvaluationReportTests(unittest.TestCase):
    def test_does_not_claim_unmeasured_accuracy(self):
        report = build_evaluation_report(
            {
                "processed_frame_count": 20,
                "frames_per_second": 10.0,
                "all_observations": 30,
                "all_track_segments": 2,
                "trusted_track_segments": 1,
                "observed_close_approach_event_count": 2,
                "risk_event_count": 1,
                "spatial_analysis_enabled": True,
                "spatial_analysis_status": "enabled_for_fixed_camera",
                "track_quality": {"1": {"trusted_for_safety_analysis": True}},
                "run_provenance": {"source_video": "data/raw/demo.mp4", "camera_mode": "fixed"},
            },
            {"1": [{"frame_index": 0, "track_position": [1, 1]}]},
        )

        self.assertIn("not a detector", report["disclaimer"])
        self.assertIn("ground truth", " ".join(report["known_limitations"]))
        self.assertEqual(report["evaluated_run_context"]["source_video"], "demo.mp4")
