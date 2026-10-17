import unittest

from src.track_quality import build_run_summary


class BuildRunSummaryTests(unittest.TestCase):
    def test_summary_reports_positions_per_processed_frame(self):
        summary = build_run_summary(
            trajectories={
                1: [{"frame_index": 0}, {"frame_index": 1}],
                2: [{"frame_index": 1}],
            },
            track_quality={
                "1": {"trusted_for_safety_analysis": True},
                "2": {"trusted_for_safety_analysis": False},
            },
            risk_evidence=[],
            review_evidence=[],
            pipeline_version="test",
            spatial_analysis_enabled=False,
            processed_frame_count=2,
            frames_per_second=30.0,
        )

        self.assertEqual(summary["all_observations"], 3)
        self.assertEqual(summary["processed_frame_count"], 2)
        self.assertEqual(summary["observations_per_frame"], 1.5)
        self.assertEqual(
            summary["spatial_analysis_status"],
            "disabled_without_camera_calibration",
        )

    def test_summary_keeps_explicit_run_provenance(self):
        summary = build_run_summary(
            trajectories={},
            track_quality={},
            risk_evidence=[],
            review_evidence=[],
            pipeline_version="test",
            spatial_analysis_enabled=False,
            processed_frame_count=0,
            frames_per_second=25.0,
            run_provenance={"detector_model": "model.pt"},
        )

        self.assertEqual(summary["run_provenance"], {"detector_model": "model.pt"})


if __name__ == "__main__":
    unittest.main()
