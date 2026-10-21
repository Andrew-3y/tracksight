import unittest

from src.camera_trust import build_camera_trust_monitor


class CameraTrustTests(unittest.TestCase):
    def test_old_moving_run_does_not_invent_stabilization_outcomes(self):
        monitor = build_camera_trust_monitor(
            {
                "spatial_analysis_enabled": True,
                "run_provenance": {"camera_mode": "moving", "camera_motion_method": "sparseOptFlow"},
                "track_quality": {"1": {"trusted_for_safety_analysis": True}},
            }
        )
        self.assertFalse(monitor["stabilization"]["recorded"])
        self.assertEqual(monitor["stabilization"]["status"], "Not recorded by this older run")

    def test_new_moving_run_reports_saved_outcomes(self):
        monitor = build_camera_trust_monitor(
            {
                "spatial_analysis_enabled": True,
                "run_provenance": {"camera_mode": "moving"},
                "track_quality": {"1": {"trusted_for_safety_analysis": True}},
                "camera_trust_diagnostics": {
                    "frame_outcomes": {"transform_updated": 80, "retained_insufficient_features": 3}
                },
            }
        )
        self.assertTrue(monitor["stabilization"]["recorded"])
        self.assertEqual(monitor["stabilization"]["updated_frames"], 80)
        self.assertEqual(monitor["stabilization"]["retained_frames"], 3)

    def test_uncalibrated_run_explains_why_stabilization_is_unavailable(self):
        monitor = build_camera_trust_monitor(
            {
                "spatial_analysis_enabled": False,
                "run_provenance": {"camera_mode": "unknown"},
                "track_quality": {},
                "camera_trust_diagnostics": {
                    "method": "not_available_without_camera_calibration",
                    "frame_outcomes": {},
                },
            }
        )
        self.assertFalse(monitor["stabilization"]["recorded"])
        self.assertIn("no camera calibration", monitor["stabilization"]["detail"])


if __name__ == "__main__":
    unittest.main()
