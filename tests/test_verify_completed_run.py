import unittest

from scripts.verify_completed_run import validate_run_summary


def complete_summary():
    return {
        "all_observations": 1,
        "all_track_segments": 1,
        "frames_per_second": 25.0,
        "observations_per_frame": 1.0,
        "observed_close_approach_event_count": 0,
        "processed_frame_count": 1,
        "risk_event_count": 0,
        "spatial_analysis_enabled": False,
        "track_quality": {},
        "trusted_track_segments": 0,
    }


class CompletedRunVerificationTests(unittest.TestCase):
    def test_accepts_a_complete_summary(self):
        validate_run_summary(complete_summary())

    def test_rejects_a_summary_missing_a_required_field(self):
        summary = complete_summary()
        del summary["track_quality"]

        with self.assertRaisesRegex(ValueError, "track_quality"):
            validate_run_summary(summary)

    def test_rejects_a_non_boolean_spatial_analysis_flag(self):
        summary = complete_summary()
        summary["spatial_analysis_enabled"] = "no"

        with self.assertRaisesRegex(ValueError, "true or false"):
            validate_run_summary(summary)


if __name__ == "__main__":
    unittest.main()
