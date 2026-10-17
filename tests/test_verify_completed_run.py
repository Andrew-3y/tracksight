import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.verify_completed_run import validate_marshal_briefing, validate_run_summary


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

    def test_rejects_a_briefing_with_mismatched_evidence_counts(self):
        run_summary = {
            "risk_event_count": 2,
            "observed_close_approach_event_count": 3,
        }
        briefing = {
            "briefing_type": "human_review",
            "summary": {
                "candidate_risk_event_count": 1,
                "observed_proximity_event_count": 3,
            },
            "predicted_risk_highlights": [],
            "observed_proximity_highlights": [],
        }

        with self.assertRaisesRegex(ValueError, "risk count"):
            validate_marshal_briefing(briefing, run_summary)

    def test_rejects_a_briefing_highlight_without_a_saved_clip(self):
        run_summary = {
            "risk_event_count": 1,
            "observed_close_approach_event_count": 0,
        }
        briefing = {
            "briefing_type": "human_review",
            "summary": {
                "candidate_risk_event_count": 1,
                "observed_proximity_event_count": 0,
            },
            "predicted_risk_highlights": [{"supporting_clip": None}],
            "observed_proximity_highlights": [],
        }

        with self.assertRaisesRegex(ValueError, "supporting clip"):
            validate_marshal_briefing(briefing, run_summary)

    def test_accepts_a_briefing_highlight_with_a_saved_clip(self):
        with TemporaryDirectory() as temporary_directory:
            clip_path = Path(temporary_directory) / "briefing.mp4"
            clip_path.touch()
            run_summary = {
                "risk_event_count": 1,
                "observed_close_approach_event_count": 0,
            }
            briefing = {
                "briefing_type": "human_review",
                "summary": {
                    "candidate_risk_event_count": 1,
                    "observed_proximity_event_count": 0,
                },
                "predicted_risk_highlights": [
                    {"supporting_clip": str(clip_path)}
                ],
                "observed_proximity_highlights": [],
            }

            with patch("scripts.verify_completed_run.validate_video") as validate_video:
                validate_marshal_briefing(briefing, run_summary)

            validate_video.assert_called_once_with(
                str(clip_path),
                "Marshal briefing supporting clip",
            )


if __name__ == "__main__":
    unittest.main()
