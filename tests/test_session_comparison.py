import unittest

from src.session_comparison import (
    build_multi_run_risk_map,
    build_session_comparison,
    build_session_snapshot,
)


def snapshot(name, profile="camera.json", x=10.0, evidence_type="Observed proximity"):
    summary = {
        "spatial_analysis_enabled": True,
        "run_provenance": {"calibration_profile": profile},
        "processed_frame_count": 100,
        "frames_per_second": 25.0,
        "trusted_track_segments": 2,
        "observed_close_approach_event_count": 1,
        "risk_event_count": 0,
    }
    locations = [{
        "event_key": f"{name}-event", "evidence_type": evidence_type,
        "time_seconds": 1.0, "track_x": x, "track_y": 20.0,
        "average_vehicle_width": 2.0,
    }]
    return build_session_snapshot(name, summary, locations, [])


class SessionComparisonTests(unittest.TestCase):
    def test_different_calibrations_are_blocked(self):
        report = build_session_comparison(snapshot("one"), snapshot("two", "other.json"))
        self.assertFalse(report["comparable"])

    def test_recurring_map_requires_same_evidence_type_across_runs(self):
        recurring, _, _ = build_multi_run_risk_map([snapshot("one"), snapshot("two", x=11.0)])
        self.assertEqual(len(recurring), 1)
        other_type, _, _ = build_multi_run_risk_map([snapshot("one"), snapshot("two", x=11.0, evidence_type="Predicted risk")])
        self.assertEqual(other_type, [])


if __name__ == "__main__":
    unittest.main()
