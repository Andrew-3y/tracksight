import unittest

from src.marshal_briefing import (
    HUMAN_REVIEW_DISCLAIMER,
    RECOMMENDED_HUMAN_ACTION,
    build_marshal_briefing,
    select_briefing_events,
)


class MarshalBriefingTests(unittest.TestCase):
    def test_keeps_predicted_and_observed_measurements_separate(self):
        briefing = build_marshal_briefing(
            run_summary={
                "processed_frame_count": 100,
                "frames_per_second": 25.0,
                "trusted_track_segments": 2,
                "spatial_analysis_status": "enabled_for_fixed_camera",
                "run_provenance": {"camera_mode": "fixed"},
            },
            risk_evidence=[
                {
                    "track_ids": [1, 2],
                    "highest_risk_frame": 50,
                    "highest_risk_time_seconds": 2.0,
                    "frame_count": 4,
                    "predicted_separation_vehicle_widths": 0.8,
                },
                {
                    "track_ids": [3, 4],
                    "highest_risk_frame": 75,
                    "highest_risk_time_seconds": 3.0,
                    "frame_count": 5,
                    "predicted_separation_vehicle_widths": 0.2,
                },
            ],
            review_evidence=[
                {
                    "track_ids": [5, 6],
                    "closest_frame": 25,
                    "closest_time_seconds": 1.0,
                    "frame_count": 3,
                    "closest_normalized_distance": 0.6,
                },
                {
                    "track_ids": [7, 8],
                    "closest_frame": 40,
                    "closest_time_seconds": 1.6,
                    "frame_count": 4,
                    "closest_normalized_distance": 0.3,
                },
            ],
        )

        self.assertEqual(briefing["disclaimer"], HUMAN_REVIEW_DISCLAIMER)
        self.assertEqual(
            briefing["predicted_risk_highlights"][0]["track_ids"], [3, 4]
        )
        self.assertEqual(
            briefing["observed_proximity_highlights"][0]["track_ids"], [7, 8]
        )
        self.assertEqual(
            briefing["predicted_risk_highlights"][0]["measurement"]["basis"],
            "average projected vehicle widths",
        )
        self.assertEqual(
            briefing["observed_proximity_highlights"][0]["measurement"]["basis"],
            "average detected-box diagonals",
        )
        self.assertEqual(
            briefing["predicted_risk_highlights"][0]["recommended_human_action"],
            RECOMMENDED_HUMAN_ACTION,
        )

    def test_rejects_zero_highlights(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            build_marshal_briefing({}, [], [], max_highlights=0)

    def test_includes_a_grounded_safety_candidate_highlight(self):
        briefing = build_marshal_briefing(
            run_summary={
                "processed_frame_count": 100,
                "frames_per_second": 25.0,
                "trusted_track_segments": 1,
                "track_quality": {"7": {"trusted_for_safety_analysis": True}},
            },
            risk_evidence=[],
            review_evidence=[],
            safety_candidate_evidence={
                "stationary_vehicle_candidates": [
                    {
                        "candidate_type": "stationary_vehicle",
                        "track_ids": [7],
                        "track_id": 7,
                        "anchor_frame": 25,
                        "anchor_time_seconds": 1.0,
                        "frame_count": 20,
                        "maximum_spread_vehicle_widths": 0.4,
                        "measurement_basis": "projected vehicle widths",
                        "review_reason": "Vehicle ID 7 remained in one place.",
                        "recommended_human_action": "Review it.",
                    }
                ],
                "restricted_area_entry_candidates": [],
            },
        )

        highlight = briefing["safety_candidate_highlights"][0]
        self.assertEqual(highlight["evidence_type"], "stationary_vehicle")
        self.assertIn("passed the saved continuity gate", highlight["evidence_grounding"])

    def test_selects_only_the_requested_number_of_closest_events(self):
        events = [
            {"distance": 0.8},
            {"distance": 0.2},
            {"distance": 0.5},
        ]

        selected = select_briefing_events(events, "distance", 2)

        self.assertEqual(selected, [{"distance": 0.2}, {"distance": 0.5}])


if __name__ == "__main__":
    unittest.main()
