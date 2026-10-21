import unittest

from src.safety_candidates import (
    build_safety_candidate_evidence,
    find_against_expected_direction_candidates,
    find_restricted_area_entry_candidates,
    find_stationary_in_active_racing_area_candidates,
    find_stationary_vehicle_candidates,
)


def point(frame_index, x, y, width=10.0):
    return {
        "frame_index": frame_index,
        "time_seconds": frame_index / 10.0,
        "track_position": [x, y],
        "track_width_units": width,
    }


def moving_point(frame_index, x, y, velocity_x):
    item = point(frame_index, x, y)
    item["motion"] = {
        "status": "measured",
        "velocity_x_track_units_per_second": velocity_x,
        "velocity_y_track_units_per_second": 0.0,
    }
    return item


class SafetyCandidateTests(unittest.TestCase):
    def test_finds_a_sustained_low_displacement_vehicle_interval(self):
        candidates = find_stationary_vehicle_candidates(
            {"7": [point(frame, 100.0 + (frame % 2) * 0.1, 50.0) for frame in range(12)]},
            minimum_stationary_seconds=1.0,
            maximum_spread_vehicle_widths=0.75,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["track_ids"], [7])
        self.assertEqual(candidates[0]["start_frame"], 0)
        self.assertGreaterEqual(candidates[0]["duration_seconds"], 1.0)
        self.assertLess(candidates[0]["maximum_spread_vehicle_widths"], 0.75)

    def test_does_not_call_a_moving_vehicle_stationary(self):
        candidates = find_stationary_vehicle_candidates(
            {"7": [point(frame, frame * 3.0, 50.0) for frame in range(15)]},
            minimum_stationary_seconds=1.0,
            maximum_spread_vehicle_widths=0.75,
        )

        self.assertEqual(candidates, [])

    def test_detection_gap_does_not_create_an_invented_stationary_interval(self):
        candidates = find_stationary_vehicle_candidates(
            {
                "7": [
                    point(0, 100.0, 50.0),
                    point(1, 100.1, 50.0),
                    point(2, 100.0, 50.0),
                    point(10, 100.1, 50.0),
                    point(11, 100.0, 50.0),
                    point(12, 100.1, 50.0),
                ]
            },
            minimum_stationary_seconds=0.5,
        )

        self.assertEqual(candidates, [])

    def test_restricted_area_requires_an_explicit_reviewer_rule(self):
        observations = [
            {
                "zone_id": "runoff",
                "zone_label": "Runoff",
                "zone_category": "restricted area",
                "zone_review_note": "Review only.",
                "track_id": 7,
                "start_frame": 10,
                "end_frame": 13,
                "start_seconds": 1.0,
                "end_seconds": 1.3,
                "frame_count": 4,
            }
        ]
        zones = [
            {
                "zone_id": "runoff",
                "candidate_rule": "restricted_area_entry",
            }
        ]

        candidates = find_restricted_area_entry_candidates(observations, zones, 3)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["candidate_type"], "restricted_area_entry")
        self.assertEqual(candidates[0]["track_ids"], [7])

    def test_without_calibration_candidates_are_withheld(self):
        evidence = build_safety_candidate_evidence(
            {"7": [point(frame, 100.0, 50.0) for frame in range(12)]},
            spatial_analysis_enabled=False,
        )

        self.assertEqual(
            evidence["analysis_status"], "withheld_without_camera_calibration"
        )
        self.assertEqual(evidence["stationary_vehicle_candidates"], [])

    def test_direction_prompt_requires_measured_opposed_motion_in_configured_zone(self):
        candidates = find_against_expected_direction_candidates(
            {"7": [moving_point(frame, 8 - frame, 5, -10.0) for frame in range(4)]},
            [{
                "zone_id": "corridor", "label": "Corridor", "category": "active racing area",
                "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "candidate_rule": "against_expected_direction",
                "expected_local_track_direction": [1.0, 0.0],
            }],
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["candidate_type"], "against_expected_direction")
        self.assertEqual(candidates[0]["track_ids"], [7])

    def test_active_racing_area_prompt_requires_an_explicit_zone_rule(self):
        candidates = find_stationary_in_active_racing_area_candidates(
            {"7": [point(frame, 5.0, 5.0) for frame in range(12)]},
            [{
                "zone_id": "surface", "label": "Racing surface", "category": "active racing area",
                "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "candidate_rule": "stationary_in_active_racing_area",
            }],
            minimum_stationary_seconds=1.0,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["candidate_type"], "stationary_in_active_racing_area")


if __name__ == "__main__":
    unittest.main()
