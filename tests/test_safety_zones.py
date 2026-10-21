import unittest

from src.safety_zones import (
    build_safety_zone_observations,
    is_point_in_polygon,
    validate_safety_zone_profile,
)


ZONE = {
    "zone_id": "pit-exit",
    "label": "Pit exit merge",
    "category": "merge area",
    "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
}


class SafetyZoneTests(unittest.TestCase):
    def test_polygon_includes_its_boundary_but_not_outside_positions(self):
        self.assertTrue(is_point_in_polygon([0, 5], ZONE["polygon"]))
        self.assertTrue(is_point_in_polygon([5, 5], ZONE["polygon"]))
        self.assertFalse(is_point_in_polygon([11, 5], ZONE["polygon"]))

    def test_zone_observations_use_only_consecutive_real_positions(self):
        observations = build_safety_zone_observations(
            {
                "1": [
                    {"frame_index": 0, "time_seconds": 0.0, "track_position": [5, 5]},
                    {"frame_index": 1, "time_seconds": 0.04, "track_position": [6, 5]},
                    {"frame_index": 3, "time_seconds": 0.12, "track_position": [7, 5]},
                ],
                "2": [{"frame_index": 0, "time_seconds": 0.0, "track_position": [5, 5]}],
            },
            [ZONE],
            trusted_track_ids={1},
        )

        self.assertEqual(len(observations), 2)
        self.assertEqual(observations[0]["track_id"], 1)
        self.assertEqual(observations[0]["frame_count"], 2)
        self.assertEqual(observations[1]["start_frame"], 3)

    def test_rejects_a_zone_profile_without_local_track_coordinates(self):
        with self.assertRaisesRegex(ValueError, "local_track_units"):
            validate_safety_zone_profile(
                {"schema_version": 1, "coordinate_system": "pixels", "zones": [ZONE]}
            )

    def test_preserves_an_explicit_restricted_area_candidate_rule(self):
        profile = validate_safety_zone_profile(
            {
                "schema_version": 1,
                "coordinate_system": "local_track_units",
                "zones": [{**ZONE, "candidate_rule": "restricted_area_entry"}],
            }
        )

        self.assertEqual(
            profile["zones"][0]["candidate_rule"], "restricted_area_entry"
        )

    def test_requires_a_nonzero_local_direction_for_a_direction_policy(self):
        with self.assertRaisesRegex(ValueError, "must not be zero"):
            validate_safety_zone_profile(
                {
                    "schema_version": 1,
                    "coordinate_system": "local_track_units",
                    "zones": [
                        {
                            **ZONE,
                            "candidate_rule": "against_expected_direction",
                            "expected_local_track_direction": [0, 0],
                        }
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
