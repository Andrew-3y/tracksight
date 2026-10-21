import unittest

from src.risk_hotspots import (
    build_calibrated_event_locations,
    build_candidate_hotspots,
)


class CandidateHotspotTests(unittest.TestCase):
    def test_locations_use_pair_midpoints_only_when_both_positions_exist(self):
        events = [
            {
                "event_key": "predicted-0",
                "evidence_type": "Predicted risk",
                "track_id_values": [1, 2],
                "frame": 10,
                "time_seconds": 0.4,
            },
            {
                "event_key": "predicted-1",
                "evidence_type": "Predicted risk",
                "track_id_values": [1, 3],
                "frame": 10,
                "time_seconds": 0.4,
            },
        ]
        trajectories = {
            "1": [
                {
                    "frame_index": 10,
                    "track_position": [10.0, 20.0],
                    "track_width_units": 2.0,
                }
            ],
            "2": [
                {
                    "frame_index": 10,
                    "track_position": [14.0, 20.0],
                    "track_width_units": 4.0,
                }
            ],
            "3": [{"frame_index": 10}],
        }

        locations = build_calibrated_event_locations(events, trajectories)

        self.assertEqual(len(locations), 1)
        self.assertEqual(locations[0]["track_x"], 12.0)
        self.assertEqual(locations[0]["average_vehicle_width"], 3.0)
        self.assertEqual(locations[0]["track_ids_label"], "1, 2")

    def test_nearby_same_type_events_form_one_hotspot_with_adaptive_radius(self):
        locations = [
            {
                "event_key": "predicted-0",
                "evidence_type": "Predicted risk",
                "track_x": 10.0,
                "track_y": 10.0,
                "average_vehicle_width": 2.0,
                "track_ids": (1, 2),
                "time_seconds": 1.0,
            },
            {
                "event_key": "predicted-1",
                "evidence_type": "Predicted risk",
                "track_x": 18.0,
                "track_y": 10.0,
                "average_vehicle_width": 2.0,
                "track_ids": (2, 3),
                "time_seconds": 2.0,
            },
            {
                "event_key": "predicted-2",
                "evidence_type": "Predicted risk",
                "track_x": 50.0,
                "track_y": 10.0,
                "average_vehicle_width": 2.0,
                "track_ids": (4, 5),
                "time_seconds": 3.0,
            },
        ]

        hotspots, radius = build_candidate_hotspots(
            locations,
            neighborhood_vehicle_widths=5.0,
        )

        self.assertEqual(radius, 10.0)
        self.assertEqual(len(hotspots), 1)
        self.assertEqual(hotspots[0]["candidate_event_count"], 2)
        self.assertEqual(hotspots[0]["vehicle_ids"], [1, 2, 3])

    def test_chain_of_events_does_not_become_one_misleading_long_hotspot(self):
        locations = [
            {
                "event_key": event_key,
                "evidence_type": "Predicted risk",
                "track_x": track_x,
                "track_y": 10.0,
                "average_vehicle_width": 2.0,
                "track_ids": (1, 2),
                "time_seconds": time_seconds,
            }
            for event_key, track_x, time_seconds in [
                ("predicted-0", 0.0, 1.0),
                ("predicted-1", 9.0, 2.0),
                ("predicted-2", 18.0, 3.0),
            ]
        ]

        hotspots, radius = build_candidate_hotspots(
            locations,
            neighborhood_vehicle_widths=5.0,
        )

        self.assertEqual(radius, 10.0)
        self.assertEqual(len(hotspots), 1)
        self.assertEqual(hotspots[0]["candidate_event_count"], 2)

    def test_different_evidence_types_do_not_form_one_hotspot(self):
        locations = [
            {
                "event_key": "predicted-0",
                "evidence_type": "Predicted risk",
                "track_x": 10.0,
                "track_y": 10.0,
                "average_vehicle_width": 2.0,
                "track_ids": (1, 2),
                "time_seconds": 1.0,
            },
            {
                "event_key": "observed-0",
                "evidence_type": "Observed proximity",
                "track_x": 10.0,
                "track_y": 10.0,
                "average_vehicle_width": 2.0,
                "track_ids": (1, 2),
                "time_seconds": 2.0,
            },
        ]

        hotspots, _ = build_candidate_hotspots(locations)

        self.assertEqual(hotspots, [])


if __name__ == "__main__":
    unittest.main()
