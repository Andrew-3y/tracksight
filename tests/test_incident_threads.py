import unittest

from src.incident_threads import build_incident_threads


def make_event(key, review_id, evidence_type, track_ids, time_seconds, frame):
    return {
        "event_key": key,
        "review_event_id": review_id,
        "evidence_type": evidence_type,
        "track_id_values": track_ids,
        "time_seconds": time_seconds,
        "frame": frame,
        "measurement_value": 0.2,
        "measurement_basis": "local review units",
        "clip": None,
    }


class IncidentThreadTests(unittest.TestCase):
    def test_groups_nearby_cross_type_evidence_for_the_same_vehicle_pair(self):
        threads = build_incident_threads(
            [
                make_event("predicted-0", "p0", "Predicted risk", [1, 2], 1.0, 25),
                make_event("observed-0", "o0", "Observed proximity", [1, 2], 1.8, 45),
                make_event("predicted-1", "p1", "Predicted risk", [8, 9], 2.0, 50),
            ]
        )

        self.assertEqual(len(threads), 2)
        self.assertEqual(threads[0]["track_ids"], [1, 2])
        self.assertEqual(threads[0]["event_count"], 2)
        self.assertEqual(threads[0]["predicted_risk_event_count"], 1)
        self.assertEqual(threads[0]["observed_proximity_event_count"], 1)

    def test_does_not_join_nearby_events_that_only_share_one_vehicle(self):
        threads = build_incident_threads(
            [
                make_event("predicted-0", "p0", "Predicted risk", [1, 2], 1.0, 25),
                make_event("predicted-1", "p1", "Predicted risk", [2, 3], 1.1, 28),
            ]
        )

        self.assertEqual(len(threads), 2)

    def test_maximum_duration_prevents_one_long_chain_becoming_one_case(self):
        threads = build_incident_threads(
            [
                make_event("predicted-0", "p0", "Predicted risk", [1, 2], 0.0, 0),
                make_event("predicted-1", "p1", "Predicted risk", [1, 2], 1.0, 25),
                make_event("predicted-2", "p2", "Predicted risk", [1, 2], 2.0, 50),
            ],
            gap_seconds=1.5,
            max_duration_seconds=1.5,
        )

        self.assertEqual([thread["event_count"] for thread in threads], [2, 1])


if __name__ == "__main__":
    unittest.main()
