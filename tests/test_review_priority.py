import unittest

from src.review_priority import (
    INSUFFICIENT_EVIDENCE,
    MONITOR,
    REVIEW_NOW,
    build_review_priority_queue,
)


def summary(*trusted):
    return {
        "spatial_analysis_enabled": True,
        "track_quality": {
            str(track_id): {"trusted_for_safety_analysis": value}
            for track_id, value in trusted
        },
    }


def thread(track_ids, event_count):
    return {
        "thread_id": f"thread-{'-'.join(map(str, track_ids))}",
        "pair_track_ids": track_ids,
        "earliest_time_seconds": 1.0,
        "latest_time_seconds": 1.2,
        "event_count": event_count,
        "predicted_risk_event_count": event_count,
        "observed_proximity_event_count": 0,
        "event_keys": ["predicted-0"] * event_count,
        "review_event_ids": ["event-0"] * event_count,
        "preferred_clip": None,
        "preferred_event_key": "predicted-0",
    }


class ReviewPriorityTests(unittest.TestCase):
    def test_repeated_evidence_with_trusted_ids_is_review_now(self):
        queue = build_review_priority_queue([thread([1, 2], 2)], summary((1, True), (2, True)))
        self.assertEqual(queue[0]["priority"], REVIEW_NOW)

    def test_single_trusted_record_is_monitor_not_a_risk_score(self):
        queue = build_review_priority_queue([thread([1, 2], 1)], summary((1, True), (2, True)))
        self.assertEqual(queue[0]["priority"], MONITOR)

    def test_untrusted_pair_cannot_be_priority_review_now(self):
        queue = build_review_priority_queue([thread([1, 2], 3)], summary((1, True), (2, False)))
        self.assertEqual(queue[0]["priority"], INSUFFICIENT_EVIDENCE)


if __name__ == "__main__":
    unittest.main()
