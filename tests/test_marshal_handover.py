import unittest

from src.marshal_handover import build_marshal_handover


class MarshalHandoverTests(unittest.TestCase):
    def test_keeps_the_next_items_review_only_and_prioritizes_escalated_work(self):
        summary = {
            "processed_frame_count": 100,
            "frames_per_second": 25.0,
            "all_track_segments": 1,
            "spatial_analysis_enabled": True,
            "track_quality": {"1": {"trusted_for_safety_analysis": True}},
        }
        rows = [
            {"event_key": "a", "review_status": "Unreviewed", "evidence_type": "Observed proximity", "track_id_values": [1, 2], "frame": 10, "time_seconds": 1.0, "measurement_basis": "box diagonals", "clip": None},
            {"event_key": "b", "review_status": "Escalate", "evidence_type": "Predicted risk", "track_id_values": [3, 4], "frame": 5, "time_seconds": 0.5, "measurement_basis": "vehicle widths", "clip": "clip.mp4"},
        ]
        handover = build_marshal_handover(summary, {"1": []}, rows)

        self.assertEqual(handover["review_status"]["escalated"], 1)
        self.assertEqual(handover["next_review_items"][0]["review_status"], "Escalate")
        self.assertIn("does not determine fault", handover["disclaimer"])
