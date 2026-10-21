import unittest

from src.race_control_log import (
    append_race_control_action,
    empty_race_control_action_log,
)


CASE = {
    "case_id": "thread-1-2",
    "track_ids": [1, 2],
    "track_ids_label": "1, 2",
    "start_time_seconds": 1.0,
    "end_time_seconds": 1.2,
    "source_event_keys": ["observed-0"],
    "priority": "Monitor",
}
CHECKLIST = {
    "supporting_video_reviewed": True,
    "identity_continuity_considered": True,
    "camera_trust_reviewed": True,
}


class RaceControlLogTests(unittest.TestCase):
    def test_append_keeps_the_prior_action_and_case_snapshot(self):
        original = empty_race_control_action_log()
        first = append_race_control_action(original, CASE, "Monitor", "Alex", "Checked video.", CHECKLIST, "2026-01-01T00:00:00+00:00")
        second = append_race_control_action(first, CASE, "Close as reviewed", "Alex", "No further action.", CHECKLIST, "2026-01-01T00:01:00+00:00")
        self.assertEqual(original["actions"], [])
        self.assertEqual(len(second["actions"]), 2)
        self.assertEqual(second["actions"][0]["track_ids"], [1, 2])
        self.assertEqual(second["actions"][0]["priority_snapshot"], "Monitor")

    def test_rejects_empty_human_provenance(self):
        with self.assertRaises(ValueError):
            append_race_control_action(empty_race_control_action_log(), CASE, "Monitor", "", "Note", CHECKLIST)

    def test_rejects_an_incomplete_review_checklist(self):
        with self.assertRaises(ValueError):
            append_race_control_action(
                empty_race_control_action_log(),
                CASE,
                "Monitor",
                "Alex",
                "Checked evidence.",
                {"supporting_video_reviewed": True},
            )


if __name__ == "__main__":
    unittest.main()
