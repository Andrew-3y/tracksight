import unittest
from math import nan

from src.learning_queue import build_learning_export, build_learning_queue


class LearningQueueTests(unittest.TestCase):
    def test_excludes_cleared_evidence_and_does_not_call_tasks_ground_truth(self):
        tasks = build_learning_queue([
            {"event_key": "one", "review_status": "Cleared", "evidence_type": "Observed proximity", "track_id_values": [1, 2], "frame": 1, "time_seconds": 0.1, "clip": None},
            {"event_key": "two", "review_status": "Unreviewed", "evidence_type": "Predicted risk", "track_id_values": [3, 4], "frame": 2, "time_seconds": 0.2, "clip": "clip.mp4"},
        ])
        export = build_learning_export("demo", tasks)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["labeling_status"], "needs_human_annotation")
        self.assertIn("not ground truth", export["disclaimer"])

    def test_ignores_a_missing_dataframe_clip_value(self):
        tasks = build_learning_queue([
            {"event_key": "one", "review_status": "Unreviewed", "evidence_type": "Observed proximity", "track_id_values": [1, 2], "frame": 1, "time_seconds": 0.1, "clip": nan},
        ])
        self.assertIsNone(tasks[0]["supporting_clip"])
