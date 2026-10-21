import unittest
from math import nan

from src.race_control_export import build_race_control_timeline_export


class RaceControlExportTests(unittest.TestCase):
    def test_export_removes_local_paths_and_preserves_review_status(self):
        report = build_race_control_timeline_export(
            "run", {"run_provenance": {"source_video": "C:/private/demo.mp4", "camera_mode": "fixed"}},
            [{"event_key": "a", "evidence_type": "Observed proximity", "track_id_values": [1, 2], "frame": 10, "time_seconds": 1.0, "measurement_value": 0.4, "measurement_basis": "box diagonals", "review_status": "Needs review", "clip": "C:/private/clip.mp4"}],
        )

        self.assertEqual(report["source_video"], "demo.mp4")
        self.assertEqual(report["timeline"][0]["supporting_clip"], "clip.mp4")

    def test_ignores_missing_dataframe_clip_value(self):
        report = build_race_control_timeline_export(
            "run", {},
            [{"event_key": "a", "evidence_type": "Observed proximity", "track_id_values": [1, 2], "frame": 10, "time_seconds": 1.0, "measurement_value": 0.4, "measurement_basis": "box diagonals", "clip": nan}],
        )
        self.assertIsNone(report["timeline"][0]["supporting_clip"])
