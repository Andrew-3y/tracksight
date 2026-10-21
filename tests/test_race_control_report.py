import unittest

from src.race_control_report import build_race_control_report_pdf


class RaceControlReportTests(unittest.TestCase):
    def test_generates_pdf_bytes(self):
        report = build_race_control_report_pdf(
            "demo_run",
            {"processed_frame_count": 250, "frames_per_second": 25.0, "trusted_track_segments": 3},
            [{"priority": "Review now", "track_ids_label": "1, 2", "start_time_seconds": 1.0, "end_time_seconds": 1.4, "event_count": 2}],
            {"actions": []},
        )
        self.assertTrue(report.startswith(b"%PDF"))
        self.assertGreater(len(report), 1000)


if __name__ == "__main__":
    unittest.main()
