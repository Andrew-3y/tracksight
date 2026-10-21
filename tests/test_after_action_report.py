import unittest

from src.after_action_report import build_after_action_report_pdf


class AfterActionReportTests(unittest.TestCase):
    def test_creates_a_pdf_from_selected_evidence_only(self):
        report = build_after_action_report_pdf(
            "demo",
            {"track_ids": [7, 11], "start_time_seconds": 4.0, "end_time_seconds": 4.5},
            {"status": "Review now", "recommended_action": "Inspect the replay.", "source_evidence_types": ["Observed proximity"]},
            [{"source": "Annotated primary camera", "status": "Available", "detail": "Recorded video."}],
            [{"event_key": "observed-0"}],
        )

        self.assertTrue(report.startswith(b"%PDF"))
