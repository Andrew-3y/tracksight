import unittest
from pathlib import Path

from src.event_correlation import build_event_correlation


class EventCorrelationTests(unittest.TestCase):
    def test_keeps_missing_optional_sources_visible(self):
        correlation = build_event_correlation(
            [{"event_key": "observed-0", "clip": None}],
            {"spatial_analysis_enabled": False, "run_provenance": {}},
            Path("missing_annotated.mp4"),
        )

        statuses = {item["source"]: item["status"] for item in correlation}
        self.assertEqual(statuses["Annotated primary camera"], "Unavailable")
        self.assertEqual(statuses["Optional telemetry"], "Not attached")
        self.assertEqual(statuses["Supporting cameras"], "Not attached")

    def test_counts_only_aligned_telemetry_for_the_selected_case(self):
        correlation = build_event_correlation(
            [{"event_key": "observed-0", "clip": None}],
            {"spatial_analysis_enabled": True, "run_provenance": {}},
            Path("missing_annotated.mp4"),
            [
                {"event_key": "observed-0", "alignment_status": "aligned"},
                {"event_key": "other", "alignment_status": "aligned"},
            ],
        )

        telemetry = next(item for item in correlation if item["source"] == "Optional telemetry")
        self.assertEqual(telemetry["status"], "1 aligned")
