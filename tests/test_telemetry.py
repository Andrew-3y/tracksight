import unittest

from src.telemetry import correlate_telemetry_events, parse_telemetry_csv


class TelemetryTests(unittest.TestCase):
    def test_keeps_only_numeric_optional_fields_and_aligns_inside_window(self):
        rows = parse_telemetry_csv("time_seconds,speed,comment\n1.0,123,ignore\n")
        results = correlate_telemetry_events([{"event_key": "a", "time_seconds": 1.2}], rows, 0.25)

        self.assertEqual(rows[0]["values"], {"speed": 123.0})
        self.assertEqual(results[0]["alignment_status"], "aligned")

    def test_withholds_sample_outside_alignment_window(self):
        rows = parse_telemetry_csv("time_seconds,speed\n1.0,123\n")
        result = correlate_telemetry_events([{"time_seconds": 2.0}], rows, 0.25)[0]

        self.assertEqual(result["alignment_status"], "withheld_outside_time_window")
        self.assertEqual(result["telemetry_values"], {})
