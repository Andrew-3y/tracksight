import unittest

from src.review_corrections import (
    append_review_correction,
    build_correction_learning_export,
    empty_review_corrections,
)


class ReviewCorrectionsTests(unittest.TestCase):
    def test_appends_human_feedback_without_calling_it_a_label(self):
        corrections = append_review_correction(
            empty_review_corrections(),
            {
                "event_key": "observed-0",
                "evidence_type": "Observed proximity",
                "track_id_values": [7, 11],
                "frame": 98,
                "time_seconds": 4.0874,
            },
            "Merged vehicles",
            "Andre",
            "Independently annotate the two adjacent cars in this frame.",
            recorded_at="2026-10-20T00:00:00+00:00",
        )
        export = build_correction_learning_export("demo", corrections)

        self.assertEqual(export["correction_count"], 1)
        self.assertEqual(export["corrections"][0]["correction_type"], "Merged vehicles")
        self.assertIn("not machine-generated labels", export["disclaimer"])

    def test_rejects_incomplete_human_feedback(self):
        with self.assertRaisesRegex(ValueError, "Reviewer name"):
            append_review_correction(
                empty_review_corrections(),
                {"event_key": "x", "evidence_type": "Observed proximity", "track_id_values": [1, 2], "frame": 1, "time_seconds": 0.0},
                "False positive", "", "A note"
            )
