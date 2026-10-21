import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.human_review import (
    HUMAN_REVIEW_SCHEMA_VERSION,
    build_human_review_report,
    empty_human_review_decisions,
    filter_current_human_review_decisions,
    is_current_human_review_decision,
    load_human_review_decisions,
    make_review_event_id,
    save_human_review_decision,
    write_human_review_decisions,
)


class HumanReviewTests(unittest.TestCase):
    def test_event_id_uses_sorted_vehicle_ids(self):
        self.assertEqual(
            make_review_event_id("Predicted risk", [7, 3], 42),
            "predicted_risk:42:3-7",
        )

    def test_decision_preserves_original_creation_time_when_updated(self):
        decision_set = save_human_review_decision(
            empty_human_review_decisions(),
            "predicted_risk:42:3-7",
            {"evidence_type": "Predicted risk", "track_ids": [3, 7]},
            "Needs review",
            "Andre",
            "Review the camera sequence before deciding.",
            reviewed_at="2026-10-18T01:00:00+00:00",
        )

        updated = save_human_review_decision(
            decision_set,
            "predicted_risk:42:3-7",
            {"evidence_type": "Predicted risk", "track_ids": [3, 7]},
            "Cleared",
            "Andre",
            "The full sequence shows normal racing proximity.",
            reviewed_at="2026-10-18T01:05:00+00:00",
        )

        decision = updated["decisions"]["predicted_risk:42:3-7"]
        self.assertEqual(decision["created_at"], "2026-10-18T01:00:00+00:00")
        self.assertEqual(decision["updated_at"], "2026-10-18T01:05:00+00:00")
        self.assertEqual(decision["outcome"], "Cleared")

    def test_decision_requires_reviewer_and_note(self):
        with self.assertRaisesRegex(ValueError, "Reviewer name"):
            save_human_review_decision(
                empty_human_review_decisions(),
                "observed_proximity:3:1-2",
                {},
                "Needs review",
                "",
                "A note",
            )
        with self.assertRaisesRegex(ValueError, "review note"):
            save_human_review_decision(
                empty_human_review_decisions(),
                "observed_proximity:3:1-2",
                {},
                "Needs review",
                "Andre",
                "",
            )

    def test_changed_evidence_does_not_reuse_a_saved_decision(self):
        decision = {
            "event": {"frame": 12, "measurement_value": 0.2},
            "outcome": "Cleared",
        }

        self.assertTrue(
            is_current_human_review_decision(
                decision,
                {"frame": 12, "measurement_value": 0.2},
            )
        )
        self.assertFalse(
            is_current_human_review_decision(
                decision,
                {"frame": 12, "measurement_value": 0.3},
            )
        )

    def test_saved_decisions_load_and_export_without_local_paths(self):
        with TemporaryDirectory() as temporary_directory:
            decision_path = Path(temporary_directory) / "decisions.json"
            decisions = save_human_review_decision(
                empty_human_review_decisions(),
                "observed_proximity:3:1-2",
                {
                    "evidence_type": "Observed proximity",
                    "track_ids": [1, 2],
                    "frame": 3,
                    "time_seconds": 0.12,
                },
                "Escalate",
                "Andre",
                "Check the full camera sequence.",
                reviewed_at="2026-10-18T01:00:00+00:00",
            )
            write_human_review_decisions(decisions, decision_path)

            loaded = load_human_review_decisions(decision_path)
            report = build_human_review_report("demo_run", loaded)

            self.assertEqual(loaded, decisions)
            self.assertEqual(report["run_name"], "demo_run")
            self.assertEqual(report["schema_version"], HUMAN_REVIEW_SCHEMA_VERSION)
            self.assertNotIn("C:\\", json.dumps(report))

    def test_only_decisions_matching_current_evidence_are_exported(self):
        decisions = save_human_review_decision(
            empty_human_review_decisions(),
            "observed_proximity:12:1-2",
            {"frame": 12, "measurement_value": 0.2},
            "Cleared",
            "A reviewer",
            "The saved frame was reviewed.",
            reviewed_at="2026-10-18T01:00:00+00:00",
        )

        current_decisions = filter_current_human_review_decisions(
            decisions,
            {
                "observed_proximity:12:1-2": {
                    "frame": 12,
                    "measurement_value": 0.3,
                }
            },
        )

        self.assertEqual(current_decisions, empty_human_review_decisions())


if __name__ == "__main__":
    unittest.main()
