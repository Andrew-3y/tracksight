"""Tests for the audit, integrity, coverage, and grounded-explainer additions."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.evidence_integrity import build_evidence_integrity, file_sha256
from src.grounded_assistant import build_grounded_case_answer
from src.human_review import empty_human_review_decisions, save_human_review_decision
from src.operational_coverage import (
    build_operational_coverage_timeline,
    summarise_operational_coverage,
)
from src.review_ledger import (
    append_second_opinion,
    build_decision_ledger,
    empty_second_opinions,
)


class AuditOperationsTests(unittest.TestCase):
    def test_human_review_keeps_prior_decision_versions(self):
        event = {"evidence_type": "Observed proximity", "track_ids": [7, 11], "frame": 98}
        decisions = save_human_review_decision(
            empty_human_review_decisions(), "observed:98:7-11", event,
            "Needs review", "Andre", "Inspect the saved replay.",
            reviewed_at="2026-10-20T10:00:00+00:00",
        )
        decisions = save_human_review_decision(
            decisions, "observed:98:7-11", event,
            "Escalate", "Andre", "Escalated after replay review.",
            reviewed_at="2026-10-20T10:05:00+00:00",
        )

        self.assertEqual(len(decisions["decisions"]["observed:98:7-11"]["history"]), 2)
        self.assertEqual(decisions["decisions"]["observed:98:7-11"]["history"][0]["outcome"], "Needs review")

    def test_independent_opinion_and_ledger_do_not_replace_primary_decision(self):
        primary = {
            "outcome": "Escalate",
            "reviewer_name": "Andre",
            "updated_at": "2026-10-20T10:00:00+00:00",
        }
        event = {
            "review_event_id": "observed:98:7-11",
            "evidence_type": "Observed proximity",
            "track_id_values": [7, 11],
            "frame": 98,
            "time_seconds": 4.08,
        }
        opinions = append_second_opinion(
            empty_second_opinions(), event, primary,
            "Disagree with recorded decision", "Sara", "The replay needs another camera angle.",
            recorded_at="2026-10-20T10:10:00+00:00",
        )
        ledger = build_decision_ledger(
            {"decisions": {event["review_event_id"]: {**primary, "event": {}, "review_note": "Escalated.", "history": []}}},
            {"actions": []}, {"corrections": []}, opinions,
        )

        self.assertEqual(opinions["opinions"][0]["recorded_decision_snapshot"]["outcome"], "Escalate")
        self.assertEqual(ledger[-1]["record_type"], "Independent second opinion")
        with self.assertRaisesRegex(ValueError, "different reviewer"):
            append_second_opinion(
                empty_second_opinions(), event, primary,
                "Agree with recorded decision", "Andre", "Not independent.",
            )

    def test_integrity_rows_are_path_free_and_hash_present_outputs(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            present = root / "present.json"
            present.write_text("saved evidence", encoding="utf-8")
            artifacts = {
                "annotated_video": present,
                "trajectory_json": present,
                "run_summary_json": present,
                "review_evidence_json": root / "review.json",
                "risk_evidence_json": root / "risk.json",
                "human_review_decisions_json": root / "reviewer.json",
                "race_control_action_log_json": root / "actions.json",
                "review_corrections_json": root / "corrections.json",
                "second_opinions_json": root / "opinions.json",
            }
            expected_hash = file_sha256(present)
            rows = build_evidence_integrity(artifacts, include_hashes=True)

        self.assertEqual(rows[0]["file_name"], "present.json")
        self.assertEqual(rows[0]["sha256"], expected_hash)
        self.assertNotIn(str(root), str(rows))

    def test_operational_coverage_marks_only_saved_trusted_pairs_as_eligible(self):
        rows = build_operational_coverage_timeline(
            {
                "1": [{"frame_index": 0, "confidence": 0.8}],
                "2": [{"frame_index": 0, "confidence": 0.9}],
                "3": [{"frame_index": 1, "confidence": 0.4}],
            },
            {
                "processed_frame_count": 3,
                "frames_per_second": 25.0,
                "spatial_analysis_enabled": True,
                "track_quality": {
                    "1": {"trusted_for_safety_analysis": True},
                    "2": {"trusted_for_safety_analysis": True},
                    "3": {"trusted_for_safety_analysis": False},
                },
            },
        )
        summary = summarise_operational_coverage(rows)

        self.assertEqual(rows[0]["operational_status"], "Pair review eligible")
        self.assertEqual(rows[1]["operational_status"], "No trusted segment")
        self.assertEqual(summary["pair_review_eligible_frames"], 1)

    def test_grounded_answer_only_reports_selected_case_material(self):
        answer = build_grounded_case_answer(
            "Why was this case surfaced?",
            {"track_ids": [7, 11], "start_time_seconds": 4.0, "end_time_seconds": 4.2},
            {"source_evidence_types": ["Observed proximity"], "status": "Monitor", "checks": [], "recommended_action": "Monitor the saved replay."},
            [], [{"event_key": "observed-0"}], [],
        )

        self.assertIn("Vehicles 7, 11", answer["answer"])
        self.assertIn("Monitor", answer["answer"])

    def test_grounded_availability_excludes_zero_available_sources(self):
        answer = build_grounded_case_answer(
            "What evidence is available?",
            {"track_ids": [7, 11], "start_time_seconds": 4.0, "end_time_seconds": 4.2},
            {"source_evidence_types": [], "status": "Monitor", "checks": [], "recommended_action": "Monitor."},
            [
                {"source": "Annotated primary camera", "status": "Available"},
                {"source": "Focused supporting clips", "status": "0 available"},
            ],
            [], [],
        )

        self.assertIn("Annotated primary camera", answer["answer"])
        self.assertNotIn("Focused supporting clips", answer["answer"])


if __name__ == "__main__":
    unittest.main()
