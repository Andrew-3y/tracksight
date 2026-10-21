import unittest

from src.decision_ladder import (
    CLEARED_BY_REVIEWER,
    ESCALATED_BY_REVIEWER,
    MONITOR,
    REVIEW_NOW,
    WITHHOLD,
    build_safety_decision_ladder,
)


def _case(**overrides):
    case = {
        "case_id": "case-1",
        "track_ids": [1, 2],
        "event_count": 1,
        "evidence_quality": "trusted",
    }
    case.update(overrides)
    return case


def _row(**overrides):
    row = {
        "event_key": "observed-0",
        "evidence_type": "Observed proximity",
        "review_status": "Unreviewed",
    }
    row.update(overrides)
    return row


class DecisionLadderTests(unittest.TestCase):
    def test_withholds_when_calibration_or_continuity_is_not_ready(self):
        ladder = build_safety_decision_ladder(
            _case(evidence_quality="insufficient"),
            {"spatial_analysis_enabled": False},
            [_row()],
            True,
        )

        self.assertEqual(ladder["status"], WITHHOLD)

    def test_multiple_independent_sources_require_human_review_now(self):
        ladder = build_safety_decision_ladder(
            _case(event_count=2),
            {"spatial_analysis_enabled": True},
            [_row(), _row(event_key="predicted-0", evidence_type="Predicted risk")],
            True,
        )

        self.assertEqual(ladder["status"], REVIEW_NOW)

    def test_single_quality_gated_record_is_monitor_only(self):
        ladder = build_safety_decision_ladder(
            _case(), {"spatial_analysis_enabled": True}, [_row()], True
        )

        self.assertEqual(ladder["status"], MONITOR)

    def test_human_status_overrides_automatic_review_priority(self):
        escalated = build_safety_decision_ladder(
            _case(), {"spatial_analysis_enabled": True}, [_row(review_status="Escalate")], True
        )
        cleared = build_safety_decision_ladder(
            _case(), {"spatial_analysis_enabled": True}, [_row(review_status="Cleared")], True
        )

        self.assertEqual(escalated["status"], ESCALATED_BY_REVIEWER)
        self.assertEqual(cleared["status"], CLEARED_BY_REVIEWER)
