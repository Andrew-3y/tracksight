"""Produce an explainable human-review recommendation for one evidence case."""


WITHHOLD = "Withhold"
MONITOR = "Monitor"
REVIEW_NOW = "Review now"
ESCALATED_BY_REVIEWER = "Escalated by reviewer"
CLEARED_BY_REVIEWER = "Cleared by reviewer"

DECISION_LADDER_DISCLAIMER = (
    "The decision ladder prioritises human attention from saved evidence quality and "
    "review status. It does not confirm contact, fault, a rule breach, or an incident."
)


def _case_review_status(case_rows):
    """Summarise only current human outcomes already overlaid on the case rows."""
    statuses = {str(row.get("review_status", "Unreviewed")) for row in case_rows}
    if "Escalate" in statuses:
        return ESCALATED_BY_REVIEWER
    if statuses and statuses == {"Cleared"}:
        return CLEARED_BY_REVIEWER
    return None


def build_safety_decision_ladder(case, run_summary, case_rows, annotated_video_available):
    """Return transparent review guidance without converting evidence into a verdict."""
    review_status = _case_review_status(case_rows)
    evidence_types = sorted({str(row["evidence_type"]) for row in case_rows})
    spatial_ready = bool(run_summary.get("spatial_analysis_enabled", False))
    continuity_ready = case.get("evidence_quality") == "trusted"
    video_ready = bool(annotated_video_available)

    if review_status is not None:
        status = review_status
        recommended_action = (
            "Keep the recorded human outcome with the evidence package; do not turn it into a fault finding."
            if status == CLEARED_BY_REVIEWER
            else "Continue the human escalation path using the saved video and evidence package."
        )
    elif not spatial_ready or not continuity_ready:
        status = WITHHOLD
        recommended_action = "Inspect the camera video only; do not use this case for a spatial safety recommendation."
    elif len(evidence_types) >= 2 or int(case.get("event_count", 0)) >= 2:
        status = REVIEW_NOW
        recommended_action = "Open the replay, inspect every source record, then record a human review decision."
    else:
        status = MONITOR
        recommended_action = "Monitor this single quality-gated evidence record or inspect its replay if operational context requires it."

    checks = [
        {
            "evidence_source": "Camera calibration",
            "status": "Available" if spatial_ready else "Unavailable",
            "detail": (
                "The run has calibrated local-track coordinates."
                if spatial_ready
                else "Spatial recommendations are withheld without a recorded camera calibration."
            ),
        },
        {
            "evidence_source": "ID continuity",
            "status": "Passed" if continuity_ready else "Withheld",
            "detail": (
                "Both vehicle segments passed the saved continuity-quality gate."
                if continuity_ready
                else "One or both vehicle segments did not pass the saved continuity-quality gate."
            ),
        },
        {
            "evidence_source": "Time-local evidence",
            "status": f"{len(case_rows)} record(s)",
            "detail": (
                "Sources: " + ", ".join(evidence_types)
                if evidence_types
                else "No source evidence record was available for this case."
            ),
        },
        {
            "evidence_source": "Annotated camera video",
            "status": "Available" if video_ready else "Unavailable",
            "detail": (
                "The recorded annotated camera video can be reviewed."
                if video_ready
                else "The annotated camera video is missing, so replay-dependent review is withheld."
            ),
        },
        {
            "evidence_source": "Human review",
            "status": review_status or "Pending",
            "detail": (
                "A current human decision is already associated with this evidence."
                if review_status
                else "No current human decision is recorded for this case."
            ),
        },
    ]
    return {
        "status": status,
        "recommended_action": recommended_action,
        "case_id": case["case_id"],
        "track_ids": list(case["track_ids"]),
        "source_evidence_types": evidence_types,
        "checks": checks,
    }
