"""Conservatively prioritise human review without making incident rulings."""


REVIEW_NOW = "Review now"
MONITOR = "Monitor"
INSUFFICIENT_EVIDENCE = "Insufficient evidence"
REVIEW_PRIORITY_LEVELS = (REVIEW_NOW, MONITOR, INSUFFICIENT_EVIDENCE)


def _trusted_pair(thread, run_summary):
    """Return whether both IDs satisfy the saved continuity-quality gate."""
    if not run_summary.get("spatial_analysis_enabled", False):
        return False
    qualities = run_summary.get("track_quality", {})
    return all(
        qualities.get(str(track_id), {}).get("trusted_for_safety_analysis", False)
        for track_id in thread["pair_track_ids"]
    )


def build_review_priority_case(thread, run_summary):
    """Turn one evidence thread into a transparent human-review priority.

    The priority expresses *review support*, never likelihood of fault, contact,
    or rule breach.  Different observed and predicted separation measures are
    deliberately never combined into a numerical risk score.
    """
    case = {
        "case_id": thread["thread_id"],
        "track_ids": list(thread["pair_track_ids"]),
        "track_ids_label": ", ".join(map(str, thread["pair_track_ids"])),
        "start_time_seconds": thread["earliest_time_seconds"],
        "end_time_seconds": thread["latest_time_seconds"],
        "event_count": thread["event_count"],
        "predicted_risk_event_count": thread["predicted_risk_event_count"],
        "observed_proximity_event_count": thread["observed_proximity_event_count"],
        "source_event_keys": list(thread["event_keys"]),
        "source_review_event_ids": list(thread["review_event_ids"]),
        "preferred_clip": thread.get("preferred_clip"),
        "preferred_event_key": thread.get("preferred_event_key"),
    }
    if not _trusted_pair(thread, run_summary):
        return {
            **case,
            "priority": INSUFFICIENT_EVIDENCE,
            "priority_order": 2,
            "evidence_quality": "insufficient",
            "reason": (
                "One or both ID segments do not meet the saved continuity and "
                "calibration gate for safety review."
            ),
            "recommended_human_action": "Inspect the camera video; do not use this case as safety evidence.",
        }
    if thread["event_count"] >= 2:
        return {
            **case,
            "priority": REVIEW_NOW,
            "priority_order": 0,
            "evidence_quality": "trusted",
            "reason": "Multiple time-local evidence records support review of the same vehicle pair.",
            "recommended_human_action": "Open the supporting video and record a human review outcome.",
        }
    return {
        **case,
        "priority": MONITOR,
        "priority_order": 1,
        "evidence_quality": "trusted",
        "reason": "One quality-gated evidence record is available for this vehicle pair.",
        "recommended_human_action": "Monitor or inspect the camera video if operational context requires it.",
    }


def build_review_priority_queue(threads, run_summary):
    """Return review cases in a stable, explainable action order."""
    cases = [build_review_priority_case(thread, run_summary) for thread in threads]
    return sorted(
        cases,
        key=lambda case: (
            case["priority_order"],
            -case["event_count"],
            case["start_time_seconds"],
            case["case_id"],
        ),
    )
