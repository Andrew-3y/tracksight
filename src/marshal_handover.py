"""Create a portable, review-only handover from TrackSight's saved evidence."""

from collections import Counter

from src.review_experience import build_evidence_coverage


HANDOVER_DISCLAIMER = (
    "This handover organizes TrackSight evidence for the next human reviewer. "
    "It does not determine fault, contact, a rule violation, or an incident outcome."
)


def _review_status_counts(review_queue_rows):
    counts = Counter(row.get("review_status", "Unreviewed") for row in review_queue_rows)
    return {
        "unreviewed": counts["Unreviewed"],
        "needs_review": counts["Needs review"],
        "cleared": counts["Cleared"],
        "escalated": counts["Escalate"],
    }


def _next_items(review_queue_rows, limit=3):
    order = {"Escalate": 0, "Needs review": 1, "Unreviewed": 2, "Cleared": 3}
    unresolved = [
        row for row in review_queue_rows
        if row.get("review_status", "Unreviewed") != "Cleared"
    ]
    return [
        {
            "review_status": row.get("review_status", "Unreviewed"),
            "evidence_type": row["evidence_type"],
            "track_ids": list(row["track_id_values"]),
            "frame": int(row["frame"]),
            "time_seconds": float(row["time_seconds"]),
            "measurement_basis": row["measurement_basis"],
            "has_supporting_clip": bool(row.get("clip")),
        }
        for row in sorted(
            unresolved,
            key=lambda row: (
                order.get(row.get("review_status", "Unreviewed"), 2),
                float(row["time_seconds"]),
                str(row.get("event_key", "")),
            ),
        )[:limit]
    ]


def build_marshal_handover(run_summary, trajectories, review_queue_rows, safety_candidate_evidence=None):
    """Build a concise session handover without altering evidence records.

    ``review_queue_rows`` comes from the dashboard's current human-decision
    overlay, while every source measurement remains in the immutable run files.
    """
    coverage = build_evidence_coverage(run_summary, trajectories)
    candidate_evidence = safety_candidate_evidence or {}
    candidate_counts = {
        "low_displacement": len(candidate_evidence.get("stationary_vehicle_candidates", [])),
        "restricted_area": len(candidate_evidence.get("restricted_area_entry_candidates", [])),
        "stationary_on_active_racing_area": len(
            candidate_evidence.get("stationary_in_active_racing_area_candidates", [])
        ),
        "against_configured_direction": len(
            candidate_evidence.get("against_expected_direction_candidates", [])
        ),
    }
    duration = (
        float(run_summary.get("processed_frame_count", 0))
        / float(run_summary.get("frames_per_second", 0))
        if run_summary.get("frames_per_second")
        else 0.0
    )
    return {
        "handover_type": "tracksight_marshal_handover",
        "disclaimer": HANDOVER_DISCLAIMER,
        "session": {
            "duration_seconds": duration,
            "processed_frames": int(run_summary.get("processed_frame_count", 0)),
            "camera_mode": (run_summary.get("run_provenance") or {}).get("camera_mode", "unknown"),
            "spatial_analysis_status": run_summary.get("spatial_analysis_status", "unknown"),
        },
        "review_status": _review_status_counts(review_queue_rows),
        "coverage": coverage,
        "policy_candidate_counts": candidate_counts,
        "next_review_items": _next_items(review_queue_rows),
        "limitations": coverage["exclusion_reasons"],
    }
