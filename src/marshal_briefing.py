"""Build a plain-language, evidence-backed briefing for a human race reviewer."""

import json


HUMAN_REVIEW_DISCLAIMER = (
    "This briefing prioritizes saved evidence for human review. It does not "
    "determine fault, contact, a rule violation, or an incident outcome."
)
RECOMMENDED_HUMAN_ACTION = (
    "Review the supporting annotated clip and the full camera sequence before "
    "making any decision."
)
DEFAULT_MAX_MARSHAL_BRIEFING_HIGHLIGHTS = 5


def select_briefing_events(evidence, priority_key, max_highlights):
    """Select the closest evidence items for one measurement type."""
    if max_highlights < 1:
        raise ValueError("At least one briefing highlight must be allowed.")
    return sorted(evidence, key=lambda event: event[priority_key])[:max_highlights]


def build_predicted_risk_highlight(event):
    """Describe one predicted-risk event without converting it into a ruling."""
    track_id_a, track_id_b = event["track_ids"]
    separation = event["predicted_separation_vehicle_widths"]
    time_seconds = event["highest_risk_time_seconds"]
    return {
        "evidence_type": "predicted_close_approach",
        "track_ids": event["track_ids"],
        "frame": event["highest_risk_frame"],
        "time_seconds": time_seconds,
        "frame_count": event["frame_count"],
        "measurement": {
            "value": separation,
            "basis": "average projected vehicle widths",
        },
        "review_reason": (
            f"Track IDs {track_id_a} and {track_id_b} maintained a predicted "
            f"separation of {separation:.2f} average projected vehicle widths "
            f"at {time_seconds:.2f}s."
        ),
        "supporting_clip": event.get("marshal_briefing_clip_path"),
        "recommended_human_action": RECOMMENDED_HUMAN_ACTION,
    }


def build_observed_proximity_highlight(event):
    """Describe one observed-proximity event without treating it as contact."""
    track_id_a, track_id_b = event["track_ids"]
    separation = event["closest_normalized_distance"]
    time_seconds = event["closest_time_seconds"]
    return {
        "evidence_type": "observed_close_approach",
        "track_ids": event["track_ids"],
        "frame": event["closest_frame"],
        "time_seconds": time_seconds,
        "frame_count": event["frame_count"],
        "measurement": {
            "value": separation,
            "basis": "average detected-box diagonals",
        },
        "review_reason": (
            f"Track IDs {track_id_a} and {track_id_b} were observed at a "
            f"centre separation of {separation:.2f} average detected-box "
            f"diagonals at {time_seconds:.2f}s."
        ),
        "supporting_clip": event.get("marshal_briefing_clip_path"),
        "recommended_human_action": RECOMMENDED_HUMAN_ACTION,
    }


def build_marshal_briefing(
    run_summary,
    risk_evidence,
    review_evidence,
    max_highlights=DEFAULT_MAX_MARSHAL_BRIEFING_HIGHLIGHTS,
):
    """Create separate, comparable-within-type review highlights from one run."""
    predicted_risk_highlights = [
        build_predicted_risk_highlight(event)
        for event in select_briefing_events(
            risk_evidence,
            "predicted_separation_vehicle_widths",
            max_highlights,
        )
    ]
    observed_proximity_highlights = [
        build_observed_proximity_highlight(event)
        for event in select_briefing_events(
            review_evidence,
            "closest_normalized_distance",
            max_highlights,
        )
    ]
    provenance = run_summary.get("run_provenance", {})
    return {
        "briefing_type": "human_review",
        "disclaimer": HUMAN_REVIEW_DISCLAIMER,
        "run": {
            "source_video": provenance.get("source_video"),
            "duration_seconds": (
                run_summary["processed_frame_count"] / run_summary["frames_per_second"]
                if run_summary.get("frames_per_second")
                else 0.0
            ),
            "camera_mode": provenance.get("camera_mode", "unknown"),
            "spatial_analysis_status": run_summary.get(
                "spatial_analysis_status", "unknown"
            ),
        },
        "summary": {
            "candidate_risk_event_count": len(risk_evidence),
            "observed_proximity_event_count": len(review_evidence),
            "trusted_track_segments": run_summary["trusted_track_segments"],
        },
        "predicted_risk_highlights": predicted_risk_highlights,
        "observed_proximity_highlights": observed_proximity_highlights,
    }


def save_marshal_briefing(briefing, output_path):
    """Save the reproducible briefing beside the rest of a completed run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(briefing, output_file, indent=2)
    return output_path
