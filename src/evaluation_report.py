"""Build a transparent per-run readiness and limitations report.

This module deliberately does not invent accuracy figures. Precision, recall,
mAP, and identity metrics require independent labelled ground truth.
"""

from pathlib import Path

from src.review_experience import build_evidence_coverage


EVALUATION_DISCLAIMER = (
    "This is a run-readiness and limitations report, not a detector or tracker "
    "accuracy benchmark. Independent labelled ground truth is required for "
    "precision, recall, mAP, and identity metrics."
)


def build_evaluation_report(run_summary, trajectories, safety_candidate_evidence=None):
    """Describe one completed run's context, outputs, safeguards, and boundaries."""
    provenance = run_summary.get("run_provenance") or {}
    coverage = build_evidence_coverage(run_summary, trajectories)
    candidates = safety_candidate_evidence or {}
    return {
        "report_type": "tracksight_run_readiness_and_limitations",
        "disclaimer": EVALUATION_DISCLAIMER,
        "evaluated_run_context": {
            "source_video": Path(str(provenance.get("source_video", "unknown"))).name,
            "camera_mode": provenance.get("camera_mode", "unknown"),
            "spatial_analysis_status": run_summary.get("spatial_analysis_status", "unknown"),
            "processed_frames": int(run_summary.get("processed_frame_count", 0)),
            "frames_per_second": float(run_summary.get("frames_per_second", 0.0)),
        },
        "recorded_outputs": {
            "detector_positions": int(run_summary.get("all_observations", 0)),
            "tracker_id_segments": int(run_summary.get("all_track_segments", 0)),
            "safety_eligible_segments": int(run_summary.get("trusted_track_segments", 0)),
            "observed_close_approach_events": int(run_summary.get("observed_close_approach_event_count", 0)),
            "predicted_close_approach_events": int(run_summary.get("risk_event_count", 0)),
            "policy_candidate_prompts": sum(
                len(candidates.get(key, []))
                for key in (
                    "stationary_vehicle_candidates",
                    "restricted_area_entry_candidates",
                    "stationary_in_active_racing_area_candidates",
                    "against_expected_direction_candidates",
                )
            ),
        },
        "coverage": coverage,
        "safeguards": [
            "Only real detector observations are saved into trajectories; estimated tracker boxes are excluded.",
            "Only ID segments that pass the saved continuous-observation gate are eligible for spatial safety evidence.",
            "Configured zones and directions are camera-specific local-track policy inputs, never inferred from another circuit.",
            "Candidate events require human review and are not incident, contact, fault, or rule-breach conclusions.",
        ],
        "known_limitations": coverage["exclusion_reasons"] + [
            "This single-run report does not prove performance on other videos, circuits, weather, camera positions, or traffic density.",
            "No accuracy or identity-continuity metric is reported because this run has not been compared with independently labelled ground truth.",
        ],
    }
