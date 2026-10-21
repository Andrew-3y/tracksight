"""Measure whether saved track segments are reliable enough for safety analysis."""

import json


def get_longest_consecutive_observation_run(trajectory):
    """Return the largest uninterrupted sequence of detector observations."""
    if not trajectory:
        return 0
    frame_indexes = sorted(point["frame_index"] for point in trajectory)
    longest_run = 1
    current_run = 1
    for previous_frame, current_frame in zip(frame_indexes, frame_indexes[1:]):
        if current_frame == previous_frame + 1:
            current_run += 1
        else:
            current_run = 1
        longest_run = max(longest_run, current_run)
    return longest_run


def build_track_quality(trajectories, minimum_consecutive_observations):
    """Describe each ID without pretending a segment is a unique physical car."""
    if minimum_consecutive_observations < 1:
        raise ValueError("Minimum consecutive observations must be at least one.")
    quality = {}
    for track_id, trajectory in trajectories.items():
        longest_run = get_longest_consecutive_observation_run(trajectory)
        quality[str(track_id)] = {
            "observation_count": len(trajectory),
            "longest_consecutive_observation_run": longest_run,
            "trusted_for_safety_analysis": (
                longest_run >= minimum_consecutive_observations
            ),
        }
    return quality


def filter_trusted_trajectories(trajectories, track_quality):
    """Keep only segments that have enough continuous real observations."""
    return {
        track_id: trajectory
        for track_id, trajectory in trajectories.items()
        if track_quality[str(track_id)]["trusted_for_safety_analysis"]
    }


def build_run_summary(
    trajectories,
    track_quality,
    risk_evidence,
    review_evidence,
    pipeline_version,
    spatial_analysis_enabled,
    processed_frame_count,
    frames_per_second,
    run_provenance=None,
    safety_zone_observation_count=0,
    camera_trust_diagnostics=None,
    safety_candidate_counts=None,
):
    """Create compact dashboard metadata from the full auditable output."""
    trusted_track_count = sum(
        item["trusted_for_safety_analysis"]
        for item in track_quality.values()
    )
    observation_count = sum(len(trajectory) for trajectory in trajectories.values())
    observations_per_frame = (
        observation_count / processed_frame_count
        if processed_frame_count
        else 0.0
    )
    summary = {
        "pipeline_version": pipeline_version,
        "spatial_analysis_enabled": spatial_analysis_enabled,
        "spatial_analysis_status": (
            "enabled_for_fixed_camera"
            if (
                spatial_analysis_enabled
                and (run_provenance or {}).get("camera_mode") == "fixed"
            )
            else (
                "enabled_with_camera_stabilization"
                if spatial_analysis_enabled
                else "disabled_without_camera_calibration"
            )
        ),
        "all_track_segments": len(trajectories),
        "trusted_track_segments": trusted_track_count,
        "all_observations": observation_count,
        "processed_frame_count": processed_frame_count,
        "frames_per_second": frames_per_second,
        "observations_per_frame": observations_per_frame,
        "risk_event_count": len(risk_evidence),
        "observed_close_approach_event_count": len(review_evidence),
        "safety_zone_observation_count": safety_zone_observation_count,
        "track_quality": track_quality,
    }
    if run_provenance is not None:
        summary["run_provenance"] = run_provenance
    if camera_trust_diagnostics is not None:
        summary["camera_trust_diagnostics"] = camera_trust_diagnostics
    if safety_candidate_counts is not None:
        summary["stationary_vehicle_candidate_count"] = int(
            safety_candidate_counts.get("stationary_vehicle", 0)
        )
        summary["restricted_area_entry_candidate_count"] = int(
            safety_candidate_counts.get("restricted_area_entry", 0)
        )
        summary["stationary_in_active_racing_area_candidate_count"] = int(
            safety_candidate_counts.get("stationary_in_active_racing_area", 0)
        )
        summary["against_expected_direction_candidate_count"] = int(
            safety_candidate_counts.get("against_expected_direction", 0)
        )
    return summary


def save_run_summary(summary, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
    return output_path
