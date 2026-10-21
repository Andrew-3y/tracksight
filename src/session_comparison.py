"""Compare only compatible calibrated TrackSight sessions conservatively."""

from math import hypot
from statistics import median


def _profile_name(summary):
    return (summary.get("run_provenance") or {}).get("calibration_profile")


def build_session_snapshot(run_name, run_summary, event_locations, hotspots):
    """Capture path-free quantities that can be compared across compatible runs."""
    return {
        "run_name": run_name,
        "spatial_analysis_enabled": bool(run_summary.get("spatial_analysis_enabled")),
        "calibration_profile": _profile_name(run_summary),
        "processed_frame_count": int(run_summary.get("processed_frame_count", 0)),
        "frames_per_second": float(run_summary.get("frames_per_second", 0.0)),
        "trusted_track_segments": int(run_summary.get("trusted_track_segments", 0)),
        "observed_event_count": int(run_summary.get("observed_close_approach_event_count", 0)),
        "predicted_event_count": int(run_summary.get("risk_event_count", 0)),
        "event_locations": list(event_locations),
        "hotspots": list(hotspots),
    }


def validate_sessions_comparable(baseline, comparison):
    if not baseline["spatial_analysis_enabled"] or not comparison["spatial_analysis_enabled"]:
        return False, "Both runs need enabled camera calibration before local-track locations can be compared."
    if not baseline["calibration_profile"] or not comparison["calibration_profile"]:
        return False, "Both runs need a recorded calibration profile before comparison."
    if baseline["calibration_profile"] != comparison["calibration_profile"]:
        return False, "These runs use different camera calibrations, so a local-track comparison would be misleading."
    return True, "The runs share one recorded camera calibration and can be compared as local-track evidence."


def _duration_seconds(snapshot):
    fps = snapshot["frames_per_second"]
    return snapshot["processed_frame_count"] / fps if fps > 0 else 0.0


def build_session_comparison(baseline, comparison):
    comparable, message = validate_sessions_comparable(baseline, comparison)
    result = {
        "comparable": comparable,
        "message": message,
        "baseline_run": baseline["run_name"],
        "comparison_run": comparison["run_name"],
    }
    if not comparable:
        return result
    fields = (
        "trusted_track_segments",
        "observed_event_count",
        "predicted_event_count",
    )
    result["metrics"] = [
        {
            "metric": field.replace("_", " ").capitalize(),
            "baseline": baseline[field],
            "comparison": comparison[field],
            "difference": comparison[field] - baseline[field],
        }
        for field in fields
    ]
    result["duration_seconds"] = {
        "baseline": _duration_seconds(baseline),
        "comparison": _duration_seconds(comparison),
    }
    return result


def build_multi_run_risk_map(snapshots, neighborhood_vehicle_widths=6.0):
    """Return repeat local evidence locations only when every run is compatible."""
    if len(snapshots) < 2:
        return [], None, "Choose at least two calibrated runs to inspect recurring local evidence."
    baseline = snapshots[0]
    for snapshot in snapshots[1:]:
        comparable, message = validate_sessions_comparable(baseline, snapshot)
        if not comparable:
            return [], None, message
    locations = [
        {**location, "run_name": snapshot["run_name"]}
        for snapshot in snapshots
        for location in snapshot["event_locations"]
    ]
    if not locations:
        return [], None, "No quality-gated evidence locations were saved for the selected runs."
    if neighborhood_vehicle_widths <= 0:
        raise ValueError("The recurring-location neighborhood must be positive.")
    radius = median(item["average_vehicle_width"] for item in locations) * neighborhood_vehicle_widths
    groups = []
    for location in sorted(locations, key=lambda item: (item["evidence_type"], item["time_seconds"], item["event_key"])):
        candidates = []
        for group in groups:
            if group[0]["evidence_type"] != location["evidence_type"]:
                continue
            if all(hypot(location["track_x"] - other["track_x"], location["track_y"] - other["track_y"]) <= radius for other in group):
                candidates.append(group)
        if candidates:
            min(candidates, key=len).append(location)
        else:
            groups.append([location])
    recurring = []
    for group in groups:
        run_names = sorted({item["run_name"] for item in group})
        if len(run_names) < 2:
            continue
        recurring.append({
            "evidence_type": group[0]["evidence_type"],
            "run_count": len(run_names),
            "run_names": run_names,
            "event_count": len(group),
            "track_x": sum(item["track_x"] for item in group) / len(group),
            "track_y": sum(item["track_y"] for item in group) / len(group),
            "radius_track_units": radius,
        })
    return recurring, locations, "Recurring locations are display groups for human review, not hazardous-track findings."
