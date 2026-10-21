"""Create conservative, calibration-dependent safety-review candidates.

The functions in this module deliberately create *candidate evidence*, never
incident rulings. They use only consecutive, real trajectory observations from
ID segments that have already passed TrackSight's continuity-quality gate.
"""

import json
from math import hypot
from statistics import median
from pathlib import Path


SAFETY_CANDIDATE_EVIDENCE_SCHEMA_VERSION = 1
RESTRICTED_AREA_ENTRY_RULE = "restricted_area_entry"
STATIONARY_IN_ACTIVE_RACING_AREA_RULE = "stationary_in_active_racing_area"
AGAINST_EXPECTED_DIRECTION_RULE = "against_expected_direction"
DEFAULT_MINIMUM_STATIONARY_SECONDS = 1.0
DEFAULT_MAXIMUM_STATIONARY_SPREAD_VEHICLE_WIDTHS = 0.75
DEFAULT_MINIMUM_RESTRICTED_AREA_FRAMES = 3


def _point_has_calibrated_position(point):
    """Return whether one saved real observation has usable local coordinates."""
    position = point.get("track_position")
    width = point.get("track_width_units")
    return (
        isinstance(position, (list, tuple))
        and len(position) == 2
        and all(isinstance(value, (int, float)) for value in position)
        and isinstance(width, (int, float))
        and width > 0
    )


def _continuous_calibrated_spans(trajectory):
    """Split trajectory data whenever real calibrated observations are not adjacent."""
    spans = []
    active_span = []
    previous_frame = None
    for point in sorted(trajectory, key=lambda item: int(item["frame_index"])):
        consecutive = (
            previous_frame is not None
            and int(point["frame_index"]) == previous_frame + 1
        )
        if _point_has_calibrated_position(point) and (
            not active_span or consecutive
        ):
            active_span.append(point)
        else:
            if active_span:
                spans.append(active_span)
            active_span = [point] if _point_has_calibrated_position(point) else []
        previous_frame = int(point["frame_index"])
    if active_span:
        spans.append(active_span)
    return spans


def _maximum_position_spread(points):
    """Return the largest distance between any two local-track positions."""
    return max(
        hypot(
            point_b["track_position"][0] - point_a["track_position"][0],
            point_b["track_position"][1] - point_a["track_position"][1],
        )
        for index, point_a in enumerate(points)
        for point_b in points[index + 1 :]
    ) if len(points) > 1 else 0.0


def _build_stationary_candidate(track_id, points):
    """Describe a bounded, low-displacement real-observation interval."""
    reference_width = median(point["track_width_units"] for point in points)
    maximum_spread = _maximum_position_spread(points)
    first_point = points[0]
    last_point = points[-1]
    duration_seconds = last_point["time_seconds"] - first_point["time_seconds"]
    return {
        "candidate_type": "stationary_vehicle",
        "track_ids": [int(track_id)],
        "track_id": int(track_id),
        "start_frame": int(first_point["frame_index"]),
        "end_frame": int(last_point["frame_index"]),
        "anchor_frame": int(first_point["frame_index"]),
        "start_time_seconds": float(first_point["time_seconds"]),
        "end_time_seconds": float(last_point["time_seconds"]),
        "anchor_time_seconds": float(first_point["time_seconds"]),
        "frame_count": len(points),
        "duration_seconds": float(duration_seconds),
        "maximum_spread_track_units": float(maximum_spread),
        "reference_vehicle_width_track_units": float(reference_width),
        "maximum_spread_vehicle_widths": float(maximum_spread / reference_width),
        "measurement_basis": (
            "maximum span of consecutive real calibrated footpoints, "
            "normalized by the segment's median projected vehicle width"
        ),
        "review_reason": (
            f"Vehicle ID {int(track_id)} remained within "
            f"{maximum_spread / reference_width:.2f} projected vehicle widths "
            f"for {duration_seconds:.2f}s."
        ),
        "recommended_human_action": (
            "Review the supporting video before treating this low-displacement "
            "candidate as a stopped vehicle or operational hazard."
        ),
    }


def _build_policy_candidate(candidate_type, track_id, zone, points, **measurements):
    """Return a review-only candidate with its explicit configured policy."""
    first_point = points[0]
    last_point = points[-1]
    candidate = {
        "candidate_type": candidate_type,
        "track_ids": [int(track_id)],
        "track_id": int(track_id),
        "zone_id": zone["zone_id"],
        "zone_label": zone["label"],
        "zone_category": zone["category"],
        "zone_review_note": zone.get("review_note", ""),
        "start_frame": int(first_point["frame_index"]),
        "end_frame": int(last_point["frame_index"]),
        "anchor_frame": int(first_point["frame_index"]),
        "start_time_seconds": float(first_point["time_seconds"]),
        "end_time_seconds": float(last_point["time_seconds"]),
        "anchor_time_seconds": float(first_point["time_seconds"]),
        "frame_count": len(points),
        "duration_seconds": float(last_point["time_seconds"] - first_point["time_seconds"]),
    }
    candidate.update(measurements)
    return candidate


def _consecutive_points_inside_zone(trajectory, polygon):
    """Return contiguous real calibrated point spans whose footpoints are in one zone."""
    from src.safety_zones import is_point_in_polygon

    spans = []
    active = []
    previous_frame = None
    for point in sorted(trajectory, key=lambda item: int(item["frame_index"])):
        inside = _point_has_calibrated_position(point) and is_point_in_polygon(
            point["track_position"], polygon
        )
        consecutive = previous_frame is not None and int(point["frame_index"]) == previous_frame + 1
        if inside and (not active or consecutive):
            active.append(point)
        else:
            if active:
                spans.append(active)
            active = [point] if inside else []
        previous_frame = int(point["frame_index"])
    if active:
        spans.append(active)
    return spans


def find_stationary_in_active_racing_area_candidates(
    trajectories,
    zones,
    minimum_stationary_seconds=DEFAULT_MINIMUM_STATIONARY_SECONDS,
    maximum_spread_vehicle_widths=DEFAULT_MAXIMUM_STATIONARY_SPREAD_VEHICLE_WIDTHS,
):
    """Find low-displacement spans only in reviewer-defined active racing areas."""
    candidates = []
    for zone in zones:
        if zone.get("candidate_rule") != STATIONARY_IN_ACTIVE_RACING_AREA_RULE:
            continue
        for raw_track_id, trajectory in trajectories.items():
            track_id = int(raw_track_id)
            for span in _consecutive_points_inside_zone(trajectory, zone["polygon"]):
                for stationary in find_stationary_vehicle_candidates(
                    {track_id: span},
                    minimum_stationary_seconds,
                    maximum_spread_vehicle_widths,
                ):
                    stationary_points = [
                        point
                        for point in span
                        if stationary["start_frame"] <= point["frame_index"] <= stationary["end_frame"]
                    ]
                    candidates.append(
                        _build_policy_candidate(
                            "stationary_in_active_racing_area",
                            track_id,
                            zone,
                            stationary_points,
                            **{key: value for key, value in stationary.items() if key not in {
                                "candidate_type", "track_ids", "track_id", "start_frame", "end_frame",
                                "anchor_frame", "start_time_seconds", "end_time_seconds",
                                "anchor_time_seconds", "frame_count", "duration_seconds",
                                "measurement_basis", "review_reason", "recommended_human_action",
                            }},
                            measurement_basis=(
                                "maximum span of consecutive real calibrated footpoints inside a "
                                "reviewer-defined active-racing-area polygon, normalized by the "
                                "segment's median projected vehicle width"
                            ),
                            review_reason=(
                                f"Vehicle ID {track_id} remained within {stationary['maximum_spread_vehicle_widths']:.2f} "
                                f"projected vehicle widths inside '{zone['label']}' for "
                                f"{stationary['duration_seconds']:.2f}s."
                            ),
                            recommended_human_action=(
                                "Review the supporting video and the configured racing-area policy; "
                                "this is not an automatic stopped-vehicle or hazard finding."
                            ),
                        )
                    )
    return sorted(candidates, key=lambda item: (item["anchor_time_seconds"], item["track_id"], item["zone_id"]))


def find_against_expected_direction_candidates(trajectories, zones, minimum_consecutive_frames=3):
    """Find measured local motion opposed to a reviewer-configured corridor direction."""
    if minimum_consecutive_frames < 1:
        raise ValueError("Minimum against-direction frames must be at least one.")
    candidates = []
    for zone in zones:
        if zone.get("candidate_rule") != AGAINST_EXPECTED_DIRECTION_RULE:
            continue
        direction_x, direction_y = zone["expected_local_track_direction"]
        direction_length = hypot(direction_x, direction_y)
        unit_x, unit_y = direction_x / direction_length, direction_y / direction_length
        for raw_track_id, trajectory in trajectories.items():
            track_id = int(raw_track_id)
            active = []
            for span in _consecutive_points_inside_zone(trajectory, zone["polygon"]):
                for point in span:
                    motion = point.get("motion", {})
                    dot_product = (
                        motion.get("velocity_x_track_units_per_second", 0.0) * unit_x
                        + motion.get("velocity_y_track_units_per_second", 0.0) * unit_y
                    )
                    opposed = motion.get("status") == "measured" and dot_product < 0
                    if opposed:
                        active.append((point, dot_product))
                    else:
                        if len(active) >= minimum_consecutive_frames:
                            points = [item[0] for item in active]
                            candidates.append(_build_policy_candidate(
                                "against_expected_direction", track_id, zone, points,
                                mean_opposed_local_speed=float(sum(-item[1] for item in active) / len(active)),
                                expected_local_track_direction=[unit_x, unit_y],
                                measurement_basis=(
                                    "consecutive measured local-track velocity opposed to the "
                                    "reviewer-configured direction inside this corridor"
                                ),
                                review_reason=(
                                    f"Vehicle ID {track_id} had {len(active)} consecutive measured "
                                    f"observations opposed to the configured direction in '{zone['label']}'."
                                ),
                                recommended_human_action=(
                                    "Review the supporting video and the configured corridor direction; "
                                    "this is not an automatic wrong-way or rule-breach finding."
                                ),
                            ))
                        active = []
                if len(active) >= minimum_consecutive_frames:
                    points = [item[0] for item in active]
                    candidates.append(_build_policy_candidate(
                        "against_expected_direction", track_id, zone, points,
                        mean_opposed_local_speed=float(sum(-item[1] for item in active) / len(active)),
                        expected_local_track_direction=[unit_x, unit_y],
                        measurement_basis=("consecutive measured local-track velocity opposed to the reviewer-configured direction inside this corridor"),
                        review_reason=(f"Vehicle ID {track_id} had {len(active)} consecutive measured observations opposed to the configured direction in '{zone['label']}'."),
                        recommended_human_action=("Review the supporting video and the configured corridor direction; this is not an automatic wrong-way or rule-breach finding."),
                    ))
                active = []
    return sorted(candidates, key=lambda item: (item["anchor_time_seconds"], item["track_id"], item["zone_id"]))


def find_stationary_vehicle_candidates(
    trajectories,
    minimum_stationary_seconds=DEFAULT_MINIMUM_STATIONARY_SECONDS,
    maximum_spread_vehicle_widths=DEFAULT_MAXIMUM_STATIONARY_SPREAD_VEHICLE_WIDTHS,
):
    """Find sustained low-displacement intervals without assuming metres or km/h.

    The threshold is expressed in each vehicle segment's projected width. This
    avoids using a pixel or circuit-specific speed threshold, and a detection
    gap always ends the candidate rather than being interpolated.
    """
    if minimum_stationary_seconds <= 0:
        raise ValueError("Minimum stationary duration must be positive.")
    if maximum_spread_vehicle_widths <= 0:
        raise ValueError("Maximum stationary spread must be positive.")

    candidates = []
    for raw_track_id, trajectory in trajectories.items():
        track_id = int(raw_track_id)
        for span in _continuous_calibrated_spans(trajectory):
            active_points = []
            for point in span:
                proposed_points = active_points + [point]
                proposed_width = median(
                    item["track_width_units"] for item in proposed_points
                )
                proposed_spread = _maximum_position_spread(proposed_points)
                if (
                    proposed_spread
                    <= proposed_width * maximum_spread_vehicle_widths
                ):
                    active_points = proposed_points
                    continue

                if (
                    len(active_points) >= 3
                    and active_points[-1]["time_seconds"]
                    - active_points[0]["time_seconds"]
                    >= minimum_stationary_seconds
                ):
                    candidates.append(
                        _build_stationary_candidate(track_id, active_points)
                    )
                active_points = [point]
            if (
                len(active_points) >= 3
                and active_points[-1]["time_seconds"]
                - active_points[0]["time_seconds"]
                >= minimum_stationary_seconds
            ):
                candidates.append(_build_stationary_candidate(track_id, active_points))
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate["anchor_time_seconds"],
            candidate["track_id"],
            candidate["start_frame"],
        ),
    )


def find_restricted_area_entry_candidates(
    safety_zone_observations,
    zones,
    minimum_restricted_area_frames=DEFAULT_MINIMUM_RESTRICTED_AREA_FRAMES,
):
    """Return candidates only for reviewer-marked restricted-area polygons."""
    if minimum_restricted_area_frames < 1:
        raise ValueError("Minimum restricted-area observations must be at least one.")
    zone_rules = {
        zone["zone_id"]: zone.get("candidate_rule", "none")
        for zone in zones
    }
    candidates = []
    for observation in safety_zone_observations:
        if (
            zone_rules.get(observation["zone_id"]) != RESTRICTED_AREA_ENTRY_RULE
            or int(observation["frame_count"]) < minimum_restricted_area_frames
        ):
            continue
        track_id = int(observation["track_id"])
        candidates.append(
            {
                "candidate_type": "restricted_area_entry",
                "track_ids": [track_id],
                "track_id": track_id,
                "zone_id": observation["zone_id"],
                "zone_label": observation["zone_label"],
                "zone_category": observation["zone_category"],
                "zone_review_note": observation.get("zone_review_note", ""),
                "start_frame": int(observation["start_frame"]),
                "end_frame": int(observation["end_frame"]),
                "anchor_frame": int(observation["start_frame"]),
                "start_time_seconds": float(observation["start_seconds"]),
                "end_time_seconds": float(observation["end_seconds"]),
                "anchor_time_seconds": float(observation["start_seconds"]),
                "frame_count": int(observation["frame_count"]),
                "duration_seconds": float(
                    observation["end_seconds"] - observation["start_seconds"]
                ),
                "measurement_basis": (
                    "consecutive real calibrated footpoints inside a "
                    "reviewer-defined restricted-area polygon"
                ),
                "review_reason": (
                    f"Vehicle ID {track_id} was observed inside the reviewer-defined "
                    f"restricted area '{observation['zone_label']}' for "
                    f"{int(observation['frame_count'])} consecutive frame(s)."
                ),
                "recommended_human_action": (
                    "Review the supporting video and the configured zone policy; "
                    "this is not an automatic track-exit or rule-breach finding."
                ),
            }
        )
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate["anchor_time_seconds"],
            candidate["track_id"],
            candidate["zone_id"],
        ),
    )


def build_safety_candidate_evidence(
    trusted_trajectories,
    spatial_analysis_enabled,
    safety_zone_profile=None,
    safety_zone_observations=None,
    minimum_stationary_seconds=DEFAULT_MINIMUM_STATIONARY_SECONDS,
    maximum_spread_vehicle_widths=DEFAULT_MAXIMUM_STATIONARY_SPREAD_VEHICLE_WIDTHS,
    minimum_restricted_area_frames=DEFAULT_MINIMUM_RESTRICTED_AREA_FRAMES,
):
    """Build one self-contained, review-only artifact for new safety candidates."""
    if not spatial_analysis_enabled:
        return {
            "schema_version": SAFETY_CANDIDATE_EVIDENCE_SCHEMA_VERSION,
            "coordinate_system": "local_track_units",
            "analysis_status": "withheld_without_camera_calibration",
            "stationary_vehicle_candidates": [],
            "restricted_area_entry_candidates": [],
            "stationary_in_active_racing_area_candidates": [],
            "against_expected_direction_candidates": [],
        }

    zones = [] if safety_zone_profile is None else safety_zone_profile["zones"]
    observations = [] if safety_zone_observations is None else safety_zone_observations
    stationary_candidates = find_stationary_vehicle_candidates(
        trusted_trajectories,
        minimum_stationary_seconds,
        maximum_spread_vehicle_widths,
    )
    restricted_area_candidates = find_restricted_area_entry_candidates(
        observations,
        zones,
        minimum_restricted_area_frames,
    )
    active_racing_area_candidates = find_stationary_in_active_racing_area_candidates(
        trusted_trajectories, zones, minimum_stationary_seconds, maximum_spread_vehicle_widths
    )
    against_direction_candidates = find_against_expected_direction_candidates(
        trusted_trajectories, zones
    )
    return {
        "schema_version": SAFETY_CANDIDATE_EVIDENCE_SCHEMA_VERSION,
        "coordinate_system": "local_track_units",
        "analysis_status": "enabled",
        "stationary_parameters": {
            "minimum_stationary_seconds": minimum_stationary_seconds,
            "maximum_spread_vehicle_widths": maximum_spread_vehicle_widths,
        },
        "restricted_area_parameters": {
            "minimum_consecutive_frames": minimum_restricted_area_frames,
            "requires_reviewer_defined_zone_rule": RESTRICTED_AREA_ENTRY_RULE,
        },
        "stationary_vehicle_candidates": stationary_candidates,
        "restricted_area_entry_candidates": restricted_area_candidates,
        "stationary_in_active_racing_area_candidates": active_racing_area_candidates,
        "against_expected_direction_candidates": against_direction_candidates,
    }


def get_safety_candidates(safety_candidate_evidence):
    """Return all candidate records in stable source-video order."""
    if not isinstance(safety_candidate_evidence, dict):
        return []
    candidates = [
        *safety_candidate_evidence.get("stationary_vehicle_candidates", []),
        *safety_candidate_evidence.get("restricted_area_entry_candidates", []),
        *safety_candidate_evidence.get("stationary_in_active_racing_area_candidates", []),
        *safety_candidate_evidence.get("against_expected_direction_candidates", []),
    ]
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate["anchor_time_seconds"],
            candidate["candidate_type"],
            candidate["track_id"],
        ),
    )


def save_safety_candidate_evidence(evidence, output_path):
    """Save the portable candidate artifact beside the completed run."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(evidence, output_file, indent=2)
    return output_path
