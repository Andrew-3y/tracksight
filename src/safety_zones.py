"""Read and evaluate explicit local-track safety zones.

Zones are camera-calibration-specific polygons in local track units. They are
not inferred by the model, so TrackSight never silently applies a zone from
one camera to another.
"""

import json
from pathlib import Path


SAFETY_ZONE_PROFILE_SCHEMA_VERSION = 1
SAFETY_ZONE_EVIDENCE_SCHEMA_VERSION = 1
SAFETY_ZONE_CANDIDATE_RULES = {
    "none",
    "restricted_area_entry",
    "stationary_in_active_racing_area",
    "against_expected_direction",
}


def _as_point(raw_point, description):
    if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
        raise ValueError(f"{description} must be an [x, y] coordinate pair.")
    try:
        return [float(raw_point[0]), float(raw_point[1])]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} coordinates must be numbers.") from error


def validate_safety_zone_profile(profile):
    """Return a safe normalised profile or reject ambiguous zone definitions."""
    if not isinstance(profile, dict):
        raise ValueError("The safety-zone profile must be a JSON object.")
    if profile.get("schema_version") != SAFETY_ZONE_PROFILE_SCHEMA_VERSION:
        raise ValueError("The safety-zone profile uses an unsupported schema version.")
    if profile.get("coordinate_system") != "local_track_units":
        raise ValueError("Safety zones must use the local_track_units coordinate system.")
    raw_zones = profile.get("zones")
    if not isinstance(raw_zones, list) or not raw_zones:
        raise ValueError("The safety-zone profile must contain at least one zone.")

    normalised_zones = []
    used_zone_ids = set()
    for index, raw_zone in enumerate(raw_zones, start=1):
        if not isinstance(raw_zone, dict):
            raise ValueError(f"Zone {index} must be a JSON object.")
        zone_id = raw_zone.get("zone_id")
        label = raw_zone.get("label")
        if not isinstance(zone_id, str) or not zone_id.strip():
            raise ValueError(f"Zone {index} needs a non-empty zone_id.")
        if zone_id in used_zone_ids:
            raise ValueError(f"Safety-zone ID {zone_id!r} is duplicated.")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Zone {zone_id!r} needs a human-readable label.")
        raw_polygon = raw_zone.get("polygon")
        if not isinstance(raw_polygon, list) or len(raw_polygon) < 3:
            raise ValueError(f"Zone {zone_id!r} needs at least three polygon points.")
        candidate_rule = raw_zone.get("candidate_rule", "none")
        if candidate_rule not in SAFETY_ZONE_CANDIDATE_RULES:
            raise ValueError(
                f"Zone {zone_id!r} has an unsupported candidate_rule. "
                "Use 'none', 'restricted_area_entry', "
                "'stationary_in_active_racing_area', or "
                "'against_expected_direction'."
            )
        expected_direction = raw_zone.get("expected_local_track_direction")
        if candidate_rule == "against_expected_direction":
            expected_direction = _as_point(
                expected_direction,
                f"Zone {zone_id!r} expected_local_track_direction",
            )
            if expected_direction == [0.0, 0.0]:
                raise ValueError(
                    f"Zone {zone_id!r} expected_local_track_direction must not be zero."
                )
        elif expected_direction is not None:
            raise ValueError(
                f"Zone {zone_id!r} only supports expected_local_track_direction "
                "with candidate_rule 'against_expected_direction'."
            )
        normalised_zones.append(
            {
                "zone_id": zone_id.strip(),
                "label": label.strip(),
                "category": str(raw_zone.get("category", "review area")).strip(),
                "polygon": [
                    _as_point(point, f"Zone {zone_id!r} polygon point {point_index}")
                    for point_index, point in enumerate(raw_polygon, start=1)
                ],
                "review_note": str(raw_zone.get("review_note", "")).strip(),
                "candidate_rule": candidate_rule,
                "expected_local_track_direction": expected_direction,
            }
        )
        used_zone_ids.add(zone_id)
    return {
        "schema_version": SAFETY_ZONE_PROFILE_SCHEMA_VERSION,
        "coordinate_system": "local_track_units",
        "zones": normalised_zones,
    }


def load_safety_zone_profile(config_path):
    """Load one explicit safety-zone profile from a JSON file."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Could not find safety-zone profile: {config_path}")
    with config_path.open(encoding="utf-8") as profile_file:
        return validate_safety_zone_profile(json.load(profile_file))


def is_point_in_polygon(point, polygon):
    """Return True for a point inside or on the boundary of a polygon."""
    x, y = _as_point(point, "Track position")
    vertices = [_as_point(vertex, "Zone polygon point") for vertex in polygon]
    inside = False
    for current, following in zip(vertices, vertices[1:] + vertices[:1]):
        x1, y1 = current
        x2, y2 = following
        cross_product = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross_product) < 1e-9 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True
        crosses_horizontal_ray = (y1 > y) != (y2 > y)
        if crosses_horizontal_ray:
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersection_x:
                inside = not inside
    return inside


def _build_observation(zone, track_id, points):
    """Describe one uninterrupted real-observation interval inside one zone."""
    first_point = points[0]
    last_point = points[-1]
    track_position = first_point["track_position"]
    return {
        "zone_id": zone["zone_id"],
        "zone_label": zone["label"],
        "zone_category": zone["category"],
        "zone_review_note": zone["review_note"],
        "zone_candidate_rule": zone["candidate_rule"],
        "track_id": int(track_id),
        "start_frame": int(first_point["frame_index"]),
        "end_frame": int(last_point["frame_index"]),
        "start_seconds": float(first_point["time_seconds"]),
        "end_seconds": float(last_point["time_seconds"]),
        "frame_count": len(points),
        "entry_track_x": float(track_position[0]),
        "entry_track_y": float(track_position[1]),
    }


def build_safety_zone_observations(trajectories, zones, trusted_track_ids=None):
    """Record contiguous, real vehicle observations within configured zones.

    Missing detections are not filled in. A lost-and-reacquired vehicle therefore
    produces separate intervals, which preserves the limits of the evidence.
    """
    normalised_zones = validate_safety_zone_profile(
        {
            "schema_version": SAFETY_ZONE_PROFILE_SCHEMA_VERSION,
            "coordinate_system": "local_track_units",
            "zones": zones,
        }
    )["zones"]
    trusted_track_ids = (
        None if trusted_track_ids is None else {int(track_id) for track_id in trusted_track_ids}
    )
    observations = []
    for raw_track_id, trajectory in trajectories.items():
        track_id = int(raw_track_id)
        if trusted_track_ids is not None and track_id not in trusted_track_ids:
            continue
        ordered_points = sorted(trajectory, key=lambda point: int(point["frame_index"]))
        for zone in normalised_zones:
            active_points = []
            previous_frame = None
            for point in ordered_points:
                track_position = point.get("track_position")
                is_inside = (
                    isinstance(track_position, (list, tuple))
                    and len(track_position) == 2
                    and is_point_in_polygon(track_position, zone["polygon"])
                )
                is_consecutive = (
                    previous_frame is not None
                    and int(point["frame_index"]) == previous_frame + 1
                )
                if is_inside and (not active_points or is_consecutive):
                    active_points.append(point)
                else:
                    if active_points:
                        observations.append(_build_observation(zone, track_id, active_points))
                    active_points = [point] if is_inside else []
                previous_frame = int(point["frame_index"])
            if active_points:
                observations.append(_build_observation(zone, track_id, active_points))
    return sorted(
        observations,
        key=lambda event: (event["start_seconds"], event["track_id"], event["zone_id"]),
    )


def build_safety_zone_evidence(profile, observations, profile_path=None):
    """Create a portable saved artifact for a completed TrackSight run."""
    normalised_profile = validate_safety_zone_profile(profile)
    return {
        "schema_version": SAFETY_ZONE_EVIDENCE_SCHEMA_VERSION,
        "coordinate_system": "local_track_units",
        "profile_file": None if profile_path is None else Path(profile_path).name,
        "zones": normalised_profile["zones"],
        "observations": observations,
    }


def save_safety_zone_evidence(evidence, output_path):
    """Save a self-contained zone profile snapshot with its observations."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as evidence_file:
        json.dump(evidence, evidence_file, indent=2)
    return output_path
