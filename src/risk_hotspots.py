"""Derive review-location groups from already saved calibrated evidence.

This module does not create a new safety score.  It only groups saved evidence
anchors that are close on one calibrated local-track view, so a reviewer can
prioritise locations with repeated candidate events.
"""

from math import hypot
from statistics import median


def _point_by_frame(trajectory):
    """Index a real saved trajectory by its source-video frame."""
    return {int(point["frame_index"]): point for point in trajectory}


def _get_calibrated_pair_anchor(event, points_by_track):
    """Return an event's midpoint and visual grouping scale, if available."""
    frame_index = int(event["frame"])
    vehicle_points = []
    for track_id in event["track_id_values"]:
        point = points_by_track.get(int(track_id), {}).get(frame_index)
        if point is None:
            return None
        track_position = point.get("track_position")
        track_width = point.get("track_width_units")
        if (
            not isinstance(track_position, (list, tuple))
            or len(track_position) != 2
            or track_width is None
            or float(track_width) <= 0
        ):
            return None
        vehicle_points.append((track_position, float(track_width)))
    if len(vehicle_points) != 2:
        return None
    return {
        "track_x": sum(point[0][0] for point in vehicle_points) / 2,
        "track_y": sum(point[0][1] for point in vehicle_points) / 2,
        "average_vehicle_width": sum(point[1] for point in vehicle_points) / 2,
    }


def build_calibrated_event_locations(events, trajectories):
    """Attach local-track anchors to evidence events with both positions saved.

    Events without two real projected positions are omitted.  This is an
    intentional evidence-quality gate: a dashboard hotspot cannot be more
    certain than the underlying calibrated observations.
    """
    points_by_track = {
        int(track_id): _point_by_frame(trajectory)
        for track_id, trajectory in trajectories.items()
    }
    locations = []
    for event in events:
        anchor = _get_calibrated_pair_anchor(event, points_by_track)
        if anchor is None:
            continue
        locations.append(
            {
                **event,
                **anchor,
                "track_ids": tuple(sorted(int(track_id) for track_id in event["track_id_values"])),
                "track_ids_label": ", ".join(
                    map(str, sorted(int(track_id) for track_id in event["track_id_values"]))
                ),
            }
        )
    return locations


def _complete_link_groups(locations, radius_track_units):
    """Build tight, deterministic groups without a transitive-chain shortcut.

    A connected-component approach could merge events A--B and B--C even if A
    and C are far apart.  For a reviewer-facing location that would overstate
    one long stretch of track as a single hotspot.  This complete-link rule
    adds an event only when it is within the selected neighborhood of every
    existing member of the group.
    """
    groups = []
    ordered_indices = sorted(
        range(len(locations)),
        key=lambda index: (
            locations[index]["evidence_type"],
            locations[index]["time_seconds"],
            locations[index]["event_key"],
        ),
    )
    for index in ordered_indices:
        location = locations[index]
        compatible_groups = []
        for group in groups:
            first_member = locations[group[0]]
            if first_member["evidence_type"] != location["evidence_type"]:
                continue
            distances = [
                hypot(
                    location["track_x"] - locations[member_index]["track_x"],
                    location["track_y"] - locations[member_index]["track_y"],
                )
                for member_index in group
            ]
            if all(distance <= radius_track_units for distance in distances):
                compatible_groups.append((sum(distances) / len(distances), group))
        if compatible_groups:
            _, selected_group = min(
                compatible_groups,
                key=lambda candidate: (candidate[0], candidate[1][0]),
            )
            selected_group.append(index)
        else:
            groups.append([index])
    return groups


def build_candidate_hotspots(
    locations,
    neighborhood_vehicle_widths=6.0,
    minimum_candidate_events=2,
):
    """Return repeated candidate-event groups using an adaptive local scale.

    The radius is expressed in median projected vehicle widths rather than a
    fixed number of pixels or metres.  It is a review-display setting, not a
    rule, risk threshold, or claim about track geometry.
    """
    if neighborhood_vehicle_widths <= 0:
        raise ValueError("The hotspot neighborhood must be positive.")
    if minimum_candidate_events < 2:
        raise ValueError("A hotspot must contain at least two candidate events.")
    if not locations:
        return [], None

    reference_vehicle_width = median(
        location["average_vehicle_width"] for location in locations
    )
    radius_track_units = reference_vehicle_width * neighborhood_vehicle_widths
    hotspots = []
    for component in _complete_link_groups(locations, radius_track_units):
        if len(component) < minimum_candidate_events:
            continue
        members = [locations[index] for index in component]
        member_track_ids = sorted(
            {
                track_id
                for member in members
                for track_id in member["track_ids"]
            }
        )
        hotspots.append(
            {
                "hotspot_id": "hotspot-" + "-".join(
                    sorted(member["event_key"] for member in members)
                ),
                "evidence_type": members[0]["evidence_type"],
                "candidate_event_count": len(members),
                "track_x": sum(member["track_x"] for member in members) / len(members),
                "track_y": sum(member["track_y"] for member in members) / len(members),
                "earliest_time_seconds": min(
                    member["time_seconds"] for member in members
                ),
                "latest_time_seconds": max(
                    member["time_seconds"] for member in members
                ),
                "vehicle_ids": member_track_ids,
                "vehicle_ids_label": ", ".join(map(str, member_track_ids)),
                "event_keys": [member["event_key"] for member in members],
                "radius_track_units": radius_track_units,
            }
        )
    return sorted(
        hotspots,
        key=lambda hotspot: (
            -hotspot["candidate_event_count"],
            hotspot["earliest_time_seconds"],
        ),
    ), radius_track_units
