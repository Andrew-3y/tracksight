"""Reconstruct the observed development of one saved two-vehicle evidence event.

The helpers here deliberately explain only what the persisted detector records
show.  They never interpolate missing detections, estimate a collision, or
turn a proximity candidate into a race-control decision.
"""

from math import hypot


NET_TREND_TOLERANCE = 0.05


def _point_has_local_track_measurement(point):
    """Return whether a real trajectory point supports width-normalised distance."""
    position = point.get("track_position")
    width = point.get("track_width_units")
    return (
        isinstance(position, (list, tuple))
        and len(position) == 2
        and all(isinstance(value, (int, float)) for value in position)
        and isinstance(width, (int, float))
        and width > 0
    )


def _points_by_frame(trajectory):
    """Index one segment's saved real observations by source-video frame."""
    return {
        int(point["frame_index"]): point
        for point in trajectory
        if _point_has_local_track_measurement(point)
    }


def build_observed_approach_trace(
    trajectories,
    track_ids,
    start_frame,
    end_frame,
):
    """Return paired real local-track observations inside one saved event window.

    A frame appears only when both selected ID segments have real calibrated
    points at that exact source-video frame.  This makes a detection gap
    explicit rather than drawing a fictional continuous approach.
    """
    if len(track_ids) != 2:
        raise ValueError("Approach reconstruction requires exactly two vehicle IDs.")
    if int(end_frame) < int(start_frame):
        raise ValueError("Event end frame cannot precede its start frame.")

    track_id_a, track_id_b = (int(track_id) for track_id in track_ids)
    points_a = _points_by_frame(
        trajectories.get(str(track_id_a), trajectories.get(track_id_a, []))
    )
    points_b = _points_by_frame(
        trajectories.get(str(track_id_b), trajectories.get(track_id_b, []))
    )
    trace = []
    for frame_index in range(int(start_frame), int(end_frame) + 1):
        point_a = points_a.get(frame_index)
        point_b = points_b.get(frame_index)
        if point_a is None or point_b is None:
            continue
        position_a = point_a["track_position"]
        position_b = point_b["track_position"]
        average_width = (
            float(point_a["track_width_units"])
            + float(point_b["track_width_units"])
        ) / 2
        trace.append(
            {
                "frame": frame_index,
                "time_seconds": float(point_a["time_seconds"]),
                "separation_vehicle_widths": hypot(
                    float(position_b[0]) - float(position_a[0]),
                    float(position_b[1]) - float(position_a[1]),
                ) / average_width,
            }
        )
    return trace


def describe_observed_approach(trace):
    """Summarise a trace without treating a change in distance as an incident."""
    if not trace:
        return {
            "status": "withheld_without_paired_real_observations",
            "observation_count": 0,
            "missing_pair_frames": 0,
            "net_trend": "withheld",
        }

    start = trace[0]["separation_vehicle_widths"]
    end = trace[-1]["separation_vehicle_widths"]
    net_change = end - start
    if net_change < -NET_TREND_TOLERANCE:
        net_trend = "net_closing"
    elif net_change > NET_TREND_TOLERANCE:
        net_trend = "net_separating"
    else:
        net_trend = "no_material_net_change"
    closest = min(trace, key=lambda point: point["separation_vehicle_widths"])
    return {
        "status": "available",
        "observation_count": len(trace),
        "start_separation_vehicle_widths": start,
        "end_separation_vehicle_widths": end,
        "closest_separation_vehicle_widths": closest["separation_vehicle_widths"],
        "closest_frame": closest["frame"],
        "closest_time_seconds": closest["time_seconds"],
        "net_change_vehicle_widths": net_change,
        "net_trend": net_trend,
    }


def _source_window(event, evidence_type):
    """Normalize the two existing saved evidence schemas to one event window."""
    if evidence_type == "Predicted risk":
        return {
            "start_frame": event["start_frame"],
            "end_frame": event["end_frame"],
            "start_time_seconds": event["start_time_seconds"],
            "end_time_seconds": event["end_time_seconds"],
            "anchor_frame": event["highest_risk_frame"],
            "anchor_time_seconds": event["highest_risk_time_seconds"],
        }
    if evidence_type == "Observed proximity":
        return {
            "start_frame": event["start_frame"],
            "end_frame": event["end_frame"],
            "start_time_seconds": event["start_seconds"],
            "end_time_seconds": event["end_seconds"],
            "anchor_frame": event["closest_frame"],
            "anchor_time_seconds": event["closest_time_seconds"],
        }
    raise ValueError(f"Unsupported evidence type: {evidence_type}")


def build_approach_reconstruction(trajectories, event, evidence_type):
    """Return a factual timeline and paired-observation trace for one event."""
    window = _source_window(event, evidence_type)
    trace = build_observed_approach_trace(
        trajectories,
        event["track_ids"],
        window["start_frame"],
        window["end_frame"],
    )
    summary = describe_observed_approach(trace)
    source_frame_count = int(window["end_frame"]) - int(window["start_frame"]) + 1
    summary["missing_pair_frames"] = max(0, source_frame_count - len(trace))
    return {
        "track_ids": [int(track_id) for track_id in event["track_ids"]],
        "evidence_type": evidence_type,
        "source_window": window,
        "trace": trace,
        "summary": summary,
        "timeline": [
            {
                "moment": "Evidence window began",
                "frame": int(window["start_frame"]),
                "time_seconds": float(window["start_time_seconds"]),
                "basis": "first saved source-evidence frame",
            },
            {
                "moment": "Saved evidence anchor",
                "frame": int(window["anchor_frame"]),
                "time_seconds": float(window["anchor_time_seconds"]),
                "basis": "closest or predicted-closest source-evidence frame",
            },
            {
                "moment": "Evidence window ended",
                "frame": int(window["end_frame"]),
                "time_seconds": float(window["end_time_seconds"]),
                "basis": "last saved source-evidence frame",
            },
        ],
        "caveat": (
            "The distance trace uses only paired real calibrated observations. "
            "Missing pair frames are omitted; TrackSight does not interpolate "
            "them or infer contact, fault, or a rule breach."
        ),
    }
