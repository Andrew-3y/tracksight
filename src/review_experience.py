"""Build focused incident-replay and evidence-coverage views from saved evidence.

The helpers in this module deliberately use only persisted, real detector
observations.  They do not interpolate a path across a detection gap or create
an additional risk score.  That keeps the dashboard's explanation aligned with
the evidence already saved by the pipeline.
"""


def _safe_processed_frame_count(run_summary):
    """Return a non-negative processed-frame count from a saved summary."""
    return max(0, int(run_summary.get("processed_frame_count", 0)))


def _safe_track_ids(track_ids):
    """Normalize an event's vehicle IDs while preserving each ID once."""
    return sorted({int(track_id) for track_id in track_ids})


def build_incident_replay_points(
    trajectories,
    track_ids,
    event_time_seconds,
    event_frame,
    pre_seconds=2.0,
    post_seconds=1.0,
):
    """Return the two involved vehicles' real observations around one event.

    A ``replay_segment`` changes whenever an observation is missing for one or
    more video frames.  A chart can therefore draw each observed trail without
    falsely connecting a vehicle across an unobserved period.
    """
    if pre_seconds < 0 or post_seconds < 0:
        raise ValueError("Replay windows must be zero or greater.")

    selected_track_ids = _safe_track_ids(track_ids)
    start_time = float(event_time_seconds) - float(pre_seconds)
    end_time = float(event_time_seconds) + float(post_seconds)
    replay_points = []
    for track_id in selected_track_ids:
        points = sorted(
            trajectories.get(str(track_id), trajectories.get(track_id, [])),
            key=lambda point: point["frame_index"],
        )
        previous_frame = None
        segment_number = 0
        for point in points:
            time_seconds = float(point["time_seconds"])
            if not start_time <= time_seconds <= end_time:
                continue
            frame_index = int(point["frame_index"])
            if previous_frame is not None and frame_index != previous_frame + 1:
                segment_number += 1
            previous_frame = frame_index
            center = point.get("center")
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                continue
            track_position = point.get("track_position")
            replay_points.append(
                {
                    "track_id": track_id,
                    "frame": frame_index,
                    "time_seconds": time_seconds,
                    "camera_x": float(center[0]),
                    "camera_y": float(center[1]),
                    "track_x": (
                        None
                        if not isinstance(track_position, (list, tuple))
                        or len(track_position) != 2
                        else float(track_position[0])
                    ),
                    "track_y": (
                        None
                        if not isinstance(track_position, (list, tuple))
                        or len(track_position) != 2
                        else float(track_position[1])
                    ),
                    "confidence": float(point["confidence"]),
                    "replay_segment": f"{track_id}-{segment_number}",
                    "is_event_frame": frame_index == int(event_frame),
                }
            )
    return replay_points


def build_incident_replay_context_points(trajectories, track_ids, event_frame):
    """Return real projected vehicle positions at one saved evidence frame.

    The result is a traffic-context snapshot for an incident replay.  It uses
    only points stored at the exact evidence frame: no position is interpolated
    from an earlier or later frame.  The selected evidence pair is marked so a
    dashboard can highlight it without hiding the surrounding traffic.
    """
    selected_track_ids = set(_safe_track_ids(track_ids))
    context_points = []
    for raw_track_id, trajectory in trajectories.items():
        track_id = int(raw_track_id)
        for point in trajectory:
            if int(point["frame_index"]) != int(event_frame):
                continue
            track_position = point.get("track_position")
            if (
                not isinstance(track_position, (list, tuple))
                or len(track_position) != 2
            ):
                continue
            context_points.append(
                {
                    "track_id": track_id,
                    "track_id_label": f"ID {track_id}",
                    "track_x": float(track_position[0]),
                    "track_y": float(track_position[1]),
                    "is_selected": track_id in selected_track_ids,
                }
            )
            break
    return sorted(context_points, key=lambda point: (point["is_selected"], point["track_id"]))


def build_incident_replay_summary(replay_points, track_ids):
    """Describe whether a focused replay has observations for both vehicles."""
    selected_track_ids = _safe_track_ids(track_ids)
    observed_track_ids = sorted({point["track_id"] for point in replay_points})
    missing_track_ids = sorted(set(selected_track_ids).difference(observed_track_ids))
    replay_segments = {point["replay_segment"] for point in replay_points}
    return {
        "selected_track_ids": selected_track_ids,
        "observed_track_ids": observed_track_ids,
        "missing_track_ids": missing_track_ids,
        "observation_count": len(replay_points),
        "contains_observation_gaps": len(replay_segments) > len(observed_track_ids),
        "has_calibrated_positions": bool(replay_points) and all(
            point["track_x"] is not None and point["track_y"] is not None
            for point in replay_points
        ),
    }


def build_evidence_coverage(run_summary, trajectories):
    """Summarize what a completed run could legitimately use as evidence.

    The returned frame ratio is *not* an accuracy score.  It only measures the
    proportion of processed frames containing at least one real observation
    from a segment that passed TrackSight's continuity gate.
    """
    processed_frame_count = _safe_processed_frame_count(run_summary)
    track_quality = run_summary.get("track_quality", {})
    trusted_track_ids = {
        int(track_id)
        for track_id, quality in track_quality.items()
        if quality.get("trusted_for_safety_analysis", False)
    }
    all_observation_frames = set()
    trusted_observation_frames = set()
    projected_trusted_frames = set()
    trusted_observations = 0
    projected_trusted_observations = 0

    for raw_track_id, trajectory in trajectories.items():
        track_id = int(raw_track_id)
        for point in trajectory:
            frame_index = int(point["frame_index"])
            all_observation_frames.add(frame_index)
            if track_id not in trusted_track_ids:
                continue
            trusted_observations += 1
            trusted_observation_frames.add(frame_index)
            track_position = point.get("track_position")
            if isinstance(track_position, (list, tuple)) and len(track_position) == 2:
                projected_trusted_observations += 1
                projected_trusted_frames.add(frame_index)

    all_track_segments = int(run_summary.get("all_track_segments", len(trajectories)))
    trusted_track_segments = len(trusted_track_ids)
    untrusted_track_segments = max(0, all_track_segments - trusted_track_segments)

    def ratio(numerator):
        return numerator / processed_frame_count if processed_frame_count else 0.0

    spatial_analysis_enabled = bool(run_summary.get("spatial_analysis_enabled", False))
    reasons = []
    if not spatial_analysis_enabled:
        reasons.append(
            "Spatial safety evidence was withheld because this completed run has no usable camera calibration."
        )
    if untrusted_track_segments:
        reasons.append(
            f"{untrusted_track_segments} ID segment(s) did not meet the continuous real-observation requirement and were excluded from safety analysis."
        )
    if not trusted_track_segments:
        reasons.append(
            "No ID segment passed the continuity gate, so TrackSight did not create safety evidence from this run."
        )
    elif not projected_trusted_observations and spatial_analysis_enabled:
        reasons.append(
            "No trusted observation could be projected into the calibrated local track view, so spatial safety evidence was withheld."
        )
    reasons.append(
        "Estimated tracker positions are intentionally excluded from saved trajectories and safety evidence."
    )

    return {
        "processed_frame_count": processed_frame_count,
        "all_observation_frames": len(all_observation_frames),
        "trusted_observation_frames": len(trusted_observation_frames),
        "projected_trusted_frames": len(projected_trusted_frames),
        "trusted_observations": trusted_observations,
        "projected_trusted_observations": projected_trusted_observations,
        "all_track_segments": all_track_segments,
        "trusted_track_segments": trusted_track_segments,
        "untrusted_track_segments": untrusted_track_segments,
        "trusted_segment_ratio": (
            trusted_track_segments / all_track_segments if all_track_segments else 0.0
        ),
        "trusted_frame_ratio": ratio(len(trusted_observation_frames)),
        "projected_trusted_frame_ratio": ratio(len(projected_trusted_frames)),
        "spatial_analysis_enabled": spatial_analysis_enabled,
        "exclusion_reasons": reasons,
    }
