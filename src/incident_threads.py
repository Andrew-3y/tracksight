"""Build reviewer-facing incident threads from immutable evidence records.

An incident thread is a time-bounded *review case*, not a conclusion that an
incident occurred. It keeps related predicted and observed evidence together
only when those records concern the same vehicle pair.
"""


DEFAULT_THREAD_GAP_SECONDS = 1.5
DEFAULT_MAX_THREAD_DURATION_SECONDS = 6.0


def _normalise_event(event):
    """Copy the timeline fields required to build one auditable thread."""
    track_ids = tuple(sorted(int(track_id) for track_id in event["track_id_values"]))
    return {
        "event_key": str(event["event_key"]),
        "review_event_id": str(event["review_event_id"]),
        "evidence_type": str(event["evidence_type"]),
        "track_ids": track_ids,
        "frame": int(event["frame"]),
        "time_seconds": float(event["time_seconds"]),
        "measurement_value": float(event["measurement_value"]),
        "measurement_basis": str(event["measurement_basis"]),
        "clip": event.get("clip"),
    }


def _can_extend_thread(thread, event, gap_seconds, max_duration_seconds):
    """Keep only temporally local evidence for the exact same vehicle pair."""
    if event["time_seconds"] - thread["latest_time_seconds"] > gap_seconds:
        return False
    if event["time_seconds"] - thread["earliest_time_seconds"] > max_duration_seconds:
        return False
    return tuple(event["track_ids"]) == tuple(thread["pair_track_ids"])


def _build_thread(events):
    """Summarise a non-empty ordered evidence group into one review case."""
    event_keys = [event["event_key"] for event in events]
    track_ids = sorted({track_id for event in events for track_id in event["track_ids"]})
    event_counts = {
        "predicted_risk_event_count": sum(
            event["evidence_type"] == "Predicted risk" for event in events
        ),
        "observed_proximity_event_count": sum(
            event["evidence_type"] == "Observed proximity" for event in events
        ),
    }
    preferred_event = next(
        (event for event in events if isinstance(event["clip"], str)), events[0]
    )
    return {
        "thread_id": "thread-" + "-".join(event_keys),
        "event_keys": event_keys,
        "review_event_ids": [event["review_event_id"] for event in events],
        "track_ids": track_ids,
        "pair_track_ids": list(events[0]["track_ids"]),
        "track_ids_label": ", ".join(map(str, track_ids)),
        "earliest_time_seconds": events[0]["time_seconds"],
        "latest_time_seconds": events[-1]["time_seconds"],
        "start_frame": events[0]["frame"],
        "end_frame": events[-1]["frame"],
        "event_count": len(events),
        **event_counts,
        "preferred_event_key": preferred_event["event_key"],
        "preferred_clip": preferred_event["clip"],
        "events": events,
    }


def build_incident_threads(
    timeline_events,
    gap_seconds=DEFAULT_THREAD_GAP_SECONDS,
    max_duration_seconds=DEFAULT_MAX_THREAD_DURATION_SECONDS,
):
    """Group nearby, vehicle-linked evidence into conservative review cases.

    Records never merge merely because their timestamps are close: every new
    record must concern the same two vehicle IDs as the current case. A maximum
    duration also prevents repeated evidence from becoming one long case.
    """
    if gap_seconds <= 0:
        raise ValueError("The incident-thread gap must be greater than zero.")
    if max_duration_seconds < gap_seconds:
        raise ValueError("The maximum incident-thread duration must cover one gap.")

    ordered_events = sorted(
        (_normalise_event(event) for event in timeline_events),
        key=lambda event: (event["time_seconds"], event["frame"], event["event_key"]),
    )
    threads = []
    active_thread_events = []
    active_thread = None
    for event in ordered_events:
        if active_thread is None or not _can_extend_thread(
            active_thread,
            event,
            gap_seconds,
            max_duration_seconds,
        ):
            if active_thread_events:
                threads.append(_build_thread(active_thread_events))
            active_thread_events = [event]
            active_thread = _build_thread(active_thread_events)
            continue
        active_thread_events.append(event)
        active_thread = _build_thread(active_thread_events)
    if active_thread_events:
        threads.append(_build_thread(active_thread_events))
    return sorted(
        threads,
        key=lambda thread: (
            thread["earliest_time_seconds"],
            thread["start_frame"],
            thread["thread_id"],
        ),
    )
