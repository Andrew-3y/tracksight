"""Build a compact, auditable evidence chain for one review case."""


def build_evidence_chain(case, source_events, run_summary, annotated_video_exists):
    """Return ordered evidence stages without exposing machine paths or conclusions."""
    qualities = run_summary.get("track_quality", {})
    pair_trusted = all(
        qualities.get(str(track_id), {}).get("trusted_for_safety_analysis", False)
        for track_id in case["track_ids"]
    )
    clip_available = bool(case.get("preferred_clip"))
    stages = [
        {
            "order": 1,
            "stage": "Source camera video",
            "status": "Available" if annotated_video_exists else "Missing",
            "detail": "Annotated camera output is retained for human inspection." if annotated_video_exists else "The required annotated camera output is missing.",
        },
        {
            "order": 2,
            "stage": "Camera calibration",
            "status": "Available" if run_summary.get("spatial_analysis_enabled") else "Withheld",
            "detail": "Local-track evidence is enabled only for this calibrated camera view." if run_summary.get("spatial_analysis_enabled") else "No local-track claim is made without calibration.",
        },
        {
            "order": 3,
            "stage": "Vehicle-ID continuity",
            "status": "Passed" if pair_trusted else "Withheld",
            "detail": "Both involved ID segments passed the saved real-observation continuity gate." if pair_trusted else "At least one involved ID segment did not pass the continuity gate.",
        },
        {
            "order": 4,
            "stage": "Saved evidence records",
            "status": f"{len(source_events)} record(s)",
            "detail": "Observed and predicted records retain their distinct measurement bases.",
        },
        {
            "order": 5,
            "stage": "Supporting replay",
            "status": "Clip available" if clip_available else "Full video fallback",
            "detail": "A focused annotated clip was saved." if clip_available else "The full annotated camera video remains the review source.",
        },
        {
            "order": 6,
            "stage": "Human race-control action",
            "status": "Required",
            "detail": "TrackSight prioritises evidence; a named human records any operational action.",
        },
    ]
    return stages
