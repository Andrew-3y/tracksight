"""Link one review case to every available evidence source without inferring data."""

from pathlib import Path


EVENT_CORRELATION_DISCLAIMER = (
    "This panel lists available evidence sources for the selected review case. "
    "An unavailable source is shown as unavailable; TrackSight never fabricates a correlation."
)


def build_event_correlation(
    case_rows,
    run_summary,
    annotated_video_path,
    telemetry_correlations=None,
    multi_camera_manifest=None,
):
    """Return an auditable source list for a selected time-bounded review case."""
    event_keys = {str(row["event_key"]) for row in case_rows}
    clip_count = sum(
        isinstance(row.get("clip"), str) and Path(row["clip"]).is_file()
        for row in case_rows
    )
    telemetry_rows = [
        row for row in (telemetry_correlations or [])
        if str(row.get("event_key")) in event_keys
    ]
    aligned_telemetry = sum(
        row.get("alignment_status") == "aligned" for row in telemetry_rows
    )
    cameras = (multi_camera_manifest or {}).get("cameras", [])
    available_supporting_cameras = sum(
        Path(str(camera.get("video_path", ""))).is_file() for camera in cameras
    )
    provenance = run_summary.get("run_provenance") or {}
    return [
        {
            "source": "Annotated primary camera",
            "status": "Available" if Path(annotated_video_path).is_file() else "Unavailable",
            "detail": "Full annotated video is available for replay." if Path(annotated_video_path).is_file() else "The saved annotated video file is missing.",
        },
        {
            "source": "Focused supporting clips",
            "status": f"{clip_count} available",
            "detail": "Saved clips are optional; the full annotated video remains the primary record.",
        },
        {
            "source": "Camera calibration",
            "status": "Available" if run_summary.get("spatial_analysis_enabled") else "Unavailable",
            "detail": provenance.get("calibration_profile", "No recorded camera calibration profile."),
        },
        {
            "source": "Optional telemetry",
            "status": f"{aligned_telemetry} aligned" if telemetry_rows else "Not attached",
            "detail": "Only telemetry within the configured timestamp window is shown; alignment is not causation or identity proof.",
        },
        {
            "source": "Supporting cameras",
            "status": f"{available_supporting_cameras} available" if cameras else "Not attached",
            "detail": "Supporting cameras use reviewer-supplied offsets and are context, not automatically inferred cross-camera identity.",
        },
    ]
