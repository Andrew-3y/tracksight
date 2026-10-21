"""Build a portable session-level race-control timeline report."""

from pathlib import Path


RACE_CONTROL_EXPORT_DISCLAIMER = (
    "This export preserves evidence and human-review status for handover. It is "
    "not an incident finding, penalty recommendation, or fault determination."
)


def _portable_clip_name(value):
    """Return a filename only for a real saved clip path, never a dataframe NaN."""
    return Path(value).name if isinstance(value, str) and value.strip() else None


def build_race_control_timeline_export(run_name, run_summary, timeline_rows):
    """Return a path-free chronological report of saved review material."""
    provenance = run_summary.get("run_provenance") or {}
    return {
        "export_type": "tracksight_race_control_timeline",
        "disclaimer": RACE_CONTROL_EXPORT_DISCLAIMER,
        "run_name": run_name,
        "source_video": Path(str(provenance.get("source_video", "unknown"))).name,
        "camera_mode": provenance.get("camera_mode", "unknown"),
        "timeline": [{
            "evidence_type": row["evidence_type"],
            "track_ids": list(row["track_id_values"]),
            "frame": int(row["frame"]),
            "time_seconds": float(row["time_seconds"]),
            "measurement_value": float(row["measurement_value"]),
            "measurement_basis": row["measurement_basis"],
            "review_status": row.get("review_status", "Unreviewed"),
            "supporting_clip": _portable_clip_name(row.get("clip")),
        } for row in sorted(timeline_rows, key=lambda row: (float(row["time_seconds"]), str(row.get("event_key", ""))))],
    }
