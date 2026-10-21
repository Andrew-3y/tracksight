"""Summarise exactly where a run had usable observations, without inventing confidence."""

from collections import defaultdict


OPERATIONAL_COVERAGE_DISCLAIMER = (
    "This timeline reports recorded detector observations and continuity gates. It is "
    "not an accuracy score, a probability of safety, or proof that unseen cars were absent."
)


def build_operational_coverage_timeline(trajectories, run_summary):
    """Return one frame row per processed frame using only saved observations."""
    total_frames = int(run_summary.get("processed_frame_count", 0))
    qualities = run_summary.get("track_quality", {})
    spatial_enabled = bool(run_summary.get("spatial_analysis_enabled", False))
    by_frame = defaultdict(list)
    for track_id, points in trajectories.items():
        trusted = bool(qualities.get(str(track_id), {}).get("trusted_for_safety_analysis"))
        for point in points:
            by_frame[int(point["frame_index"])].append(
                {"trusted": trusted, "confidence": float(point.get("confidence", 0.0))}
            )
    rows = []
    for frame in range(total_frames):
        observations = by_frame.get(frame, [])
        trusted_count = sum(item["trusted"] for item in observations)
        if not spatial_enabled:
            status = "Spatial review withheld"
        elif trusted_count >= 2:
            status = "Pair review eligible"
        elif trusted_count == 1:
            status = "One trusted segment"
        else:
            status = "No trusted segment"
        rows.append(
            {
                "frame": frame,
                "time_seconds": frame / float(run_summary.get("frames_per_second", 1.0) or 1.0),
                "detector_positions": len(observations),
                "trusted_positions": trusted_count,
                "untrusted_positions": len(observations) - trusted_count,
                "mean_detector_confidence": (
                    sum(item["confidence"] for item in observations) / len(observations)
                    if observations else None
                ),
                "operational_status": status,
            }
        )
    return rows


def summarise_operational_coverage(rows):
    """Return transparent coverage counts for the dashboard KPI row."""
    return {
        "processed_frames": len(rows),
        "frames_with_positions": sum(row["detector_positions"] > 0 for row in rows),
        "pair_review_eligible_frames": sum(row["operational_status"] == "Pair review eligible" for row in rows),
        "withheld_frames": sum(row["operational_status"] == "Spatial review withheld" for row in rows),
    }
