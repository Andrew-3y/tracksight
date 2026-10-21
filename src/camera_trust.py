"""Describe what a completed run can safely claim about its camera evidence."""


def build_camera_trust_monitor(run_summary):
    """Return transparent run-level camera and ID evidence gates.

    This does not invent a camera-confidence percentage. For moving-camera
    runs, aggregate stabilisation outcomes are shown only when the run saved
    them; earlier runs explicitly report that the diagnostic was unavailable.
    """
    provenance = run_summary.get("run_provenance") or {}
    spatial_enabled = bool(run_summary.get("spatial_analysis_enabled"))
    camera_mode = provenance.get("camera_mode", "unknown")
    diagnostics = run_summary.get("camera_trust_diagnostics")
    quality = run_summary.get("track_quality", {})
    trusted_segments = sum(
        item.get("trusted_for_safety_analysis", False)
        for item in quality.values()
    )
    all_segments = len(quality)
    checks = [
        {
            "check": "Camera calibration",
            "status": "Available" if spatial_enabled else "Unavailable",
            "detail": (
                "A camera-specific calibration enabled local-track evidence."
                if spatial_enabled
                else "This run has no usable calibration, so spatial safety evidence is withheld."
            ),
        },
        {
            "check": "Camera mode",
            "status": camera_mode.replace("_", " ").capitalize(),
            "detail": provenance.get("camera_motion_method", "No camera-motion method recorded."),
        },
        {
            "check": "ID continuity gate",
            "status": f"{trusted_segments} / {all_segments} trusted",
            "detail": "Only continuity-qualified real-observation segments may support safety evidence.",
        },
    ]
    stabilization = {
        "recorded": False,
        "status": "Not recorded",
        "detail": "This run did not save a moving-camera stabilization diagnostic.",
    }
    if camera_mode == "moving" and diagnostics is None:
        stabilization = {
            "recorded": False,
            "status": "Not recorded by this older run",
            "detail": "The saved pipeline records that stabilization was configured, but it did not persist per-run outcomes for this completed run.",
        }
    elif camera_mode == "moving" and diagnostics is not None:
        outcomes = diagnostics.get("frame_outcomes", {})
        updated = int(outcomes.get("transform_updated", 0))
        retained = sum(
            int(outcomes.get(key, 0))
            for key in (
                "retained_insufficient_features",
                "retained_insufficient_inliers",
                "retained_invalid_transform",
            )
        )
        stabilization = {
            "recorded": True,
            "status": "Recorded stabilization outcomes",
            "detail": f"Transform updated on {updated} frame(s); the last reliable transform was retained on {retained} frame(s).",
            "updated_frames": updated,
            "retained_frames": retained,
        }
    elif camera_mode == "fixed" and spatial_enabled:
        stabilization = {
            "recorded": True,
            "status": "Not required for fixed camera",
            "detail": "The calibration is applied directly to one fixed camera view.",
        }
    elif diagnostics is not None:
        method = diagnostics.get("method", "not recorded")
        stabilization = {
            "recorded": False,
            "status": "Not available",
            "detail": (
                "This run has no camera calibration, so no camera-motion correction was used."
                if method == "not_available_without_camera_calibration"
                else "The saved camera mode does not support a moving-camera stabilization claim."
            ),
        }
    checks.append(
        {
            "check": "Moving-camera stabilization",
            "status": stabilization["status"],
            "detail": stabilization["detail"],
        }
    )
    overall_status = (
        "Spatial review ready"
        if spatial_enabled and trusted_segments
        else "Camera-only review"
    )
    return {
        "overall_status": overall_status,
        "spatial_analysis_enabled": spatial_enabled,
        "trusted_segments": trusted_segments,
        "all_segments": all_segments,
        "stabilization": stabilization,
        "checks": checks,
    }
