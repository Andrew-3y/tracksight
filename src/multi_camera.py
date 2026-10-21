"""Represent reviewer-configured multi-camera playback without false synchrony."""

from pathlib import Path
from math import isfinite


MULTI_CAMERA_DISCLAIMER = (
    "Additional camera offsets are reviewer-supplied synchronization settings. "
    "TrackSight does not infer cross-camera identity or synchrony from the videos."
)


def validate_multi_camera_manifest(manifest):
    """Validate a small portable manifest of optional supporting camera videos."""
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cameras"), list):
        raise ValueError("Multi-camera manifest must contain a cameras list.")
    if not manifest["cameras"]:
        raise ValueError("Multi-camera manifest must contain at least one supporting camera.")
    cameras = []
    ids = set()
    for index, camera in enumerate(manifest["cameras"], start=1):
        if not isinstance(camera, dict):
            raise ValueError(f"Camera {index} must be an object.")
        camera_id = str(camera.get("camera_id", "")).strip()
        label = str(camera.get("label", "")).strip()
        video_path = str(camera.get("video_path", "")).strip()
        if not camera_id or camera_id in ids or not label or not video_path:
            raise ValueError("Each camera needs a unique camera_id, label, and video_path.")
        try:
            offset = float(camera.get("video_time_offset_seconds", 0.0))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Camera {camera_id!r} has an invalid time offset.") from error
        if not isfinite(offset):
            raise ValueError(f"Camera {camera_id!r} has a non-finite time offset.")
        cameras.append({
            "camera_id": camera_id, "label": label, "video_path": video_path,
            "video_time_offset_seconds": offset,
        })
        ids.add(camera_id)
    return {"cameras": cameras}


def build_multi_camera_playback(event_time_seconds, manifest):
    """Return each configured camera's requested playback time and availability."""
    cameras = validate_multi_camera_manifest(manifest)["cameras"]
    return [{
        **camera,
        "playback_time_seconds": max(0.0, float(event_time_seconds) + camera["video_time_offset_seconds"]),
        "video_available": Path(camera["video_path"]).is_file(),
    } for camera in cameras]
