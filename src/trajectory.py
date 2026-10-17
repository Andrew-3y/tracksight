import json

from src.tracking import get_box_center
from src.calibration import project_vehicle_box, project_vehicle_width


def add_trajectory_points(
    trajectories,
    tracked_detections,
    frame_index,
    fps,
    calibration_homography=None,
):
    for detection in tracked_detections:
        if detection["is_estimated"] or not detection.get("is_confirmed", True):
            continue
        track_id = detection["track_id"]
        if track_id not in trajectories:
            trajectories[track_id] = []
        trajectory = trajectories[track_id]
        center = get_box_center(detection["xyxy"])
        x1, y1, x2, y2 = detection["xyxy"]
        time_seconds = frame_index / fps
        trajectory_point = {
            "frame_index": frame_index,
            "time_seconds": time_seconds,
            "center": center,
            "confidence": detection["confidence"],
            "box_width": x2 - x1,
            "box_height": y2 - y1,
        }
        if calibration_homography is not None:
            trajectory_point["track_position"] = project_vehicle_box(
                detection["xyxy"],
                calibration_homography,
            )
            trajectory_point["track_width_units"] = project_vehicle_width(
                detection["xyxy"],
                calibration_homography,
            )
        trajectory.append(trajectory_point)
    return trajectories


def save_trajectories(trajectories, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(trajectories, output_file, indent=2)
    return output_path
