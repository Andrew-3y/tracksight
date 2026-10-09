from src.tracking import get_box_center

def add_trajectory_points(trajectories, tracked_detections, frame_index, fps):
    for detection in tracked_detections:
        if detection["is_estimated"]:
            continue
        track_id = detection["track_id"]
        if track_id not in trajectories:
            trajectories[track_id] = []
        trajectory = trajectories[track_id]
        center = get_box_center(detection["xyxy"])
        time_seconds = frame_index / fps
        trajectory.append(
            {
                "frame_index": frame_index,
                "time_seconds": time_seconds,
                "center": center,
                "confidence": detection["confidence"]
            }
        )
    return trajectories