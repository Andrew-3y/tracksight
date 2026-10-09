import cv2
from src.detect_vehicles import detect_vehicles, convert_boxes_to_full_frame, remove_duplicate_detections, remove_shared_crop_edge_detections
from src.tracking import Tracker
from src.trajectory import add_trajectory_points

def process_video(
    model,
    video_path,
    output_video_path,
    track_regions,
    max_distance,
    max_missing_frames,
    duplicate_iou_threshold,
    crop_edge_margin,
    overlap_fraction_threshold
):
    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    video_codec = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_video_path), video_codec, fps, (frame_width, frame_height))
    if not video_writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video_path}")
    tracker = Tracker()
    is_first_frame = True
    trajectories = {}
    frame_index = 0
    while True:
        frame_read_success, frame = video_capture.read()
        if not frame_read_success:
            break
        frame_detections = []
        for region_name, current_region in track_regions.items():
            roi_left, roi_top, roi_right, roi_bottom = current_region
            first_result, _ = detect_vehicles(model, frame, roi_left, roi_top, roi_right, roi_bottom)
            full_frame_detections = convert_boxes_to_full_frame(first_result.boxes, roi_left, roi_top)
            for detection in full_frame_detections:
                detection["region_name"] = region_name
            frame_detections.extend(full_frame_detections)
        edge_checked_frame_detections = remove_shared_crop_edge_detections(frame_detections, track_regions, crop_edge_margin, overlap_fraction_threshold)
        unique_frame_detections = remove_duplicate_detections(edge_checked_frame_detections, iou_threshold=duplicate_iou_threshold)
        if is_first_frame:
            unique_frame_detections = sorted(unique_frame_detections, key=lambda detection: detection["xyxy"][1])
            is_first_frame = False
        tracked_detections = tracker.update(unique_frame_detections, max_distance, max_missing_frames)
        trajectories = add_trajectory_points(trajectories, tracked_detections, frame_index, fps)
        annotated_frame = frame.copy()
        for detection in tracked_detections:
            x1, y1, x2, y2 = detection["xyxy"]
            is_estimated = detection["is_estimated"]
            if is_estimated:
                annotation_color = (0, 255, 255)
            else:
                annotation_color = (0, 255, 0)
            cv2.rectangle(annotated_frame, (round(x1), round(y1)), (round(x2), round(y2)), annotation_color, 2)
            track_id = detection["track_id"]
            if is_estimated:
                label = f"ID {track_id} estimated"
            else:
                label = f"ID {track_id}"
            cv2.putText(annotated_frame, label, (round(x1), max(0, round(y1) - 10)), cv2.FONT_HERSHEY_COMPLEX, 0.6, annotation_color, 2)
        video_writer.write(annotated_frame)
        frame_index += 1
    video_capture.release()
    video_writer.release()
    return output_video_path, trajectories
