from pathlib import Path
from ultralytics import YOLO
import cv2
from src.detect_vehicles import convert_boxes_to_full_frame, detect_vehicles, remove_duplicate_detections
from src.tracking import Tracker

tracker = Tracker()
max_distance = 50
max_missing_frames = 1

project_root = Path(__file__).resolve().parents[1]
video_path = project_root / "data" / "raw" / "racecars_pexels_3774642_1920x1080_25fps.mp4"  
model_path = project_root / "models" / "yolo26m.pt"
model = YOLO(model_path)
track_regions = {
    "centre": (650, 400, 1300, 800),
    "right": (1150, 400, 1920, 850),
}

video_capture = cv2.VideoCapture(str(video_path))
if not video_capture.isOpened():
    raise FileNotFoundError(f"Could not open video: {video_path}")
start_frame_index = 0
frames_to_process = 10
video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
model_name = model_path.stem
tracking_output_dir = (
    project_root / "outputs" / "validation" / model_name / "tracking"
)
tracking_output_dir.mkdir(parents=True, exist_ok=True)
for frame_offset in range(frames_to_process):
    frame_index = start_frame_index + frame_offset
    frame_read_success, validation_frame = video_capture.read()
    if not frame_read_success:
        raise RuntimeError(f"Could not read the frame {frame_index}.")
    frame_detections = []
    for region_name, (roi_left, roi_top, roi_right, roi_bottom) in track_regions.items():
        first_result, _ = detect_vehicles(
            model,
            validation_frame,
            roi_left,
            roi_top,
            roi_right,
            roi_bottom,
        )
        full_frame_detections = convert_boxes_to_full_frame(
            first_result.boxes,
            roi_left,
            roi_top
        )
        frame_detections.extend(full_frame_detections)
    unique_frame_detections = remove_duplicate_detections(frame_detections, iou_threshold=0.5)
    if frame_index == start_frame_index:
        unique_frame_detections = sorted(unique_frame_detections, key= lambda detection: detection["xyxy"][1])
    tracked_detections = tracker.update(unique_frame_detections,max_distance, max_missing_frames)
    full_frame_annotated_image = validation_frame.copy()
    for detection in tracked_detections:
        x1, y1, x2, y2 = detection["xyxy"]
        is_estimated = detection["is_estimated"]
        if is_estimated:
            annotation_color = (0, 255, 255)
        else:
            annotation_color = (0, 255, 0)
        cv2.rectangle(full_frame_annotated_image, (round(x1), round(y1)), (round(x2), round(y2)), annotation_color, 2)
        track_id = detection["track_id"]
        if is_estimated:
            label = f"ID {track_id} estimated"
        else:
            label = f"ID {track_id}"
        cv2.putText(full_frame_annotated_image, label, (round(x1), max(0, round(y1) - 10)), cv2.FONT_HERSHEY_COMPLEX, 0.6, annotation_color, 2)
    tracking_image_path = tracking_output_dir / f"tracking_frame_{frame_index:04d}.png"
    image_saved = cv2.imwrite(str(tracking_image_path), full_frame_annotated_image)
    if not image_saved:
        raise RuntimeError(f"Could not save tracking image: {tracking_image_path}")

video_capture.release()
