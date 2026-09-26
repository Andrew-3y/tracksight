from pathlib import Path
from ultralytics import YOLO
import cv2
from src.detect_vehicles import convert_boxes_to_full_frame, detect_vehicles, remove_duplicate_detections

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
fps = video_capture.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)
frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
print("Frame Count:", frame_count, "frames")
duration_seconds = frame_count / fps
print("Duration:", duration_seconds , "seconds")
validation_times_seconds = {
    "start" : 0,
    "middle" : duration_seconds/2,
    "end" : duration_seconds - 1
}
model_name = model_path.stem
validation_output_dir = project_root / "outputs" / "validation" / model_name
validation_output_dir.mkdir(parents=True, exist_ok=True)
for label, time_seconds in validation_times_seconds.items():
    frame_index = int(time_seconds * fps)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    frame_read_success, validation_frame = video_capture.read()
    if not frame_read_success:
        raise RuntimeError(f"Could not read the {label} validation frame.")
    frame_detections = []
    for region_name, (roi_left, roi_top, roi_right, roi_bottom) in track_regions.items():
        first_result, annotated_image = detect_vehicles(
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
        annotated_image_path = validation_output_dir / f"{label}_{region_name}_annotated.png"
        annotated_image_save = cv2.imwrite(str(annotated_image_path), annotated_image)
        if not annotated_image_save:
            raise RuntimeError(f"Could not save annotated image: {annotated_image_path}")
        detection_count = len(first_result.boxes)
        print(f"{label}, {region_name}: frame {frame_index}, {detection_count} detections")
    print(f"{label}: {len(frame_detections)} raw full-frame detections")
    unique_frame_detections = remove_duplicate_detections(frame_detections, iou_threshold=0.5)
    print(f"{label}: {len(unique_frame_detections)} unique full-frame detections")
    full_frame_annotated_image = validation_frame.copy()
    for detection in unique_frame_detections:
        x1, y1, x2, y2 = detection["xyxy"]
        cv2.rectangle(full_frame_annotated_image, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2)
    full_frame_image_path = validation_output_dir / f"{label}_full_frame_unique.png"   
    full_frame_image_save = cv2.imwrite(str(full_frame_image_path), full_frame_annotated_image)
    if not full_frame_image_save:
        raise RuntimeError(f"Could not save full-frame image to {full_frame_image_path}")

video_capture.release()
