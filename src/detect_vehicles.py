from pathlib import Path
from ultralytics import YOLO
import cv2

def detect_vehicles(model, source_image, roi_left, roi_top, roi_right, roi_bottom):
    track_region = source_image[roi_top:roi_bottom, roi_left:roi_right]
    results = model(track_region)
    first_result = results[0]
    annotated_image = first_result.plot()
    return first_result, annotated_image

project_root = Path(__file__).resolve().parents[1]
model_path = project_root / "models" / "yolo26s.pt"
test_image_path = project_root / "outputs" / "frames" / "first_frame.png"
roi_left = 650 
roi_top = 400
roi_right = 1250
roi_bottom = 800
source_image = cv2.imread(str(test_image_path))
if source_image is None:
    raise Exception(f"Failed to read image from {test_image_path}")
model = YOLO(model_path)
first_result, annotated_image = detect_vehicles(model, source_image, roi_left, roi_top, roi_right, roi_bottom)
annotated_image_path = project_root / "outputs" / "detections" / "annotated_track_region.png"
annotated_image_saved = cv2.imwrite(str(annotated_image_path), annotated_image)
if not annotated_image_saved:
    raise Exception(f"Failed to save annotated image to {annotated_image_path}")