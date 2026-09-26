from pathlib import Path
from ultralytics import YOLO
import cv2

def detect_vehicles(model, source_image, roi_left, roi_top, roi_right, roi_bottom):
    track_region = source_image[roi_top:roi_bottom, roi_left:roi_right]
    results = model(track_region, classes=[2])
    first_result = results[0]
    annotated_image = first_result.plot()
    return first_result, annotated_image

def convert_boxes_to_full_frame(boxes, roi_left, roi_top):
    full_frame_detections = []
    for x1, y1, x2, y2, confidence, class_id in boxes.data.cpu().tolist():
        full_frame_detections.append(
            {
                "xyxy": (
                    x1 + roi_left,
                    y1 + roi_top,
                    x2 + roi_left,
                    y2 + roi_top,
                ),
                "confidence": confidence,
                "class_id": int(class_id),
            }
        )
    return full_frame_detections

def calculate_iou(box_a, box_b):
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b
    intersection_left = max(x1_a, x1_b)
    intersection_right = min(x2_a, x2_b)
    intersection_top = max(y1_a, y1_b)
    intersection_bottom = min(y2_a, y2_b)
    intersection_width = max(0, intersection_right - intersection_left)
    intersection_height = max(0, intersection_bottom - intersection_top)
    intersection_area = intersection_width * intersection_height
    box_a_area = (x2_a - x1_a) * (y2_a - y1_a)
    box_b_area = (x2_b - x1_b) * (y2_b - y1_b)
    union_area = box_a_area + box_b_area - intersection_area
    if union_area == 0:
        return 0
    iou = intersection_area / union_area
    return iou

def remove_duplicate_detections(detections, iou_threshold):
    detections_by_confidence = sorted(
        detections,
        key=lambda detection: detection["confidence"],
        reverse=True,
    )
    unique_detections = []
    for candidate_detection in detections_by_confidence:
        candidate_box = candidate_detection["xyxy"]
        is_duplicate = False
        for kept_detection in unique_detections:
            kept_box = kept_detection["xyxy"]
            iou = calculate_iou(candidate_box, kept_box)
            if iou >= iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_detections.append(candidate_detection)
    return unique_detections

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "yolo26m.pt"
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
