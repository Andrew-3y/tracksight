"""Evaluate a YOLO model against TrackSight's one-class vehicle labels.

Generic YOLO models name a car with a model-specific class ID (for example,
COCO's ``car`` is class 2), while TrackSight training data intentionally uses
only class 0: ``vehicle``.  Ultralytics' built-in validator compares raw
numeric IDs, so it cannot fairly evaluate the generic model against this
dataset.  This script first filters a model to its recognised vehicle classes,
then scores those detections as the one TrackSight class.
"""

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.utils import YAML

from src.detect_vehicles import get_vehicle_class_ids
from src.training_data import (
    SUPPORTED_IMAGE_SUFFIXES,
    find_label_path,
    get_dataset_root,
    validate_yolo_label_file,
)


def parse_arguments():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO model as TrackSight's one-class vehicle detector."
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        required=True,
        help="One-class TrackSight YOLO dataset.yaml file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=project_root / "models" / "yolo26m.pt",
        help="YOLO model file to evaluate.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Dataset split to evaluate; validation is the default.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.2,
        help="Minimum vehicle-detection confidence to score.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Minimum overlap required for a true-positive match.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1280,
        help="Inference image size in pixels.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        required=True,
        help="New JSON file where the evaluation report is written.",
    )
    return parser.parse_args()


def calculate_iou(box_a, box_b):
    """Return IoU for two pixel-coordinate ``(left, top, right, bottom)`` boxes."""
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection_width = max(0.0, right - left)
    intersection_height = max(0.0, bottom - top)
    intersection_area = intersection_width * intersection_height
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - intersection_area
    return intersection_area / union_area if union_area else 0.0


def load_ground_truth_boxes(label_path, image_width, image_height):
    """Convert one validated YOLO label file into pixel-coordinate boxes."""
    validate_yolo_label_file(label_path)
    boxes = []
    for label_line in label_path.read_text(encoding="utf-8").splitlines():
        _, center_x, center_y, box_width, box_height = map(float, label_line.split())
        half_width = box_width * image_width / 2
        half_height = box_height * image_height / 2
        boxes.append(
            (
                center_x * image_width - half_width,
                center_y * image_height - half_height,
                center_x * image_width + half_width,
                center_y * image_height + half_height,
            )
        )
    return boxes


def match_detections(predictions, ground_truth_boxes, iou_threshold):
    """Greedily match high-confidence predictions to different labelled cars.

    Each prediction can match at most one label and each label can match at
    most one prediction.  Sorting by confidence gives the most confident box
    first chance to represent a car, which is the normal detection-scoring
    convention.
    """
    matched_ground_truth_indexes = set()
    true_positives = 0
    for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        best_index = None
        best_iou = iou_threshold
        for ground_truth_index, ground_truth_box in enumerate(ground_truth_boxes):
            if ground_truth_index in matched_ground_truth_indexes:
                continue
            iou = calculate_iou(prediction["xyxy"], ground_truth_box)
            if iou >= best_iou:
                best_index = ground_truth_index
                best_iou = iou
        if best_index is not None:
            matched_ground_truth_indexes.add(best_index)
            true_positives += 1

    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth_boxes) - true_positives
    return true_positives, false_positives, false_negatives


def get_split_image_paths(dataset_config_path, split_name):
    """Load the requested validated dataset split without changing its files."""
    dataset_config = YAML.load(dataset_config_path)
    dataset_root = get_dataset_root(dataset_config, dataset_config_path)
    configured_split_path = dataset_config.get(split_name)
    if not isinstance(configured_split_path, str):
        raise ValueError(f"Dataset configuration is missing a folder path for '{split_name}'.")
    images_directory = Path(configured_split_path)
    if not images_directory.is_absolute():
        images_directory = dataset_root / images_directory
    image_paths = sorted(
        image_path
        for image_path in images_directory.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in '{split_name}': {images_directory}")
    return image_paths


def evaluate_vehicle_detector(
    dataset_config_path,
    model_path,
    split_name,
    confidence_threshold,
    iou_threshold,
    image_size,
):
    """Evaluate recognised vehicle classes as TrackSight's one vehicle class."""
    if not dataset_config_path.is_file():
        raise FileNotFoundError(f"Could not find dataset configuration: {dataset_config_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Could not find model: {model_path}")
    if not 0 < confidence_threshold <= 1:
        raise ValueError("Confidence threshold must be above zero and at most one.")
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be above zero and at most one.")
    if image_size < 32:
        raise ValueError("Image size must be at least 32 pixels.")

    model = YOLO(model_path)
    vehicle_class_ids = get_vehicle_class_ids(model)
    image_paths = get_split_image_paths(dataset_config_path, split_name)
    totals = {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
    image_reports = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read evaluation image: {image_path}")
        image_height, image_width = image.shape[:2]
        ground_truth_boxes = load_ground_truth_boxes(
            find_label_path(image_path), image_width, image_height
        )
        result = model(
            image,
            classes=vehicle_class_ids,
            conf=confidence_threshold,
            imgsz=image_size,
            verbose=False,
        )[0]
        predictions = [
            {"xyxy": tuple(box[:4]), "confidence": box[4]}
            for box in result.boxes.data.cpu().tolist()
        ]
        true_positives, false_positives, false_negatives = match_detections(
            predictions, ground_truth_boxes, iou_threshold
        )
        totals["true_positives"] += true_positives
        totals["false_positives"] += false_positives
        totals["false_negatives"] += false_negatives
        image_reports.append(
            {
                "image": image_path.name,
                "labelled_vehicle_count": len(ground_truth_boxes),
                "predicted_vehicle_count": len(predictions),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            }
        )

    precision_denominator = totals["true_positives"] + totals["false_positives"]
    recall_denominator = totals["true_positives"] + totals["false_negatives"]
    precision = totals["true_positives"] / precision_denominator if precision_denominator else 0.0
    recall = totals["true_positives"] / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1_score = 2 * precision * recall / f1_denominator if f1_denominator else 0.0
    return {
        "model_path": str(model_path),
        "dataset_config_path": str(dataset_config_path),
        "split": split_name,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "image_size": image_size,
        "model_vehicle_class_ids": vehicle_class_ids,
        "true_positives": totals["true_positives"],
        "false_positives": totals["false_positives"],
        "false_negatives": totals["false_negatives"],
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "images": image_reports,
    }


def main():
    arguments = parse_arguments()
    if arguments.report_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing evaluation report: {arguments.report_path}"
        )
    report = evaluate_vehicle_detector(
        dataset_config_path=arguments.dataset_config,
        model_path=arguments.model_path,
        split_name=arguments.split,
        confidence_threshold=arguments.confidence_threshold,
        iou_threshold=arguments.iou_threshold,
        image_size=arguments.image_size,
    )
    arguments.report_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"{report['split']}: precision={report['precision']:.3f}, "
        f"recall={report['recall']:.3f}, f1={report['f1_score']:.3f}"
    )
    print(f"Saved vehicle evaluation report to: {arguments.report_path}")


if __name__ == "__main__":
    main()
