"""Create a read-only quality report for a one-class YOLO label set."""

import argparse
import json
from pathlib import Path

from src.training_data import (
    SUPPORTED_IMAGE_SUFFIXES,
    find_near_duplicate_boxes,
    validate_yolo_label_values,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Audit YOLO vehicle labels without changing images or labels."
    )
    parser.add_argument("images_directory", type=Path)
    parser.add_argument("labels_directory", type=Path)
    parser.add_argument("report_path", type=Path)
    parser.add_argument(
        "--duplicate-iou-threshold",
        type=float,
        default=0.98,
        help="Only report nearly identical box pairs at or above this IoU score.",
    )
    return parser.parse_args()


def load_validated_boxes(label_path):
    """Load one label file, preserving the source order for audit reporting."""
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label file: {label_path}")
    boxes = []
    for line_number, label_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        values = label_line.split()
        validate_yolo_label_values(values, f"{label_path}:{line_number}")
        _, center_x, center_y, box_width, box_height = map(float, values)
        boxes.append((center_x, center_y, box_width, box_height))
    return boxes


def audit_vehicle_labels(images_directory, labels_directory, duplicate_iou_threshold=0.98):
    """Return an auditable report; this function never mutates source data."""
    if not images_directory.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_directory}")
    if not labels_directory.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {labels_directory}")

    image_paths = sorted(
        image_path
        for image_path in images_directory.iterdir()
        if image_path.is_file()
        and image_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {images_directory}")

    image_reports = []
    for image_path in image_paths:
        label_path = labels_directory / f"{image_path.stem}.txt"
        boxes = load_validated_boxes(label_path)
        near_duplicate_pairs = find_near_duplicate_boxes(
            boxes,
            duplicate_iou_threshold,
        )
        image_reports.append(
            {
                "image": image_path.name,
                "label": label_path.name,
                "vehicle_box_count": len(boxes),
                "near_duplicate_box_pairs": near_duplicate_pairs,
            }
        )

    return {
        "duplicate_iou_threshold": duplicate_iou_threshold,
        "image_count": len(image_reports),
        "vehicle_box_count": sum(item["vehicle_box_count"] for item in image_reports),
        "images_with_near_duplicate_boxes": sum(
            bool(item["near_duplicate_box_pairs"]) for item in image_reports
        ),
        "images": image_reports,
    }


def main():
    arguments = parse_arguments()
    report = audit_vehicle_labels(
        arguments.images_directory,
        arguments.labels_directory,
        arguments.duplicate_iou_threshold,
    )
    if arguments.report_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing audit report: {arguments.report_path}"
        )
    arguments.report_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Audited {report['image_count']} images and {report['vehicle_box_count']} boxes.")
    print(
        "Images with near-duplicate labels requiring review: "
        f"{report['images_with_near_duplicate_boxes']}"
    )
    print(f"Saved read-only audit report to: {arguments.report_path}")


if __name__ == "__main__":
    main()
