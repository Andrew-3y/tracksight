"""Render YOLO vehicle labels onto their source images for human review."""

import argparse
import json
from pathlib import Path

import cv2

from src.training_data import SUPPORTED_IMAGE_SUFFIXES, validate_yolo_label_values


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create a visual review set from one YOLO image/label directory pair."
    )
    parser.add_argument("images_directory", type=Path)
    parser.add_argument("labels_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser.parse_args()


def load_yolo_boxes(label_path):
    """Read one label file after validating every normalized vehicle box."""
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


def normalized_box_to_pixels(box, image_width, image_height):
    center_x, center_y, box_width, box_height = box
    left = round((center_x - box_width / 2) * image_width)
    top = round((center_y - box_height / 2) * image_height)
    right = round((center_x + box_width / 2) * image_width)
    bottom = round((center_y + box_height / 2) * image_height)
    return left, top, right, bottom


def render_label_review(images_directory, labels_directory, output_directory):
    """Save annotated copies and a compact, auditable review manifest.

    This never changes source images or source labels. ``output_directory`` must
    be empty so an older review cannot be confused with a new one.
    """
    if not images_directory.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_directory}")
    if not labels_directory.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {labels_directory}")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"Review output directory must be empty: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        image_path
        for image_path in images_directory.iterdir()
        if image_path.is_file()
        and image_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {images_directory}")

    manifest = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        image_height, image_width = image.shape[:2]
        label_path = labels_directory / f"{image_path.stem}.txt"
        boxes = load_yolo_boxes(label_path)
        reviewed_image = image.copy()
        for box in boxes:
            left, top, right, bottom = normalized_box_to_pixels(
                box,
                image_width,
                image_height,
            )
            cv2.rectangle(reviewed_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            reviewed_image,
            f"vehicle boxes: {len(boxes)}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 0),
            4,
        )
        cv2.putText(
            reviewed_image,
            f"vehicle boxes: {len(boxes)}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
        )
        output_path = output_directory / image_path.name
        if not cv2.imwrite(str(output_path), reviewed_image):
            raise RuntimeError(f"Could not save review image: {output_path}")
        manifest.append(
            {
                "image": image_path.name,
                "label": label_path.name,
                "vehicle_box_count": len(boxes),
                "review_image": output_path.name,
            }
        )

    manifest_path = output_directory / "review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, manifest


def main():
    arguments = parse_arguments()
    manifest_path, manifest = render_label_review(
        arguments.images_directory,
        arguments.labels_directory,
        arguments.output_directory,
    )
    print(f"Saved {len(manifest)} label-review images to: {arguments.output_directory}")
    print(f"Saved review manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
