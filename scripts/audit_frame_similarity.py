"""Report visually near-duplicate candidate frames without changing them."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from src.training_data import SUPPORTED_IMAGE_SUFFIXES


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Audit candidate-frame similarity before annotation."
    )
    parser.add_argument(
        "images_directories",
        nargs="+",
        type=Path,
        help="One or more directories containing candidate frame images.",
    )
    parser.add_argument("report_path", type=Path)
    parser.add_argument(
        "--maximum-distance",
        type=int,
        default=8,
        help="Maximum 64-bit difference-hash distance reported as near-duplicate.",
    )
    return parser.parse_args()


def calculate_difference_hash(image):
    """Return a compact 64-bit visual fingerprint for one image.

    The image is converted to grey-scale and reduced to eight rows and nine
    columns.  Each bit records whether a pixel is brighter than its immediate
    neighbour.  The fingerprint is resilient to small compression and lighting
    changes, while still flagging frames that are almost the same view.
    """
    if image is None:
        raise ValueError("Cannot calculate a visual hash for an unreadable image.")
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reduced_image = cv2.resize(grey_image, (9, 8), interpolation=cv2.INTER_AREA)
    return (reduced_image[:, 1:] > reduced_image[:, :-1]).reshape(-1)


def calculate_hash_distance(first_hash, second_hash):
    """Return the number of visual-fingerprint bits that disagree."""
    return int(np.count_nonzero(first_hash != second_hash))


def get_image_paths(images_directories):
    image_paths = []
    for images_directory in images_directories:
        if not images_directory.is_dir():
            raise FileNotFoundError(f"Missing candidate-frame directory: {images_directory}")
        image_paths.extend(
            image_path
            for image_path in sorted(images_directory.iterdir())
            if image_path.is_file()
            and image_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
    if not image_paths:
        raise FileNotFoundError("No candidate-frame images were found.")
    return image_paths


def audit_frame_similarity(images_directories, maximum_distance=8):
    """Return all close visual pairs; never edits candidate image files."""
    if not 0 <= maximum_distance <= 64:
        raise ValueError("Maximum hash distance must be between 0 and 64.")

    image_paths = get_image_paths(images_directories)
    fingerprints = {}
    for image_path in image_paths:
        fingerprints[image_path] = calculate_difference_hash(cv2.imread(str(image_path)))

    near_duplicate_pairs = []
    for first_path, second_path in combinations(image_paths, 2):
        distance = calculate_hash_distance(
            fingerprints[first_path], fingerprints[second_path]
        )
        if distance <= maximum_distance:
            near_duplicate_pairs.append(
                {
                    "first_image": str(first_path),
                    "second_image": str(second_path),
                    "hash_distance": distance,
                }
            )

    return {
        "image_count": len(image_paths),
        "maximum_distance": maximum_distance,
        "near_duplicate_pair_count": len(near_duplicate_pairs),
        "near_duplicate_pairs": near_duplicate_pairs,
    }


def main():
    arguments = parse_arguments()
    if arguments.report_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing audit report: {arguments.report_path}"
        )
    report = audit_frame_similarity(
        arguments.images_directories,
        arguments.maximum_distance,
    )
    arguments.report_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Audited {report['image_count']} images; found "
        f"{report['near_duplicate_pair_count']} near-duplicate pairs."
    )
    print(f"Saved read-only similarity report to: {arguments.report_path}")


if __name__ == "__main__":
    main()
