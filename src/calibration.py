import json
from math import hypot
from pathlib import Path

import cv2
import numpy as np


def validate_calibration_points(image_points, track_points):
    image_points = np.asarray(image_points, dtype=np.float32)
    track_points = np.asarray(track_points, dtype=np.float32)
    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError("Image calibration points must be two-dimensional points.")
    if track_points.ndim != 2 or track_points.shape[1] != 2:
        raise ValueError("Track calibration points must be two-dimensional points.")
    if len(image_points) != len(track_points):
        raise ValueError("Image and track calibration points must have matching counts.")
    if len(image_points) < 4:
        raise ValueError("At least four point correspondences are required for calibration.")
    if not np.isfinite(image_points).all() or not np.isfinite(track_points).all():
        raise ValueError("Calibration points must be finite numbers.")
    return image_points, track_points


def calculate_homography(image_points, track_points):
    image_points, track_points = validate_calibration_points(
        image_points,
        track_points,
    )
    homography, _ = cv2.findHomography(image_points, track_points, method=0)
    if homography is None:
        raise ValueError("Could not calculate a homography from these calibration points.")
    return homography


def get_vehicle_footpoint(box):
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2, y2)


def project_image_point(image_point, homography):
    image_point_array = np.asarray([[image_point]], dtype=np.float32)
    projected_points = cv2.perspectiveTransform(image_point_array, homography)
    projected_x, projected_y = projected_points[0][0]
    return float(projected_x), float(projected_y)


def project_vehicle_box(box, homography):
    footpoint = get_vehicle_footpoint(box)
    return project_image_point(footpoint, homography)


def project_vehicle_width(box, homography):
    x1, _, x2, y2 = box
    left_road_point = project_image_point((x1, y2), homography)
    right_road_point = project_image_point((x2, y2), homography)
    return hypot(
        right_road_point[0] - left_road_point[0],
        right_road_point[1] - left_road_point[1],
    )


def create_top_down_view(frame, homography, track_map_size):
    map_width, map_height = track_map_size
    if map_width < 1 or map_height < 1:
        raise ValueError("Track-map width and height must be positive.")
    return cv2.warpPerspective(frame, homography, (map_width, map_height))


def load_calibration(calibration_path):
    calibration_path = Path(calibration_path)
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Could not find calibration profile: {calibration_path}")
    with calibration_path.open(encoding="utf-8") as calibration_file:
        calibration = json.load(calibration_file)
    image_points = calibration.get("image_points")
    track_points = calibration.get("track_points")
    track_map_size = calibration.get("track_map_size")
    camera_mode = calibration.get("camera_mode", "unknown")
    validate_calibration_points(image_points, track_points)
    if (
        not isinstance(track_map_size, list)
        or len(track_map_size) != 2
        or not all(isinstance(value, int) and value > 0 for value in track_map_size)
    ):
        raise ValueError("Track-map size must contain two positive whole numbers.")
    if camera_mode not in {"fixed", "moving", "unknown"}:
        raise ValueError("Camera mode must be 'fixed', 'moving', or 'unknown'.")
    return image_points, track_points, tuple(track_map_size), camera_mode


def save_calibration(
    image_points,
    track_points,
    track_map_size,
    calibration_path,
    camera_mode="fixed",
):
    image_points, track_points = validate_calibration_points(
        image_points,
        track_points,
    )
    map_width, map_height = track_map_size
    if map_width < 1 or map_height < 1:
        raise ValueError("Track-map width and height must be positive.")
    if camera_mode != "fixed":
        raise ValueError("A reusable calibration profile is only valid for a fixed camera.")
    calibration_path = Path(calibration_path)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration = {
        "image_points": image_points.tolist(),
        "track_points": track_points.tolist(),
        "track_map_size": [map_width, map_height],
        "camera_mode": camera_mode,
    }
    with calibration_path.open("w", encoding="utf-8") as calibration_file:
        json.dump(calibration, calibration_file, indent=2)
    return calibration_path
