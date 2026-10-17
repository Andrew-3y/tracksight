import argparse
from pathlib import Path

import cv2
import numpy as np

from src.calibration import (
    calculate_homography,
    create_top_down_view,
    save_calibration,
)


def parse_arguments():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create a camera-to-track calibration profile by clicking four track points."
    )
    parser.add_argument(
        "video_path",
        type=Path,
        help="Path to a fixed-camera race video.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Where to save the calibration JSON. Defaults to configs/<video>_calibration.json.",
    )
    parser.add_argument(
        "--map-width",
        type=int,
        default=1280,
        help="Width of the generated top-down map in pixels.",
    )
    parser.add_argument(
        "--map-height",
        type=int,
        default=720,
        help="Height of the generated top-down map in pixels.",
    )
    parser.add_argument(
        "--preview-path",
        type=Path,
        default=None,
        help="Where to save a top-down preview image. Defaults to outputs/calibration.",
    )
    return parser.parse_args(), project_root


def draw_calibration_points(frame, image_points):
    annotated_frame = frame.copy()
    for point_index, point in enumerate(image_points, start=1):
        x, y = point
        cv2.circle(annotated_frame, (x, y), 7, (0, 255, 0), -1)
        cv2.putText(
            annotated_frame,
            str(point_index),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    if len(image_points) > 1:
        cv2.polylines(
            annotated_frame,
            [np.asarray(image_points, dtype=np.int32)],
            False,
            (0, 255, 0),
            2,
        )
    instructions = "Click four corners clockwise | R: reset | S: save | Esc: cancel"
    cv2.putText(
        annotated_frame,
        instructions,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return annotated_frame


def collect_image_points(frame):
    window_name = "TrackSight calibration"
    image_points = []

    def add_image_point(event, x, y, _flags, _parameters):
        if event == cv2.EVENT_LBUTTONDOWN and len(image_points) < 4:
            image_points.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, add_image_point)
    while True:
        cv2.imshow(window_name, draw_calibration_points(frame, image_points))
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            image_points.clear()
        elif key == ord("s"):
            if len(image_points) == 4:
                break
            print("Choose exactly four points before saving.")
        elif key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError("Calibration cancelled by user.")
    cv2.destroyWindow(window_name)
    return image_points


def calibrate_track():
    arguments, project_root = parse_arguments()
    if arguments.map_width < 1 or arguments.map_height < 1:
        raise ValueError("Map width and map height must both be positive.")

    video_capture = cv2.VideoCapture(str(arguments.video_path))
    if not video_capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {arguments.video_path}")
    frame_read, frame = video_capture.read()
    video_capture.release()
    if not frame_read:
        raise RuntimeError("Could not read the first frame for calibration.")

    image_points = collect_image_points(frame)
    track_map_size = (arguments.map_width, arguments.map_height)
    track_points = [
        (0, 0),
        (arguments.map_width, 0),
        (arguments.map_width, arguments.map_height),
        (0, arguments.map_height),
    ]
    homography = calculate_homography(image_points, track_points)

    output_path = arguments.output_path or (
        project_root / "configs" / f"{arguments.video_path.stem}_calibration.json"
    )
    save_calibration(image_points, track_points, track_map_size, output_path)

    preview_path = arguments.preview_path or (
        project_root
        / "outputs"
        / "calibration"
        / f"{arguments.video_path.stem}_top_down_preview.png"
    )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    top_down_view = create_top_down_view(frame, homography, track_map_size)
    if not cv2.imwrite(str(preview_path), top_down_view):
        raise RuntimeError(f"Could not save calibration preview: {preview_path}")

    print(f"Saved calibration profile to: {output_path}")
    print(f"Saved top-down preview to: {preview_path}")


if __name__ == "__main__":
    calibrate_track()
