import argparse
import json
from pathlib import Path

import cv2


def should_sample_frame(frame_index, start_frame_index, frame_step):
    """Return whether ``frame_index`` is on this collection run's sample grid."""
    return (frame_index - start_frame_index) % frame_step == 0


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract representative frames from a video for vehicle annotation."
    )
    parser.add_argument(
        "video_path",
        type=Path,
        help="Video from which to collect training frames.",
    )
    parser.add_argument(
        "--source-name",
        default=None,
        help="Name for this source inside the training-data folder.",
    )
    parser.add_argument(
        "--seconds-between-frames",
        type=float,
        default=1.0,
        help="Minimum time between extracted frames.",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help="Time in seconds at which to begin sampling the video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on the number of frames to extract.",
    )
    return parser.parse_args()


def collect_training_frames(
    video_path,
    dataset_root,
    source_name,
    start_seconds,
    seconds_between_frames,
    max_frames,
):
    if seconds_between_frames <= 0:
        raise ValueError("Seconds between frames must be greater than zero.")
    if start_seconds < 0:
        raise ValueError("Start time cannot be negative.")
    if max_frames is not None and max_frames < 1:
        raise ValueError("Maximum frames must be at least one when provided.")

    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
    if frames_per_second <= 0:
        raise RuntimeError(f"Could not read a valid frame rate from: {video_path}")

    frame_step = max(1, round(frames_per_second * seconds_between_frames))
    start_frame_index = round(frames_per_second * start_seconds)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
    source_directory = dataset_root / "sources" / source_name
    images_directory = source_directory / "images"
    images_directory.mkdir(parents=True, exist_ok=True)

    saved_frames = 0
    frame_index = start_frame_index
    while True:
        frame_read_success, frame = video_capture.read()
        if not frame_read_success:
            break

        if should_sample_frame(frame_index, start_frame_index, frame_step):
            image_path = images_directory / f"frame_{frame_index:06d}.jpg"
            image_saved = cv2.imwrite(str(image_path), frame)
            if not image_saved:
                raise RuntimeError(f"Could not save training frame: {image_path}")
            saved_frames += 1
            if max_frames is not None and saved_frames >= max_frames:
                break

        frame_index += 1

    video_capture.release()
    source_metadata = {
        "video_path": str(video_path),
        "frames_per_second": frames_per_second,
        "start_seconds": start_seconds,
        "seconds_between_frames": seconds_between_frames,
        "vehicle_class": "vehicle",
        "extracted_frames": saved_frames,
    }
    metadata_path = source_directory / "source.json"
    metadata_path.write_text(json.dumps(source_metadata, indent=2), encoding="utf-8")
    return images_directory, saved_frames


def main():
    arguments = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]
    source_name = arguments.source_name or arguments.video_path.stem
    dataset_root = project_root / "data" / "training" / "race_vehicle"
    images_directory, saved_frames = collect_training_frames(
        video_path=arguments.video_path,
        dataset_root=dataset_root,
        source_name=source_name,
        start_seconds=arguments.start_seconds,
        seconds_between_frames=arguments.seconds_between_frames,
        max_frames=arguments.max_frames,
    )
    print(f"Saved {saved_frames} frames to: {images_directory}")


if __name__ == "__main__":
    main()
