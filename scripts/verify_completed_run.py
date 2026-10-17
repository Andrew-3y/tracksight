"""Verify that a completed TrackSight run is internally consistent and reviewable."""

import argparse
import json
from pathlib import Path

import cv2


REQUIRED_RUN_SUMMARY_FIELDS = {
    "all_observations",
    "all_track_segments",
    "frames_per_second",
    "observations_per_frame",
    "observed_close_approach_event_count",
    "processed_frame_count",
    "risk_event_count",
    "spatial_analysis_enabled",
    "track_quality",
    "trusted_track_segments",
}


def get_run_artifacts(output_video_path):
    """Return the files produced beside one annotated TrackSight video."""
    output_video_path = Path(output_video_path)
    return {
        "video": output_video_path,
        "trajectories": output_video_path.with_name(
            f"{output_video_path.stem}_trajectories.json"
        ),
        "review_evidence": output_video_path.with_name(
            f"{output_video_path.stem}_review_evidence.json"
        ),
        "risk_evidence": output_video_path.with_name(
            f"{output_video_path.stem}_risk_evidence.json"
        ),
        "run_summary": output_video_path.with_name(
            f"{output_video_path.stem}_run_summary.json"
        ),
        "top_down_video": output_video_path.with_name(
            f"{output_video_path.stem}_top_down.mp4"
        ),
    }


def load_json_artifact(artifact_path, artifact_name):
    """Load one required JSON artifact with a clear failure message."""
    artifact_path = Path(artifact_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing {artifact_name}: {artifact_path}")
    with artifact_path.open(encoding="utf-8") as artifact_file:
        return json.load(artifact_file)


def validate_run_summary(run_summary):
    """Check the summary schema that makes a run safe to review in the dashboard."""
    if not isinstance(run_summary, dict):
        raise ValueError("Run summary must be a JSON object.")
    missing_fields = REQUIRED_RUN_SUMMARY_FIELDS.difference(run_summary)
    if missing_fields:
        raise ValueError(
            "Run summary is missing required fields: "
            + ", ".join(sorted(missing_fields))
        )
    if not isinstance(run_summary["track_quality"], dict):
        raise ValueError("Run summary field 'track_quality' must be an object.")
    if not isinstance(run_summary["spatial_analysis_enabled"], bool):
        raise ValueError(
            "Run summary field 'spatial_analysis_enabled' must be true or false."
        )


def validate_video(video_path, artifact_name):
    """Confirm an MP4 exists and OpenCV can decode its first frame."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing {artifact_name}: {video_path}")
    video_capture = cv2.VideoCapture(str(video_path))
    try:
        if not video_capture.isOpened():
            raise RuntimeError(f"Could not open {artifact_name}: {video_path}")
        frame_read, _ = video_capture.read()
        if not frame_read:
            raise RuntimeError(f"Could not decode a frame from {artifact_name}: {video_path}")
    finally:
        video_capture.release()


def validate_completed_run(output_video_path):
    """Verify the files and counts from a completed run without changing them."""
    artifacts = get_run_artifacts(output_video_path)
    validate_video(artifacts["video"], "annotated video")
    trajectories = load_json_artifact(artifacts["trajectories"], "trajectory data")
    review_evidence = load_json_artifact(
        artifacts["review_evidence"], "review evidence"
    )
    risk_evidence = load_json_artifact(artifacts["risk_evidence"], "risk evidence")
    run_summary = load_json_artifact(artifacts["run_summary"], "run summary")
    validate_run_summary(run_summary)

    if not isinstance(trajectories, dict):
        raise ValueError("Trajectory data must be a JSON object keyed by tracker ID.")
    if not isinstance(review_evidence, list) or not isinstance(risk_evidence, list):
        raise ValueError("Evidence artifacts must be JSON lists.")

    observation_count = sum(len(points) for points in trajectories.values())
    if len(trajectories) != run_summary["all_track_segments"]:
        raise ValueError("Trajectory segment count does not match the run summary.")
    if observation_count != run_summary["all_observations"]:
        raise ValueError("Trajectory observation count does not match the run summary.")
    if len(review_evidence) != run_summary["observed_close_approach_event_count"]:
        raise ValueError("Observed-evidence count does not match the run summary.")
    if len(risk_evidence) != run_summary["risk_event_count"]:
        raise ValueError("Risk-evidence count does not match the run summary.")

    if run_summary["spatial_analysis_enabled"]:
        validate_video(artifacts["top_down_video"], "top-down video")
    elif review_evidence or risk_evidence:
        raise ValueError(
            "A run without stable spatial analysis must not contain safety evidence."
        )

    return {
        "track_segments": len(trajectories),
        "observations": observation_count,
        "spatial_analysis_enabled": run_summary["spatial_analysis_enabled"],
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Verify a completed TrackSight annotated-video run."
    )
    parser.add_argument(
        "output_video_path",
        type=Path,
        help="Annotated MP4 produced by src.run_video_tracking.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    verification = validate_completed_run(arguments.output_video_path)
    print(f"Verified completed run: {arguments.output_video_path}")
    print(f"Tracker ID segments: {verification['track_segments']}")
    print(f"Detector positions: {verification['observations']}")
    print(f"Spatial analysis enabled: {verification['spatial_analysis_enabled']}")


if __name__ == "__main__":
    main()
