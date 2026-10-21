"""Verify that a completed TrackSight run is internally consistent and reviewable."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import cv2

from src.safety_zones import (
    SAFETY_ZONE_EVIDENCE_SCHEMA_VERSION,
    validate_safety_zone_profile,
)
from src.safety_candidates import (
    SAFETY_CANDIDATE_EVIDENCE_SCHEMA_VERSION,
    get_safety_candidates,
)
from src.evidence_package import make_portable_safety_candidate_evidence


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
        "marshal_briefing": output_video_path.with_name(
            f"{output_video_path.stem}_marshal_briefing.json"
        ),
        "marshal_evidence_package": output_video_path.with_name(
            f"{output_video_path.stem}_marshal_evidence_package.zip"
        ),
        "run_summary": output_video_path.with_name(
            f"{output_video_path.stem}_run_summary.json"
        ),
        "top_down_video": output_video_path.with_name(
            f"{output_video_path.stem}_top_down.mp4"
        ),
        "safety_zone_evidence": output_video_path.with_name(
            f"{output_video_path.stem}_safety_zone_evidence.json"
        ),
        "safety_candidate_evidence": output_video_path.with_name(
            f"{output_video_path.stem}_safety_candidates.json"
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


def validate_marshal_briefing(briefing, run_summary):
    """Check an optional briefing agrees with the evidence counts it summarizes."""
    if not isinstance(briefing, dict):
        raise ValueError("Marshal briefing must be a JSON object.")
    if briefing.get("briefing_type") != "human_review":
        raise ValueError("Marshal briefing must be marked as human_review.")
    summary = briefing.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("Marshal briefing is missing its summary.")
    if summary.get("candidate_risk_event_count") != run_summary["risk_event_count"]:
        raise ValueError("Marshal briefing risk count does not match the run summary.")
    if (
        summary.get("observed_proximity_event_count")
        != run_summary["observed_close_approach_event_count"]
    ):
        raise ValueError(
            "Marshal briefing observed-proximity count does not match the run summary."
        )
    for briefing_key, run_summary_key in (
        (
            "stationary_vehicle_candidate_count",
            "stationary_vehicle_candidate_count",
        ),
        (
            "restricted_area_entry_candidate_count",
            "restricted_area_entry_candidate_count",
        ),
    ):
        if (
            run_summary_key in run_summary
            and summary.get(briefing_key) != run_summary[run_summary_key]
        ):
            raise ValueError(
                "Marshal briefing safety-candidate count does not match the run summary."
            )
    if not isinstance(briefing.get("predicted_risk_highlights"), list):
        raise ValueError("Marshal briefing predicted-risk highlights must be a list.")
    if not isinstance(briefing.get("observed_proximity_highlights"), list):
        raise ValueError("Marshal briefing observed-proximity highlights must be a list.")
    candidate_highlights = briefing.get("safety_candidate_highlights", [])
    if not isinstance(candidate_highlights, list):
        raise ValueError("Marshal briefing safety-candidate highlights must be a list.")
    for highlight_type in (
        "predicted_risk_highlights",
        "observed_proximity_highlights",
        "safety_candidate_highlights",
    ):
        for highlight in briefing.get(highlight_type, []):
            supporting_clip = highlight.get("supporting_clip")
            if not isinstance(supporting_clip, str) or not Path(supporting_clip).is_file():
                raise ValueError(
                    "Each Marshal briefing highlight must have a saved supporting clip."
                )
            validate_video(
                supporting_clip,
                "Marshal briefing supporting clip",
            )


def validate_marshal_evidence_package(
    package_path,
    briefing,
    safety_zone_evidence=None,
    safety_candidate_evidence=None,
):
    """Check the optional portable handoff contains every advertised briefing clip."""
    if not is_zipfile(package_path):
        raise ValueError("Marshal evidence package is not a valid ZIP archive.")
    with ZipFile(package_path) as archive:
        archive_names = set(archive.namelist())
        required_names = {
            "README.txt",
            "manifest.json",
            "marshal_briefing.json",
            "run_summary.json",
            "predicted_risk_evidence.json",
            "observed_proximity_evidence.json",
            "trajectories.json",
        }
        if not required_names.issubset(archive_names):
            raise ValueError("Marshal evidence package is missing required files.")
        packaged_briefing = json.loads(archive.read("marshal_briefing.json"))
        for highlight_type in (
            "predicted_risk_highlights",
            "observed_proximity_highlights",
            "safety_candidate_highlights",
        ):
            highlights = briefing.get(highlight_type, [])
            packaged_highlights = packaged_briefing.get(highlight_type, [])
            if not isinstance(packaged_highlights, list) or len(packaged_highlights) != len(highlights):
                raise ValueError("Marshal evidence package has inconsistent briefing highlights.")
            for highlight, packaged_highlight in zip(highlights, packaged_highlights):
                expected_name = f"briefing_clips/{Path(highlight['supporting_clip']).name}"
                if expected_name not in archive_names:
                    raise ValueError(
                        "Marshal evidence package is missing a briefing supporting clip."
                    )
                if packaged_highlight.get("supporting_clip") != expected_name:
                    raise ValueError(
                        "Marshal evidence package exposes a non-portable briefing clip path."
                    )
        if archive.testzip() is not None:
            raise ValueError("Marshal evidence package contains a corrupt file.")
        if safety_zone_evidence is not None:
            if "safety_zone_evidence.json" not in archive_names:
                raise ValueError(
                    "Marshal evidence package is missing its safety-zone evidence."
                )
            packaged_zone_evidence = json.loads(
                archive.read("safety_zone_evidence.json")
            )
            if packaged_zone_evidence != safety_zone_evidence:
                raise ValueError(
                    "Marshal evidence package safety-zone evidence does not match the run."
                )
        if safety_candidate_evidence is not None:
            if "safety_candidate_evidence.json" not in archive_names:
                raise ValueError(
                    "Marshal evidence package is missing its safety-candidate evidence."
                )
            packaged_candidate_evidence = json.loads(
                archive.read("safety_candidate_evidence.json")
            )
            if packaged_candidate_evidence != make_portable_safety_candidate_evidence(
                safety_candidate_evidence
            ):
                raise ValueError(
                    "Marshal evidence package safety-candidate evidence does not match the run."
                )


def validate_safety_zone_evidence(safety_zone_evidence, run_summary):
    """Check optional zone evidence remains tied to valid calibrated observations."""
    if not isinstance(safety_zone_evidence, dict):
        raise ValueError("Safety-zone evidence must be a JSON object.")
    if safety_zone_evidence.get("schema_version") != SAFETY_ZONE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Safety-zone evidence uses an unsupported schema version.")
    validate_safety_zone_profile(
        {
            "schema_version": 1,
            "coordinate_system": safety_zone_evidence.get("coordinate_system"),
            "zones": safety_zone_evidence.get("zones"),
        }
    )
    observations = safety_zone_evidence.get("observations")
    if not isinstance(observations, list):
        raise ValueError("Safety-zone observations must be a JSON list.")
    if run_summary.get("safety_zone_observation_count") != len(observations):
        raise ValueError("Safety-zone observation count does not match the run summary.")


def validate_safety_candidate_evidence(safety_candidate_evidence, run_summary):
    """Check optional candidate evidence stays calibrated and count-consistent."""
    if not isinstance(safety_candidate_evidence, dict):
        raise ValueError("Safety-candidate evidence must be a JSON object.")
    if (
        safety_candidate_evidence.get("schema_version")
        != SAFETY_CANDIDATE_EVIDENCE_SCHEMA_VERSION
    ):
        raise ValueError("Safety-candidate evidence uses an unsupported schema version.")
    candidate_types = (
        "stationary_vehicle_candidates",
        "restricted_area_entry_candidates",
    )
    for candidate_type in candidate_types:
        if not isinstance(safety_candidate_evidence.get(candidate_type), list):
            raise ValueError(f"Safety-candidate field {candidate_type} must be a list.")
    if not run_summary["spatial_analysis_enabled"] and get_safety_candidates(
        safety_candidate_evidence
    ):
        raise ValueError(
            "A run without stable spatial analysis must not contain safety candidates."
        )
    counts = {
        "stationary_vehicle_candidate_count": len(
            safety_candidate_evidence["stationary_vehicle_candidates"]
        ),
        "restricted_area_entry_candidate_count": len(
            safety_candidate_evidence["restricted_area_entry_candidates"]
        ),
    }
    for summary_key, count in counts.items():
        if summary_key in run_summary and run_summary[summary_key] != count:
            raise ValueError(
                "Safety-candidate count does not match the run summary."
            )


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
    safety_zone_evidence = None
    if artifacts["safety_zone_evidence"].is_file():
        safety_zone_evidence = load_json_artifact(
            artifacts["safety_zone_evidence"], "safety-zone evidence"
        )
        validate_safety_zone_evidence(safety_zone_evidence, run_summary)
    elif run_summary.get("run_provenance", {}).get("safety_zone_profile"):
        raise FileNotFoundError(
            f"Missing safety-zone evidence: {artifacts['safety_zone_evidence']}"
        )
    safety_candidate_evidence = None
    if artifacts["safety_candidate_evidence"].is_file():
        safety_candidate_evidence = load_json_artifact(
            artifacts["safety_candidate_evidence"], "safety-candidate evidence"
        )
        validate_safety_candidate_evidence(safety_candidate_evidence, run_summary)
    elif (
        "stationary_vehicle_candidate_count" in run_summary
        or "restricted_area_entry_candidate_count" in run_summary
    ):
        raise FileNotFoundError(
            "Missing safety-candidate evidence: "
            f"{artifacts['safety_candidate_evidence']}"
        )
    if artifacts["marshal_briefing"].is_file():
        marshal_briefing = load_json_artifact(
            artifacts["marshal_briefing"], "Marshal briefing"
        )
        validate_marshal_briefing(marshal_briefing, run_summary)
        if not artifacts["marshal_evidence_package"].is_file():
            raise FileNotFoundError(
                f"Missing Marshal evidence package: {artifacts['marshal_evidence_package']}"
            )
        validate_marshal_evidence_package(
            artifacts["marshal_evidence_package"],
            marshal_briefing,
            safety_zone_evidence,
            safety_candidate_evidence,
        )

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
