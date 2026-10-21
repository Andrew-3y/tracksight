import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from src.calibration import calculate_homography, load_calibration
from src.process_video import process_video
from src.safety_zones import load_safety_zone_profile
from src.input_sources import classify_input_source


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Detect and track vehicles in a video."
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Existing video path, or an rtsp/http(s) input URL TrackSight should record and process.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "yolo26m.pt",
        help="Path to a YOLO vehicle-detection model file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Where to save the annotated MP4. Defaults to outputs/tracking.",
    )
    parser.add_argument(
        "--regions-config",
        type=Path,
        default=None,
        help="Optional JSON file describing race areas. Defaults to a same-named profile when available.",
    )
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=None,
        help="Optional camera-to-track calibration JSON. Defaults to a same-named calibration profile when available.",
    )
    parser.add_argument(
        "--safety-zone-config",
        type=Path,
        default=None,
        help="Optional local-track safety-zone JSON. Defaults to a same-named profile when available and requires camera calibration.",
    )
    parser.add_argument(
        "--max-missing-seconds",
        type=float,
        default=1.5,
        help="How long the motion-compensated tracker keeps a lost ID available for a valid recovery.",
    )
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.2,
        help="Minimum YOLO confidence required before a vehicle reaches the tracker.",
    )
    parser.add_argument(
        "--tracker-high-confidence",
        type=float,
        default=0.3,
        help="Confidence for BoT-SORT's first, high-quality association stage.",
    )
    parser.add_argument(
        "--tracker-low-confidence",
        type=float,
        default=0.2,
        help="Confidence for BoT-SORT's recovery association stage.",
    )
    parser.add_argument(
        "--new-track-confidence",
        type=float,
        default=0.45,
        help="Confidence required before BoT-SORT creates a brand-new internal track.",
    )
    parser.add_argument(
        "--tracker-match-threshold",
        type=float,
        default=0.8,
        help="BoT-SORT global-assignment threshold; higher values permit weaker matches.",
    )
    parser.add_argument(
        "--close-approach-ratio",
        type=float,
        default=1.0,
        help="Largest center distance, measured in average vehicle diagonals, for observed proximity evidence.",
    )
    parser.add_argument(
        "--minimum-close-frames",
        type=int,
        default=3,
        help="Consecutive close frames required before TrackSight records review evidence.",
    )
    parser.add_argument(
        "--review-pre-seconds",
        type=float,
        default=2.0,
        help="Seconds included before the closest point in each review clip.",
    )
    parser.add_argument(
        "--review-post-seconds",
        type=float,
        default=2.0,
        help="Seconds included after the closest point in each review clip.",
    )
    parser.add_argument(
        "--max-review-clips",
        type=int,
        default=10,
        help="Maximum number of non-overlapping review clips to save.",
    )
    parser.add_argument(
        "--maximum-motion-gap-seconds",
        type=float,
        default=0.25,
        help="Longest observation gap allowed when calculating a measured motion segment.",
    )
    parser.add_argument(
        "--prediction-horizon-seconds",
        type=float,
        default=1.5,
        help="How far ahead TrackSight projects measured vehicle motion for candidate-risk analysis.",
    )
    parser.add_argument(
        "--maximum-predicted-separation-vehicle-widths",
        type=float,
        default=1.5,
        help="Largest predicted separation, in average vehicle widths, recorded as candidate risk evidence.",
    )
    parser.add_argument(
        "--minimum-risk-frames",
        type=int,
        default=18,
        help="Consecutive candidate-risk frames required before TrackSight saves a risk event.",
    )
    parser.add_argument(
        "--minimum-trusted-track-frames",
        type=int,
        default=30,
        help="Continuous real observations required before a track may create safety evidence.",
    )
    parser.add_argument(
        "--minimum-confirmed-seconds",
        type=float,
        default=0.2,
        help="Continuous real-observation time required before a track receives a visible ID and is saved as trajectory data.",
    )
    parser.add_argument(
        "--camera-motion-method",
        choices=("sparseOptFlow", "none"),
        default="sparseOptFlow",
        help="Camera-motion compensation: sparseOptFlow for moving cameras, none for static cameras.",
    )
    parser.add_argument(
        "--max-marshal-briefing-highlights",
        type=int,
        default=5,
        help="Maximum supported review cards saved for each Marshal briefing evidence type.",
    )
    parser.add_argument(
        "--minimum-stationary-seconds",
        type=float,
        default=1.0,
        help="Consecutive real calibrated seconds required before TrackSight saves a low-displacement vehicle candidate.",
    )
    parser.add_argument(
        "--maximum-stationary-spread-vehicle-widths",
        type=float,
        default=0.75,
        help="Largest observed stationary-candidate spread, measured in that segment's median projected vehicle widths.",
    )
    parser.add_argument(
        "--minimum-restricted-area-frames",
        type=int,
        default=3,
        help="Consecutive real calibrated frames required in a reviewer-defined restricted area before saving a candidate.",
    )
    return parser.parse_args()


def load_tracking_config(config_path):
    if config_path is None:
        return {"full_frame": [0.0, 0.0, 1.0, 1.0]}
    if not config_path.is_file():
        raise FileNotFoundError(f"Could not find track-region profile: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    normalized_track_regions = config.get("normalized_track_regions")
    if not isinstance(normalized_track_regions, dict) or not normalized_track_regions:
        raise ValueError(
            "The track-region profile must contain a non-empty 'normalized_track_regions' object."
        )

    return normalized_track_regions


def load_calibration_homography(config_path):
    if config_path is None:
        return None, None, "unknown"
    image_points, track_points, track_map_size, camera_mode = load_calibration(config_path)
    if camera_mode == "unknown":
        return None, None, camera_mode
    calibration_homography = calculate_homography(image_points, track_points)
    return calibration_homography, track_map_size, camera_mode


def run_video_tracking():
    arguments = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]
    source_info = classify_input_source(arguments.video_path)
    video_path = (
        Path(source_info["source"])
        if source_info["source_type"] == "video_file"
        else source_info["source"]
    )
    source_stem = (
        Path(source_info["source"]).stem
        if source_info["source_type"] == "video_file"
        else "live_stream"
    )
    model_path = arguments.model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"Could not find detector model: {model_path}")
    if arguments.regions_config is None:
        default_regions_config_path = (
            project_root / "configs" / f"{source_stem}.json"
        )
        regions_config_path = (
            default_regions_config_path
            if default_regions_config_path.is_file()
            else None
        )
    else:
        regions_config_path = arguments.regions_config
    if arguments.calibration_config is None:
        default_calibration_config_path = (
            project_root / "configs" / f"{source_stem}_calibration.json"
        )
        calibration_config_path = (
            default_calibration_config_path
            if default_calibration_config_path.is_file()
            else None
        )
    else:
        calibration_config_path = arguments.calibration_config
    if arguments.safety_zone_config is None:
        default_safety_zone_config_path = (
            project_root / "configs" / f"{source_stem}_safety_zones.json"
        )
        safety_zone_config_path = (
            default_safety_zone_config_path
            if default_safety_zone_config_path.is_file()
            else None
        )
    else:
        safety_zone_config_path = arguments.safety_zone_config
    normalized_track_regions = load_tracking_config(regions_config_path)
    calibration_homography, track_map_size, camera_mode = load_calibration_homography(
        calibration_config_path
    )
    if safety_zone_config_path is not None and calibration_homography is None:
        raise ValueError(
            "Safety zones require a usable camera calibration because their coordinates are local track units."
        )
    safety_zone_profile = (
        None
        if safety_zone_config_path is None
        else load_safety_zone_profile(safety_zone_config_path)
    )
    if arguments.output_path is None:
        if source_info["requires_recording_output"]:
            raise ValueError(
                "A network stream needs an explicit --output-path so TrackSight can "
                "record a reviewable MP4 rather than overwrite an unknown source."
            )
        output_video_path = (
            project_root
            / "outputs"
            / "tracking"
            / f"annotated_{source_stem}.mp4"
        )
    else:
        output_video_path = arguments.output_path
    if (
        source_info["source_type"] == "video_file"
        and output_video_path.resolve() == video_path.resolve()
    ):
        raise ValueError("Output video path must not overwrite the input video.")

    run_provenance = {
        "source_video": (
            str(video_path.resolve())
            if source_info["source_type"] == "video_file"
            else source_info["source"]
        ),
        "input_source_type": source_info["source_type"],
        "detector_model": str(model_path.resolve()),
        "regions_profile": (
            "full_frame" if regions_config_path is None else str(regions_config_path.resolve())
        ),
        "calibration_profile": (
            None
            if calibration_config_path is None
            else str(calibration_config_path.resolve())
        ),
        "safety_zone_profile": (
            None
            if safety_zone_config_path is None
            else str(safety_zone_config_path.resolve())
        ),
        "camera_mode": camera_mode,
        "detection_confidence": arguments.detection_confidence,
        "tracker_high_confidence": arguments.tracker_high_confidence,
        "tracker_low_confidence": arguments.tracker_low_confidence,
        "new_track_confidence": arguments.new_track_confidence,
        "tracker_match_threshold": arguments.tracker_match_threshold,
        "max_missing_seconds": arguments.max_missing_seconds,
        "minimum_trusted_track_frames": arguments.minimum_trusted_track_frames,
        "minimum_confirmed_seconds": arguments.minimum_confirmed_seconds,
        "camera_motion_method": arguments.camera_motion_method,
        "minimum_stationary_seconds": arguments.minimum_stationary_seconds,
        "maximum_stationary_spread_vehicle_widths": (
            arguments.maximum_stationary_spread_vehicle_widths
        ),
        "minimum_restricted_area_frames": arguments.minimum_restricted_area_frames,
    }
    model = YOLO(model_path)
    (
        completed_video_path,
        trajectories,
        trajectory_output_path,
        review_evidence_output_path,
        review_clip_paths,
        top_down_video_path,
        risk_evidence_output_path,
        risk_review_clip_paths,
        run_summary_output_path,
        marshal_briefing_output_path,
        marshal_evidence_package_output_path,
        safety_zone_evidence_output_path,
        safety_candidate_evidence_output_path,
    ) = process_video(
        model=model,
        video_path=video_path,
        output_video_path=output_video_path,
        normalized_track_regions=normalized_track_regions,
        max_missing_seconds=arguments.max_missing_seconds,
        detection_confidence=arguments.detection_confidence,
        tracker_high_confidence=arguments.tracker_high_confidence,
        tracker_low_confidence=arguments.tracker_low_confidence,
        new_track_confidence=arguments.new_track_confidence,
        tracker_match_threshold=arguments.tracker_match_threshold,
        close_approach_ratio=arguments.close_approach_ratio,
        minimum_close_frames=arguments.minimum_close_frames,
        review_pre_seconds=arguments.review_pre_seconds,
        review_post_seconds=arguments.review_post_seconds,
        max_review_clips=arguments.max_review_clips,
        calibration_homography=calibration_homography,
        track_map_size=track_map_size,
        maximum_motion_gap_seconds=arguments.maximum_motion_gap_seconds,
        prediction_horizon_seconds=arguments.prediction_horizon_seconds,
        maximum_predicted_separation_vehicle_widths=(
            arguments.maximum_predicted_separation_vehicle_widths
        ),
        minimum_risk_frames=arguments.minimum_risk_frames,
        minimum_trusted_track_frames=arguments.minimum_trusted_track_frames,
        minimum_confirmed_seconds=arguments.minimum_confirmed_seconds,
        camera_motion_method=arguments.camera_motion_method,
        camera_mode=camera_mode,
        run_provenance=run_provenance,
        max_marshal_briefing_highlights=arguments.max_marshal_briefing_highlights,
        safety_zone_profile=safety_zone_profile,
        safety_zone_profile_path=safety_zone_config_path,
        minimum_stationary_seconds=arguments.minimum_stationary_seconds,
        maximum_stationary_spread_vehicle_widths=(
            arguments.maximum_stationary_spread_vehicle_widths
        ),
        minimum_restricted_area_frames=arguments.minimum_restricted_area_frames,
    )
    print(f"Saved annotated video to: {completed_video_path}")
    print(f"Saved trajectory data to: {trajectory_output_path}")
    print(f"Saved predicted-risk evidence to: {risk_evidence_output_path}")
    print(f"Saved run summary to: {run_summary_output_path}")
    print(f"Saved Marshal briefing to: {marshal_briefing_output_path}")
    print(f"Saved Marshal evidence package to: {marshal_evidence_package_output_path}")
    if safety_zone_evidence_output_path is not None:
        print(f"Saved safety-zone evidence to: {safety_zone_evidence_output_path}")
    print(f"Saved safety candidates to: {safety_candidate_evidence_output_path}")
    print(
        f"Saved {len(risk_review_clip_paths)} candidate-risk clips to: "
        f"{risk_evidence_output_path.parent}"
    )
    print(f"Saved review evidence to: {review_evidence_output_path}")
    print(f"Saved {len(review_clip_paths)} review clips to: {review_evidence_output_path.parent}")
    if calibration_config_path is not None and camera_mode in {"fixed", "moving"}:
        print(
            "Used camera calibration from: "
            f"{calibration_config_path} ({track_map_size[0]} x {track_map_size[1]} map, "
            f"{camera_mode}-camera mode)"
        )
    if top_down_video_path is not None:
        print(f"Saved top-down map video to: {top_down_video_path}")
    elif calibration_config_path is not None:
        print(
            "Skipped spatial analysis because this calibration profile is marked "
            "as unknown camera motion. The annotated camera video remains valid."
        )
    print(f"Recorded {len(trajectories)} trajectory segments.")


if __name__ == "__main__":
    run_video_tracking()
