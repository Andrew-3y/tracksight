import cv2
from src.calibration import project_vehicle_box
from src.camera_motion import CameraReferenceTransformer
from src.detect_vehicles import detect_full_frame_vehicles, filter_boxes_to_regions
from src.evidence_package import create_marshal_evidence_package
from src.regions import scale_normalized_regions
from src.native_tracking import MotionCompensatedVehicleTracker
from src.marshal_briefing import (
    DEFAULT_MAX_MARSHAL_BRIEFING_HIGHLIGHTS,
    build_marshal_briefing,
    save_marshal_briefing,
    select_briefing_events,
)
from src.pipeline import PIPELINE_VERSION
from src.review_clips import create_event_clips, create_review_clips
from src.review_evidence import find_close_approach_evidence, save_review_evidence
from src.risk_assessment import (
    find_predicted_close_approach_evidence,
    save_risk_evidence,
)
from src.spatial_analysis import add_motion_measurements
from src.safety_zones import (
    build_safety_zone_evidence,
    build_safety_zone_observations,
    save_safety_zone_evidence,
)
from src.safety_candidates import (
    DEFAULT_MAXIMUM_STATIONARY_SPREAD_VEHICLE_WIDTHS,
    DEFAULT_MINIMUM_RESTRICTED_AREA_FRAMES,
    DEFAULT_MINIMUM_STATIONARY_SECONDS,
    build_safety_candidate_evidence,
    save_safety_candidate_evidence,
)
from src.track_quality import (
    build_run_summary,
    build_track_quality,
    filter_trusted_trajectories,
    save_run_summary,
)
from src.trajectory import add_trajectory_points, save_trajectories
from src.top_down import create_top_down_canvas, draw_top_down_marker
from src.video_output import create_working_video_writer, finalize_browser_video

def process_video(
    model,
    video_path,
    output_video_path,
    normalized_track_regions,
    max_missing_seconds,
    detection_confidence,
    tracker_high_confidence,
    tracker_low_confidence,
    new_track_confidence,
    tracker_match_threshold,
    close_approach_ratio,
    minimum_close_frames,
    review_pre_seconds,
    review_post_seconds,
    max_review_clips,
    calibration_homography=None,
    track_map_size=None,
    maximum_motion_gap_seconds=0.25,
    prediction_horizon_seconds=1.5,
    maximum_predicted_separation_vehicle_widths=1.5,
    minimum_risk_frames=3,
    minimum_trusted_track_frames=30,
    minimum_confirmed_seconds=0.2,
    camera_motion_method="sparseOptFlow",
    camera_mode="unknown",
    run_provenance=None,
    max_marshal_briefing_highlights=DEFAULT_MAX_MARSHAL_BRIEFING_HIGHLIGHTS,
    safety_zone_profile=None,
    safety_zone_profile_path=None,
    minimum_stationary_seconds=DEFAULT_MINIMUM_STATIONARY_SECONDS,
    maximum_stationary_spread_vehicle_widths=(
        DEFAULT_MAXIMUM_STATIONARY_SPREAD_VEHICLE_WIDTHS
    ),
    minimum_restricted_area_frames=DEFAULT_MINIMUM_RESTRICTED_AREA_FRAMES,
):
    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    track_regions = scale_normalized_regions(
        normalized_track_regions,
        frame_width,
        frame_height,
    )
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    video_writer, working_video_path = create_working_video_writer(
        output_video_path,
        fps,
        (frame_width, frame_height),
    )
    top_down_video_path = None
    top_down_writer = None
    if calibration_homography is not None:
        if track_map_size is None:
            raise ValueError("A track-map size is required when calibration is enabled.")
        top_down_video_path = output_video_path.with_name(
            f"{output_video_path.stem}_top_down.mp4"
        )
        top_down_writer, working_top_down_video_path = create_working_video_writer(
            top_down_video_path,
            fps,
            track_map_size,
        )
    tracker = MotionCompensatedVehicleTracker(
        fps,
        max_missing_seconds,
        tracker_high_confidence,
        tracker_low_confidence,
        new_track_confidence,
        tracker_match_threshold,
        minimum_confirmed_seconds,
        camera_motion_method,
    )
    camera_reference_transformer = (
        CameraReferenceTransformer()
        if camera_mode == "moving" and calibration_homography is not None
        else None
    )
    trajectories = {}
    frame_index = 0
    while True:
        frame_read_success, frame = video_capture.read()
        if not frame_read_success:
            break
        detection_result = detect_full_frame_vehicles(
            model,
            frame,
            detection_confidence,
        )
        region_boxes = filter_boxes_to_regions(
            detection_result.boxes,
            track_regions,
        )
        frame_calibration_homography = calibration_homography
        if camera_reference_transformer is not None:
            frame_calibration_homography = (
                calibration_homography
                @ camera_reference_transformer.update(
                    frame,
                    region_boxes.xyxy.cpu().tolist(),
                )
            )
        tracked_detections = tracker.update(region_boxes, frame)
        trajectories = add_trajectory_points(
            trajectories,
            tracked_detections,
            frame_index,
            fps,
            frame_calibration_homography,
        )
        annotated_frame = frame.copy()
        for detection in tracked_detections:
            if detection["is_estimated"] or not detection["is_confirmed"]:
                continue
            x1, y1, x2, y2 = detection["xyxy"]
            annotation_color = (0, 255, 0)
            cv2.rectangle(annotated_frame, (round(x1), round(y1)), (round(x2), round(y2)), annotation_color, 2)
            track_id = detection["track_id"]
            label = f"ID {track_id}"
            cv2.putText(annotated_frame, label, (round(x1), max(0, round(y1) - 10)), cv2.FONT_HERSHEY_COMPLEX, 0.6, annotation_color, 2)
        video_writer.write(annotated_frame)
        if top_down_writer is not None:
            top_down_frame = create_top_down_canvas(track_map_size)
            for detection in tracked_detections:
                if detection["is_estimated"] or not detection["is_confirmed"]:
                    continue
                track_position = project_vehicle_box(
                    detection["xyxy"],
                    frame_calibration_homography,
                )
                draw_top_down_marker(
                    top_down_frame,
                    detection["track_id"],
                    track_position,
                )
            top_down_writer.write(top_down_frame)
        frame_index += 1
    video_capture.release()
    video_writer.release()
    finalize_browser_video(working_video_path, output_video_path)
    if top_down_writer is not None:
        top_down_writer.release()
        finalize_browser_video(working_top_down_video_path, top_down_video_path)
    trajectories = add_motion_measurements(
        trajectories,
        maximum_motion_gap_seconds,
    )
    track_quality = build_track_quality(
        trajectories,
        minimum_trusted_track_frames,
    )
    trusted_trajectories = filter_trusted_trajectories(
        trajectories,
        track_quality,
    )
    if calibration_homography is None:
        risk_evidence = []
        review_evidence = []
        safety_zone_observations = []
    else:
        risk_evidence = find_predicted_close_approach_evidence(
            trusted_trajectories,
            prediction_horizon_seconds,
            maximum_predicted_separation_vehicle_widths,
            minimum_risk_frames,
        )
        review_evidence = find_close_approach_evidence(
            trusted_trajectories,
            close_approach_ratio,
            minimum_close_frames,
        )
        safety_zone_observations = (
            build_safety_zone_observations(
                trusted_trajectories,
                safety_zone_profile["zones"],
                trusted_track_ids=trusted_trajectories.keys(),
            )
            if safety_zone_profile is not None
            else []
        )
    safety_candidate_evidence = build_safety_candidate_evidence(
        trusted_trajectories,
        spatial_analysis_enabled=calibration_homography is not None,
        safety_zone_profile=safety_zone_profile,
        safety_zone_observations=safety_zone_observations,
        minimum_stationary_seconds=minimum_stationary_seconds,
        maximum_spread_vehicle_widths=maximum_stationary_spread_vehicle_widths,
        minimum_restricted_area_frames=minimum_restricted_area_frames,
    )
    trajectory_output_path = output_video_path.with_name(
        f"{output_video_path.stem}_trajectories.json"
    )
    save_trajectories(trajectories, trajectory_output_path)
    risk_evidence_output_path = output_video_path.with_name(
        f"{output_video_path.stem}_risk_evidence.json"
    )
    review_evidence_output_path = output_video_path.with_name(
        f"{output_video_path.stem}_review_evidence.json"
    )
    review_clips_directory = output_video_path.with_name(
        f"{output_video_path.stem}_review_clips"
    )
    review_clip_paths = create_review_clips(
        output_video_path,
        review_evidence,
        review_clips_directory,
        review_pre_seconds,
        review_post_seconds,
        max_review_clips,
    )
    risk_review_clips_directory = output_video_path.with_name(
        f"{output_video_path.stem}_risk_review_clips"
    )
    risk_review_clip_paths = create_review_clips(
        output_video_path,
        risk_evidence,
        risk_review_clips_directory,
        review_pre_seconds,
        review_post_seconds,
        max_review_clips,
        anchor_frame_key="highest_risk_frame",
        priority_key="predicted_separation_vehicle_widths",
        clip_prefix="risk",
    )
    safety_candidate_clips_directory = output_video_path.with_name(
        f"{output_video_path.stem}_safety_candidate_clips"
    )
    safety_candidate_briefing_events = sorted(
        [
            *safety_candidate_evidence["stationary_vehicle_candidates"],
            *safety_candidate_evidence["restricted_area_entry_candidates"],
            *safety_candidate_evidence["stationary_in_active_racing_area_candidates"],
            *safety_candidate_evidence["against_expected_direction_candidates"],
        ],
        key=lambda event: (
            event["anchor_time_seconds"],
            event["candidate_type"],
            event["track_id"],
        ),
    )[:max_marshal_briefing_highlights]
    create_event_clips(
        output_video_path,
        safety_candidate_briefing_events,
        safety_candidate_clips_directory,
        review_pre_seconds,
        review_post_seconds,
        anchor_frame_key="anchor_frame",
        clip_prefix="safety_candidate",
        clip_path_key="marshal_briefing_clip_path",
    )
    briefing_clip_directory = output_video_path.with_name(
        f"{output_video_path.stem}_marshal_briefing_clips"
    )
    create_event_clips(
        output_video_path,
        select_briefing_events(
            risk_evidence,
            "predicted_separation_vehicle_widths",
            max_marshal_briefing_highlights,
        ),
        briefing_clip_directory,
        review_pre_seconds,
        review_post_seconds,
        anchor_frame_key="highest_risk_frame",
        clip_prefix="predicted_risk",
        clip_path_key="marshal_briefing_clip_path",
    )
    create_event_clips(
        output_video_path,
        select_briefing_events(
            review_evidence,
            "closest_normalized_distance",
            max_marshal_briefing_highlights,
        ),
        briefing_clip_directory,
        review_pre_seconds,
        review_post_seconds,
        clip_prefix="observed_proximity",
        clip_path_key="marshal_briefing_clip_path",
    )
    save_risk_evidence(risk_evidence, risk_evidence_output_path)
    save_review_evidence(review_evidence, review_evidence_output_path)
    safety_zone_evidence_output_path = None
    if safety_zone_profile is not None:
        safety_zone_evidence_output_path = output_video_path.with_name(
            f"{output_video_path.stem}_safety_zone_evidence.json"
        )
        save_safety_zone_evidence(
            build_safety_zone_evidence(
                safety_zone_profile,
                safety_zone_observations,
                profile_path=safety_zone_profile_path,
            ),
            safety_zone_evidence_output_path,
        )
    run_summary_output_path = output_video_path.with_name(
        f"{output_video_path.stem}_run_summary.json"
    )
    run_summary = build_run_summary(
        trajectories,
        track_quality,
        risk_evidence,
        review_evidence,
        pipeline_version=PIPELINE_VERSION,
        spatial_analysis_enabled=calibration_homography is not None,
        processed_frame_count=frame_index,
        frames_per_second=fps,
        run_provenance=run_provenance,
        safety_zone_observation_count=len(safety_zone_observations),
        camera_trust_diagnostics=(
            camera_reference_transformer.get_diagnostics()
            if camera_reference_transformer is not None
            else {
                "method": (
                    "not_available_without_camera_calibration"
                    if calibration_homography is None
                    else "not_required_fixed_camera"
                    if camera_mode == "fixed"
                    else "not_available_for_unknown_camera_mode"
                ),
                "frame_outcomes": {},
            }
        ),
        safety_candidate_counts={
            "stationary_vehicle": len(
                safety_candidate_evidence["stationary_vehicle_candidates"]
            ),
            "restricted_area_entry": len(
                safety_candidate_evidence["restricted_area_entry_candidates"]
            ),
            "stationary_in_active_racing_area": len(
                safety_candidate_evidence["stationary_in_active_racing_area_candidates"]
            ),
            "against_expected_direction": len(
                safety_candidate_evidence["against_expected_direction_candidates"]
            ),
        },
    )
    save_run_summary(run_summary, run_summary_output_path)
    marshal_briefing_output_path = output_video_path.with_name(
        f"{output_video_path.stem}_marshal_briefing.json"
    )
    marshal_briefing = build_marshal_briefing(
        run_summary,
        risk_evidence,
        review_evidence,
        max_marshal_briefing_highlights,
        safety_candidate_evidence=safety_candidate_evidence,
    )
    save_marshal_briefing(marshal_briefing, marshal_briefing_output_path)
    safety_candidate_evidence_output_path = output_video_path.with_name(
        f"{output_video_path.stem}_safety_candidates.json"
    )
    save_safety_candidate_evidence(
        safety_candidate_evidence,
        safety_candidate_evidence_output_path,
    )
    marshal_evidence_package_output_path = output_video_path.with_name(
        f"{output_video_path.stem}_marshal_evidence_package.zip"
    )
    create_marshal_evidence_package(
        marshal_evidence_package_output_path,
        marshal_briefing,
        run_summary,
        risk_evidence,
        review_evidence,
        trajectories,
        safety_zone_evidence=(
            None
            if safety_zone_profile is None
            else build_safety_zone_evidence(
                safety_zone_profile,
                safety_zone_observations,
                profile_path=safety_zone_profile_path,
            )
        ),
        safety_candidate_evidence=safety_candidate_evidence,
    )
    return (
        output_video_path,
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
    )
