import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
TRACKING_OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "tracking"
REQUIRED_REVIEWABLE_RUN_FIELDS = {
    "all_observations",
    "all_track_segments",
    "frames_per_second",
    "observations_per_frame",
    "observed_close_approach_event_count",
    "processed_frame_count",
    "risk_event_count",
    "spatial_analysis_enabled",
    "spatial_analysis_status",
    "track_quality",
    "trusted_track_segments",
}


@st.cache_data(show_spinner="Loading TrackSight evidence...")
def load_json(json_path, modified_time_ns):
    """Load one saved artifact and invalidate the cache when it is replaced."""
    del modified_time_ns
    with Path(json_path).open(encoding="utf-8") as json_file:
        return json.load(json_file)


def load_saved_json(json_path):
    """Load a JSON artifact using its modification time as the cache version."""
    json_path = Path(json_path)
    return load_json(str(json_path), json_path.stat().st_mtime_ns)


def is_reviewable_run_summary(run_summary):
    """Return whether a saved run has the safety fields this dashboard requires.

    The output schema, not a cosmetic pipeline-version label, decides whether a
    run can be shown. This keeps a valid H.264 run reviewable after an
    unrelated application update while still withholding old, incomplete data.
    """
    return (
        isinstance(run_summary, dict)
        and REQUIRED_REVIEWABLE_RUN_FIELDS.issubset(run_summary)
        and isinstance(run_summary["track_quality"], dict)
    )


def format_table_values(values):
    """Convert mixed provenance values to table-safe text for Streamlit."""
    return {
        label: "none" if value is None else str(value)
        for label, value in values.items()
    }


def get_artifact_display_name(path_value):
    """Return a compact artifact name while full paths remain in saved JSON."""
    if not path_value:
        return "none"
    if path_value == "unknown":
        return "unknown"
    return Path(path_value).name


def get_default_evidence_event_index(event_dataframe):
    """Prefer the closest listed event that has a saved review clip."""
    for event_index, event in event_dataframe.iterrows():
        clip = event["clip"]
        if isinstance(clip, str) and Path(clip).is_file():
            return event_index
    return event_dataframe.index[0]


def find_available_runs():
    if not TRACKING_OUTPUT_DIRECTORY.is_dir():
        return []
    current_runs = []
    for trajectory_path in TRACKING_OUTPUT_DIRECTORY.glob("*_trajectories.json"):
        artifacts = get_run_artifacts(trajectory_path)
        if not (
            artifacts["annotated_video"].is_file()
            and artifacts["run_summary_json"].is_file()
        ):
            continue
        with artifacts["run_summary_json"].open(encoding="utf-8") as summary_file:
            run_summary = json.load(summary_file)
        if is_reviewable_run_summary(run_summary):
            current_runs.append(trajectory_path)
    return sorted(current_runs, key=lambda path: path.stat().st_mtime, reverse=True)


def get_run_artifacts(trajectory_path):
    run_name = trajectory_path.name.removesuffix("_trajectories.json")
    return {
        "run_name": run_name,
        "annotated_video": trajectory_path.with_name(f"{run_name}.mp4"),
        "top_down_video": trajectory_path.with_name(f"{run_name}_top_down.mp4"),
        "trajectory_json": trajectory_path,
        "review_evidence_json": trajectory_path.with_name(
            f"{run_name}_review_evidence.json"
        ),
        "risk_evidence_json": trajectory_path.with_name(
            f"{run_name}_risk_evidence.json"
        ),
        "marshal_briefing_json": trajectory_path.with_name(
            f"{run_name}_marshal_briefing.json"
        ),
        "run_summary_json": trajectory_path.with_name(
            f"{run_name}_run_summary.json"
        ),
    }


def build_trajectory_dataframe(trajectories, trusted_track_ids=None):
    rows = []
    for track_id, trajectory in trajectories.items():
        if trusted_track_ids is not None and int(track_id) not in trusted_track_ids:
            continue
        for point in trajectory:
            motion = point.get("motion", {})
            track_position = point.get("track_position")
            rows.append(
                {
                    "track_id": int(track_id),
                    "frame": point["frame_index"],
                    "time_seconds": point["time_seconds"],
                    "confidence": point["confidence"],
                    "motion_status": motion.get("status", "not_calculated"),
                    "speed_track_units_per_second": motion.get(
                        "speed_track_units_per_second"
                    ),
                    "heading_degrees": motion.get("heading_degrees"),
                    "track_x": None if track_position is None else track_position[0],
                    "track_y": None if track_position is None else track_position[1],
                }
            )
    trajectory_dataframe = pd.DataFrame(rows)
    if trajectory_dataframe.empty:
        return trajectory_dataframe

    trajectory_dataframe = trajectory_dataframe.sort_values(
        ["track_id", "frame"]
    ).copy()
    starts_new_segment = trajectory_dataframe["motion_status"].eq("gap_too_large")
    segment_number = starts_new_segment.groupby(
        trajectory_dataframe["track_id"]
    ).cumsum()
    trajectory_dataframe["trajectory_segment"] = (
        trajectory_dataframe["track_id"].astype(str)
        + "-"
        + segment_number.astype(int).astype(str)
    )
    return trajectory_dataframe


def build_event_dataframe(events, risk_events):
    rows = []
    for event in events:
        rows.append(
            {
                "evidence_type": "observed close approach",
                "track_ids": ", ".join(map(str, event["track_ids"])),
                "frame": event["closest_frame"],
                "time_seconds": event["closest_time_seconds"],
                "separation_value": event["closest_normalized_distance"],
                "separation_basis": "average detected-box diagonals",
                "clip": event.get("review_clip_path"),
            }
        )
    for event in risk_events:
        rows.append(
            {
                "evidence_type": "predicted close approach",
                "track_ids": ", ".join(map(str, event["track_ids"])),
                "frame": event["highest_risk_frame"],
                "time_seconds": event["highest_risk_time_seconds"],
                "separation_value": event[
                    "predicted_separation_vehicle_widths"
                ],
                "separation_basis": "average projected vehicle widths",
                "clip": event.get("review_clip_path"),
            }
        )
    return pd.DataFrame(rows)


def render_overview(
    trajectory_data,
    review_events,
    risk_events,
    run_summary,
    trusted_track_ids,
    spatial_analysis_enabled,
):
    trajectory_dataframe = build_trajectory_dataframe(trajectory_data)
    trusted_observation_count = len(
        build_trajectory_dataframe(trajectory_data, trusted_track_ids)
    )
    frames_per_second = run_summary.get("frames_per_second", 0.0)
    video_duration_seconds = (
        run_summary.get("processed_frame_count", 0) / frames_per_second
        if frames_per_second
        else 0.0
    )

    with st.container(horizontal=True):
        st.metric(
            "Continuous ID segments",
            run_summary["all_track_segments"],
            border=True,
            help="One uninterrupted tracker ID segment. This is not a count of unique physical vehicles.",
        )
        st.metric(
            "Safety-eligible segments",
            run_summary["trusted_track_segments"],
            border=True,
            help="Segments with enough consecutive real detections to be eligible for safety analysis.",
        )
        st.metric(
            "Analysed video duration",
            f"{video_duration_seconds:.1f} s",
            border=True,
            help="The duration of video footage analysed in this completed run.",
        )
        st.metric(
            "Candidate-risk events",
            run_summary["risk_event_count"],
            border=True,
            help="Model-supported proximity candidates that need human review; not confirmed incidents.",
        )

    st.header("Review status", icon=":material/fact_check:")
    if not spatial_analysis_enabled:
        st.info(
            "Safety review was not evaluated: this run uses a moving camera without a stable spatial calibration.",
            icon=":material/videocam_off:",
        )
    elif run_summary["risk_event_count"]:
        st.warning(
            "Candidate risks need human review. They are model-supported evidence, not a finding of fault or contact.",
            icon=":material/warning:",
        )
    else:
        st.success(
            "No predicted close-approach candidates met the configured evidence threshold.",
            icon=":material/check_circle:",
        )
    if spatial_analysis_enabled:
        st.caption(
            f"Observed close-approach evidence: "
            f"{run_summary['observed_close_approach_event_count']} event(s). "
            "Only real detections contribute to saved trajectories; estimated boxes are excluded."
        )
    if not trajectory_dataframe.empty and trusted_observation_count == 0:
        st.info(
            "No segment was continuous enough for safety analysis. The video is still saved, but TrackSight will not create candidate-risk evidence from it.",
            icon=":material/info:",
        )
    st.caption(
        "The Overview intentionally omits repeated frame-by-frame detector observations. "
        "Those detailed audit counts remain available in Provenance."
    )


def render_camera_video(artifacts, run_is_quality_assessed):
    st.header("Annotated camera view", icon=":material/videocam:")
    if not run_is_quality_assessed:
        st.warning(
            "Diagnostic preview only: this older run is preserved so you can inspect it, "
            "but its IDs, map, and evidence must not be used as TrackSight analysis.",
            icon=":material/error:",
        )
    if artifacts["annotated_video"].is_file():
        st.video(
            str(artifacts["annotated_video"]),
            alt="Trackside video with detected and tracked vehicles labelled by ID.",
        )
    else:
        st.warning("The annotated video for this run is missing.")


def render_top_down_map(
    artifacts,
    trajectories,
    trusted_track_ids,
    run_is_quality_assessed,
    spatial_analysis_enabled,
    spatial_analysis_status,
):
    st.header("Spatial digital twin", icon=":material/map:")
    if not run_is_quality_assessed:
        st.warning(
            "This map was created before track-quality controls. Rerun the video to create a reviewable spatial map.",
            icon=":material/error:",
        )
        return
    if not spatial_analysis_enabled:
        st.info(
            "Spatial mapping is disabled because this run has no usable camera calibration. "
            "Create a calibration profile before asking TrackSight to produce local-track positions or safety evidence.",
            icon=":material/videocam_off:",
        )
        return
    if spatial_analysis_status == "enabled_with_camera_stabilization":
        st.info(
            "This moving-camera view is aligned to its calibration reference using stable background features before vehicle positions are projected onto the local track map. "
            "The result is review evidence, not surveyed metre-level telemetry.",
            icon=":material/info:",
        )
    if artifacts["top_down_video"].is_file():
        st.video(
            str(artifacts["top_down_video"]),
            alt="Perspective-corrected track map with confirmed vehicle ID markers.",
        )
    else:
        st.info(
            "This run has no top-down video. Run TrackSight with a matching calibration profile to create one."
        )

    trajectory_dataframe = build_trajectory_dataframe(
        trajectories,
        trusted_track_ids,
    )
    calibrated_points = trajectory_dataframe.dropna(subset=["track_x", "track_y"])
    if calibrated_points.empty:
        return

    track_lengths = calibrated_points.groupby("track_id").size()
    default_track_ids = track_lengths.sort_values(ascending=False).head(6).index.tolist()
    selected_track_ids = st.multiselect(
        "Track segments to display",
        options=track_lengths.index.tolist(),
        default=default_track_ids,
        max_selections=6,
        select_all=False,
        help="The map displays up to six of the longest safety-eligible ID segments so overlapping paths remain readable. Remove one before adding another.",
    )
    if not selected_track_ids:
        st.info("Choose one or more track segments to draw the spatial paths.")
        return
    calibrated_points = calibrated_points[
        calibrated_points["track_id"].isin(selected_track_ids)
    ].sort_values(["track_id", "frame"])

    chart = (
        alt.Chart(calibrated_points)
        .mark_line()
        .encode(
            x=alt.X("track_x:Q", title="Local track X"),
            y=alt.Y(
                "track_y:Q",
                title="Local track Y",
                scale=alt.Scale(reverse=True),
            ),
            color=alt.Color("track_id:N", title="Vehicle ID"),
            detail=alt.Detail("trajectory_segment:N"),
            order=alt.Order("frame:Q"),
            tooltip=[
                alt.Tooltip("track_id:N", title="Vehicle ID"),
                alt.Tooltip("time_seconds:Q", title="Time (s)", format=".2f"),
                alt.Tooltip(
                    "speed_track_units_per_second:Q",
                    title="Speed (local units/s)",
                    format=".1f",
                ),
                alt.Tooltip("motion_status:N", title="Motion status"),
            ],
        )
        .properties(height=450)
    )
    st.altair_chart(
        chart,
        width="stretch",
        alt="Calibrated vehicle paths in the local top-down track coordinate system.",
    )
    st.caption(
        "The map shows selected safety-eligible ID segments in local track units. "
        "For fixed views it uses direct perspective calibration; for moving views it first aligns frames to the calibration reference. "
        "It does not claim metres or km/h until a real-world scale reference is supplied."
    )


def render_evidence(
    artifacts,
    review_events,
    risk_events,
    run_is_quality_assessed,
    spatial_analysis_enabled,
):
    st.header("Evidence queue", icon=":material/manage_search:")
    if not run_is_quality_assessed:
        st.warning(
            "This legacy run was created before TrackSight added track-quality controls. Rerun the video before using its evidence queue.",
            icon=":material/error:",
        )
        return
    if not spatial_analysis_enabled:
        st.info(
            "Evidence is unavailable because this moving camera has no stable spatial calibration.",
            icon=":material/videocam_off:",
        )
        return
    event_dataframe = build_event_dataframe(review_events, risk_events)
    if event_dataframe.empty:
        st.info("This run did not generate evidence events.")
        return

    evidence_view = st.segmented_control(
        "Evidence type",
        ["Predicted risk", "Observed proximity"],
        default="Predicted risk",
        help=(
            "Predicted and observed separation use different valid normalizations, "
            "so TrackSight shows them separately instead of ranking unlike units together."
        ),
    )
    if evidence_view == "Predicted risk":
        event_dataframe = event_dataframe[
            event_dataframe["evidence_type"] == "predicted close approach"
        ]
        distance_description = "predicted separation in average projected vehicle widths"
    else:
        event_dataframe = event_dataframe[
            event_dataframe["evidence_type"] == "observed close approach"
        ]
        distance_description = "observed centre separation in average detected-box diagonals"
    event_dataframe = event_dataframe.sort_values("separation_value").head(25)
    if event_dataframe.empty:
        st.info(f"This run did not generate {evidence_view.lower()} evidence.")
        return
    st.caption(
        f"Showing the 25 smallest {distance_description}. "
        "They are prompts for human review, not confirmed incidents."
    )
    st.dataframe(
        event_dataframe.drop(columns="clip"),
        hide_index=True,
        alt="Observed and predicted vehicle proximity evidence for this run.",
    )
    event_indices = event_dataframe.index.tolist()
    default_event_index = get_default_evidence_event_index(event_dataframe)
    selected_event_index = st.selectbox(
        "Open evidence event",
        options=event_indices,
        index=event_indices.index(default_event_index),
        format_func=lambda index: (
            f"{event_dataframe.loc[index, 'evidence_type']} — "
            f"vehicles {event_dataframe.loc[index, 'track_ids']} at "
            f"{event_dataframe.loc[index, 'time_seconds']:.2f}s"
        ),
    )
    selected_event = event_dataframe.loc[selected_event_index]
    st.caption(
        f"Lower separation values mean a closer {distance_description}."
    )
    if isinstance(selected_event["clip"], str) and Path(selected_event["clip"]).is_file():
        clip_path = Path(selected_event["clip"])
        st.video(
            str(clip_path),
            alt="Short annotated video clip supporting the selected evidence event.",
        )
    else:
        st.info(
            "A clip was not generated for this event. The full annotated video remains available in the camera view."
        )


def render_marshal_briefing(artifacts, run_is_quality_assessed):
    """Present saved evidence as a concise briefing without adding new claims."""
    st.header("Marshal briefing", icon=":material/assignment:")
    if not run_is_quality_assessed:
        st.warning(
            "This legacy run is not eligible for a Marshal briefing. Reprocess the video with the current pipeline.",
            icon=":material/error:",
        )
        return
    briefing_path = artifacts["marshal_briefing_json"]
    if not briefing_path.is_file():
        st.info(
            "This completed run predates the Marshal briefing. Reprocess the video to create its saved briefing artifact.",
            icon=":material/info:",
        )
        return

    briefing = load_saved_json(briefing_path)
    st.warning(briefing["disclaimer"], icon=":material/warning:")
    summary = briefing["summary"]
    with st.container(horizontal=True):
        st.metric(
            "Candidate-risk events",
            summary["candidate_risk_event_count"],
            border=True,
        )
        st.metric(
            "Observed-proximity events",
            summary["observed_proximity_event_count"],
            border=True,
        )
        st.metric(
            "Safety-eligible segments",
            summary["trusted_track_segments"],
            border=True,
        )
    st.caption(
        "The two lists stay separate because their separation measurements use different valid normalizations."
    )
    for title, highlights in (
        ("Predicted-risk review highlights", briefing["predicted_risk_highlights"]),
        ("Observed-proximity review highlights", briefing["observed_proximity_highlights"]),
    ):
        st.subheader(title)
        if not highlights:
            st.info(f"No {title.lower()} were saved for this run.")
            continue
        for highlight in highlights:
            track_ids = ", ".join(map(str, highlight["track_ids"]))
            with st.expander(
                f"Vehicles {track_ids} at {highlight['time_seconds']:.2f}s",
                expanded=False,
            ):
                st.write(highlight["review_reason"])
                st.caption(
                    f"Measurement: {highlight['measurement']['value']:.3f} "
                    f"{highlight['measurement']['basis']}."
                )
                st.write(highlight["recommended_human_action"])
                clip_path = highlight.get("supporting_clip")
                if isinstance(clip_path, str) and Path(clip_path).is_file():
                    st.video(
                        clip_path,
                        alt="Annotated camera clip supporting this Marshal briefing item.",
                    )
                else:
                    st.error(
                        "This briefing is incomplete because its required supporting clip is missing. Reprocess the run before review.",
                        icon=":material/error:",
                    )


def render_raw_data(artifacts, run_summary):
    st.header("Evidence provenance", icon=":material/description:")
    st.caption(
        "This data is an auditable record of what the system observed and calculated. It is not raw vehicle telemetry."
    )
    st.table(
        format_table_values(
            {
            "Tracker ID segments": run_summary["all_track_segments"],
            "Trusted safety-analysis segments": run_summary[
                "trusted_track_segments"
            ],
            "Detector positions": run_summary["all_observations"],
            "Processed video frames": run_summary.get("processed_frame_count", "unknown"),
            "Mean detections per frame": (
                f"{run_summary['observations_per_frame']:.1f}"
                if "observations_per_frame" in run_summary
                else "unknown"
            ),
            "Observed proximity events": run_summary[
                "observed_close_approach_event_count"
            ],
            "Predicted-risk events": run_summary["risk_event_count"],
            }
        ),
        border="horizontal",
        width="content",
    )
    run_provenance = run_summary.get("run_provenance")
    if run_provenance:
        st.subheader("Run configuration")
        st.caption(
            "These saved settings identify exactly which video, model, and camera setup produced this evidence."
        )
        st.table(
            format_table_values(
                {
                "Source video": get_artifact_display_name(
                    run_provenance.get("source_video", "unknown")
                ),
                "Detector model": get_artifact_display_name(
                    run_provenance.get("detector_model", "unknown")
                ),
                "Race-area profile": get_artifact_display_name(
                    run_provenance.get("regions_profile", "unknown")
                ),
                "Calibration profile": get_artifact_display_name(
                    run_provenance.get("calibration_profile")
                ),
                "Camera mode": run_provenance.get("camera_mode", "unknown"),
                "Spatial mapping": run_summary.get(
                    "spatial_analysis_status", "unknown"
                ),
                "Camera compensation": run_provenance.get(
                    "camera_motion_method", "none"
                ),
                "Detection confidence": run_provenance.get(
                    "detection_confidence", "unknown"
                ),
                "Tracker match threshold": run_provenance.get(
                    "tracker_match_threshold", "unknown"
                ),
                "Trusted-frame requirement": run_provenance.get(
                    "minimum_trusted_track_frames", "unknown"
                ),
                }
            ),
            border="horizontal",
            width="content",
        )
    for label, artifact_key in (
        ("Download trajectory data", "trajectory_json"),
        ("Download observed-evidence data", "review_evidence_json"),
        ("Download predicted-risk data", "risk_evidence_json"),
        ("Download Marshal briefing", "marshal_briefing_json"),
        ("Download run summary", "run_summary_json"),
    ):
        artifact_path = artifacts[artifact_key]
        if artifact_path.is_file():
            st.download_button(
                label,
                data=artifact_path.read_bytes(),
                file_name=artifact_path.name,
                mime="application/json",
                key=f"download_{artifact_key}",
            )


def main():
    st.set_page_config(
        page_title="TrackSight Marshal",
        page_icon=":material/monitoring:",
        layout="wide",
    )
    st.title("TrackSight Marshal", icon=":material/monitoring:")
    st.caption(
        "Calibrated race-video intelligence with auditable candidate-risk evidence."
    )

    available_runs = find_available_runs()
    if not available_runs:
        st.info(
            "No completed run from the current TrackSight pipeline was found. "
            "Process a video first; older outputs remain on disk for audit but are intentionally withheld."
        )
        return

    with st.sidebar:
        st.header("Run selection", icon=":material/folder_open:")
        selected_trajectory_path = st.selectbox(
            "Completed run",
            available_runs,
            format_func=lambda path: path.name.removesuffix("_trajectories.json"),
        )
        st.caption("The dashboard reads saved outputs; it does not rerun the detector.")

    artifacts = get_run_artifacts(selected_trajectory_path)
    trajectories = load_saved_json(artifacts["trajectory_json"])
    review_events = (
        load_saved_json(artifacts["review_evidence_json"])
        if artifacts["review_evidence_json"].is_file()
        else []
    )
    risk_events = (
        load_saved_json(artifacts["risk_evidence_json"])
        if artifacts["risk_evidence_json"].is_file()
        else []
    )
    run_is_quality_assessed = artifacts["run_summary_json"].is_file()
    if run_is_quality_assessed:
        run_summary = load_saved_json(artifacts["run_summary_json"])
        run_is_quality_assessed = is_reviewable_run_summary(run_summary)
    else:
        run_summary = {
            "all_track_segments": len(trajectories),
            "trusted_track_segments": 0,
            "all_observations": sum(len(points) for points in trajectories.values()),
            "processed_frame_count": 0,
            "frames_per_second": 0.0,
            "observations_per_frame": 0.0,
            "risk_event_count": 0,
            "observed_close_approach_event_count": 0,
            "track_quality": {},
        }
        st.warning(
            "This is an earlier TrackSight run. It was created before the current motion-compensated tracker and browser-compatible video pipeline, so its IDs, map, and evidence queue are withheld.",
            icon=":material/error:",
        )
    if not run_is_quality_assessed:
        st.info(
            "Select a run created by the current pipeline after processing a video again. The old files are preserved for audit, but the dashboard will not turn them into a misleading map or evidence queue.",
            icon=":material/info:",
        )
        render_camera_video(artifacts, run_is_quality_assessed)
        render_raw_data(artifacts, run_summary)
        return
    trusted_track_ids = {
        int(track_id)
        for track_id, quality in run_summary["track_quality"].items()
        if quality["trusted_for_safety_analysis"]
    }

    view = st.segmented_control(
        "Review view",
        [
            "Overview",
            "Camera video",
            "Spatial map",
            "Evidence queue",
            "Marshal briefing",
            "Provenance",
        ],
        default="Overview",
    )
    if view == "Overview":
        render_overview(
            trajectories,
            review_events,
            risk_events,
            run_summary,
            trusted_track_ids,
            run_summary.get("spatial_analysis_enabled", False),
        )
    elif view == "Camera video":
        render_camera_video(artifacts, run_is_quality_assessed)
    elif view == "Spatial map":
        render_top_down_map(
            artifacts,
            trajectories,
            trusted_track_ids,
            run_is_quality_assessed,
            run_summary.get("spatial_analysis_enabled", False),
            run_summary.get("spatial_analysis_status", "disabled_without_camera_calibration"),
        )
    elif view == "Evidence queue":
        render_evidence(
            artifacts,
            review_events,
            risk_events,
            run_is_quality_assessed,
            run_summary.get("spatial_analysis_enabled", False),
        )
    elif view == "Marshal briefing":
        render_marshal_briefing(artifacts, run_is_quality_assessed)
    else:
        render_raw_data(artifacts, run_summary)


if __name__ == "__main__":
    main()
