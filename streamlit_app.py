import json
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from scripts.verify_completed_run import validate_completed_run
from src.dashboard_analysis import (
    get_matching_profile_path,
    list_available_videos,
    make_dashboard_output_path,
    persist_uploaded_video,
    build_tracking_command,
    run_tracking_command,
)
from src.incident_packet import build_incident_packet
from src.incident_threads import build_incident_threads
from src.camera_trust import build_camera_trust_monitor
from src.evidence_chain import build_evidence_chain
from src.race_control_log import (
    RACE_CONTROL_ACTIONS,
    RACE_CONTROL_LOG_DISCLAIMER,
    REQUIRED_REVIEW_CHECKS,
    append_race_control_action,
    build_race_control_action_report,
    load_race_control_action_log,
    write_race_control_action_log,
)
from src.race_control_report import build_race_control_report_pdf
from src.review_priority import (
    INSUFFICIENT_EVIDENCE,
    MONITOR,
    REVIEW_NOW,
    build_review_priority_queue,
)
from src.session_comparison import (
    build_multi_run_risk_map,
    build_session_comparison,
    build_session_snapshot,
)
from src.human_review import (
    HUMAN_REVIEW_DISCLAIMER,
    HUMAN_REVIEW_OUTCOMES,
    build_human_review_report,
    filter_current_human_review_decisions,
    is_current_human_review_decision,
    load_human_review_decisions,
    make_review_event_id,
    save_human_review_decision,
    write_human_review_decisions,
)
from src.review_experience import (
    build_evidence_coverage,
    build_incident_replay_context_points,
    build_incident_replay_points,
    build_incident_replay_summary,
)
from src.approach_reconstruction import build_approach_reconstruction
from src.evaluation_report import build_evaluation_report
from src.marshal_handover import build_marshal_handover
from src.learning_queue import LEARNING_QUEUE_DISCLAIMER, build_learning_export, build_learning_queue
from src.multi_camera import MULTI_CAMERA_DISCLAIMER, build_multi_camera_playback, validate_multi_camera_manifest
from src.race_control_export import RACE_CONTROL_EXPORT_DISCLAIMER, build_race_control_timeline_export
from src.telemetry import TELEMETRY_DISCLAIMER, correlate_telemetry_events, parse_telemetry_csv
from src.decision_ladder import DECISION_LADDER_DISCLAIMER, build_safety_decision_ladder
from src.event_correlation import EVENT_CORRELATION_DISCLAIMER, build_event_correlation
from src.after_action_report import build_after_action_report_pdf
from src.review_corrections import (
    CORRECTION_TYPES,
    REVIEW_CORRECTIONS_DISCLAIMER,
    append_review_correction,
    build_correction_learning_export,
    load_review_corrections,
    write_review_corrections,
)
from src.review_ledger import (
    DECISION_LEDGER_DISCLAIMER,
    SECOND_OPINION_DISCLAIMER,
    SECOND_OPINION_TYPES,
    append_second_opinion,
    build_decision_ledger,
    load_second_opinions,
    write_second_opinions,
)
from src.evidence_integrity import (
    EVIDENCE_INTEGRITY_DISCLAIMER,
    build_evidence_integrity,
    build_integrity_manifest,
)
from src.operational_coverage import (
    OPERATIONAL_COVERAGE_DISCLAIMER,
    build_operational_coverage_timeline,
    summarise_operational_coverage,
)
from src.grounded_assistant import (
    GROUNDED_ASSISTANT_DISCLAIMER,
    GROUNDED_QUESTIONS,
    build_grounded_case_answer,
)
from src.risk_hotspots import (
    build_calibrated_event_locations,
    build_candidate_hotspots,
)
from src.safety_zones import SAFETY_ZONE_EVIDENCE_SCHEMA_VERSION
from src.safety_candidates import (
    SAFETY_CANDIDATE_EVIDENCE_SCHEMA_VERSION,
    get_safety_candidates,
)

PROJECT_ROOT = Path(__file__).resolve().parent
TRACKING_OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "tracking"
RAW_VIDEO_DIRECTORY = PROJECT_ROOT / "data" / "raw"
CONFIG_DIRECTORY = PROJECT_ROOT / "configs"
MODEL_DIRECTORY = PROJECT_ROOT / "models"
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

REVIEW_VIEW_LABELS = {
    "Overview": "Review · Overview",
    "Camera video": "Review · Camera video",
    "Evidence quality": "Review · Evidence quality",
    "Camera trust monitor": "Review · Camera trust monitor",
    "Operational coverage": "Review · Operational coverage",
    "Evaluation and limits": "Review · Evaluation and limits",
    "Provenance": "Review · Provenance",
    "Evidence integrity": "Audit · Evidence integrity",
    "Spatial map": "Spatial analysis · Map",
    "Evidence queue": "Spatial analysis · Evidence queue",
    "Race-control timeline": "Spatial analysis · Timeline",
    "Incident replay": "Spatial analysis · Incident replay",
    "Candidate hotspots": "Spatial analysis · Candidate hotspots",
    "Incident threads": "Spatial analysis · Incident threads",
    "Safety command center": "Case review · Safety command center",
    "Live race-control simulation": "Case review · Live race-control simulation",
    "Race-control desk": "Case review · Race-control desk",
    "Human review decisions": "Case review · Human review decisions",
    "Reviewer corrections": "Case review · Reviewer corrections",
    "Learning queue": "Case review · Learning queue",
    "Grounded case assistant": "Case review · Grounded case assistant",
    "Telemetry correlation": "Case review · Telemetry correlation",
    "Multi-camera review": "Case review · Multi-camera review",
    "Race-control export": "Case review · Race-control export",
    "Marshal handover": "Case review · Marshal handover",
    "Independent second review": "Audit · Independent second review",
    "Decision ledger": "Audit · Decision ledger",
    "Marshal briefing": "Audit · Marshal briefing",
    "Safety zones": "Policy review · Safety zones",
    "Safety candidates": "Policy review · Safety candidates",
    "Session comparison": "Across runs · Session comparison",
    "Recurring review map": "Across runs · Recurring review map",
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
        "marshal_evidence_package": trajectory_path.with_name(
            f"{run_name}_marshal_evidence_package.zip"
        ),
        "human_review_decisions_json": trajectory_path.with_name(
            f"{run_name}_human_review_decisions.json"
        ),
        "review_corrections_json": trajectory_path.with_name(
            f"{run_name}_review_corrections.json"
        ),
        "second_opinions_json": trajectory_path.with_name(
            f"{run_name}_second_opinions.json"
        ),
        "race_control_action_log_json": trajectory_path.with_name(
            f"{run_name}_race_control_action_log.json"
        ),
        "safety_zone_evidence_json": trajectory_path.with_name(
            f"{run_name}_safety_zone_evidence.json"
        ),
        "safety_candidate_evidence_json": trajectory_path.with_name(
            f"{run_name}_safety_candidates.json"
        ),
        "run_summary_json": trajectory_path.with_name(
            f"{run_name}_run_summary.json"
        ),
    }


def get_available_review_views(
    artifacts,
    review_events,
    risk_events,
    run_summary,
    safety_zone_evidence,
    safety_candidate_evidence,
    available_run_count,
):
    """Return only dashboard views supported by this run's saved evidence.

    A completed video can be valid for camera review while having no calibrated
    spatial evidence or candidate events. Hiding tools that cannot add useful
    information is clearer than sending a reviewer to an empty-state screen.
    """
    views = [
        "Overview",
        "Camera video",
        "Evidence quality",
        "Camera trust monitor",
        "Operational coverage",
        "Evaluation and limits",
        "Provenance",
        "Evidence integrity",
    ]
    has_events = bool(review_events or risk_events)
    has_spatial_evidence = bool(run_summary.get("spatial_analysis_enabled"))

    if has_spatial_evidence:
        views.append("Spatial map")
    if has_events and has_spatial_evidence:
        views.extend(
            [
                "Evidence queue",
                "Race-control timeline",
                "Incident replay",
                "Candidate hotspots",
                "Incident threads",
            ]
        )
    if has_events:
        views.extend(
            [
                "Safety command center",
                "Race-control desk",
                "Human review decisions",
                "Reviewer corrections",
                "Learning queue",
                "Grounded case assistant",
                "Telemetry correlation",
                "Multi-camera review",
                "Race-control export",
                "Marshal handover",
            ]
        )
        if artifacts["annotated_video"].is_file():
            views.append("Live race-control simulation")
    if safety_zone_evidence:
        views.append("Safety zones")
    if safety_candidate_evidence:
        views.append("Safety candidates")
    if (
        (has_events or safety_candidate_evidence)
        and artifacts["marshal_briefing_json"].is_file()
        and artifacts["marshal_evidence_package"].is_file()
    ):
        views.append("Marshal briefing")
    if any(
        artifacts[artifact_key].is_file()
        for artifact_key in (
            "human_review_decisions_json",
            "race_control_action_log_json",
            "review_corrections_json",
            "second_opinions_json",
        )
    ):
        views.append("Decision ledger")
    if artifacts["human_review_decisions_json"].is_file():
        views.append("Independent second review")
    if has_events and available_run_count > 1:
        views.extend(["Session comparison", "Recurring review map"])
    return views


def get_default_dashboard_run_label(video_path):
    """Return a unique-looking, editable run label for one dashboard launch."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{Path(video_path).stem}_dashboard_{timestamp}"


def render_run_analysis_launcher():
    """Start the established command-line pipeline from a guided local form."""
    st.header("Run analysis", icon=":material/play_circle:")
    st.write(
        "Create a new, auditable TrackSight run. The dashboard uses the same "
        "validated pipeline as the command-line runner; it does not use a "
        "second detector or different thresholds."
    )
    source_mode = st.segmented_control(
        "Video source",
        ["Project video", "Upload video"],
        default="Project video",
        key="analysis_source_mode",
    )
    available_videos = list_available_videos(RAW_VIDEO_DIRECTORY)
    selected_video_path = None
    uploaded_video = None
    if source_mode == "Project video":
        if not available_videos:
            st.error("No supported videos were found in data/raw.")
            return
        selected_video_path = st.selectbox(
            "Project video",
            available_videos,
            format_func=lambda video_path: video_path.name,
            key="analysis_project_video",
            help="Videos already stored in this project. Select one for a reproducible run.",
        )
        st.caption(f"Source: {selected_video_path.name}")
    else:
        uploaded_video = st.file_uploader(
            "Upload a local video",
            type=["mp4", "webm", "mov", "mkv", "avi"],
            key="analysis_uploaded_video",
            help="The selected file is copied into data/raw/dashboard_uploads only when analysis starts.",
        )
        if uploaded_video is not None:
            st.caption(
                f"Ready to copy locally: {uploaded_video.name} "
                f"({uploaded_video.size / 1_000_000:.1f} MB)"
            )

    selected_video_name = (
        selected_video_path.name
        if selected_video_path is not None
        else (uploaded_video.name if uploaded_video is not None else "uploaded_video")
    )
    matching_regions_profile = (
        get_matching_profile_path(selected_video_path, CONFIG_DIRECTORY)
        if selected_video_path is not None
        else None
    )
    matching_calibration_profile = (
        get_matching_profile_path(
            selected_video_path,
            CONFIG_DIRECTORY,
            "_calibration",
        )
        if selected_video_path is not None
        else None
    )
    matching_safety_zone_profile = (
        get_matching_profile_path(
            selected_video_path,
            CONFIG_DIRECTORY,
            "_safety_zones",
        )
        if selected_video_path is not None
        else None
    )
    available_models = sorted(MODEL_DIRECTORY.glob("*.pt"))
    if not available_models:
        st.error("No detector model was found in models.")
        return
    default_model_index = next(
        (
            index
            for index, model_path in enumerate(available_models)
            if model_path.name == "yolo26m.pt"
        ),
        0,
    )

    with st.container(border=True):
        st.subheader("Analysis setup", icon=":material/tune:")
        if selected_video_path is None:
            st.info(
                "An uploaded video starts with the full frame and no spatial calibration. "
                "A calibration is camera-specific, so TrackSight will not reuse one from another video.",
                icon=":material/info:",
            )
        elif matching_calibration_profile is None:
            st.info(
                "No matching camera calibration is available. The camera video will still be analysed, "
                "but spatial mapping and safety evidence remain disabled until this camera is calibrated.",
                icon=":material/videocam_off:",
            )

        with st.form("analysis_launcher_form"):
            run_label = st.text_input(
                "Run label",
                value=get_default_dashboard_run_label(selected_video_name),
                max_chars=80,
                help="Creates a new output name. Existing evidence is never overwritten.",
            )
            model_path = st.selectbox(
                "Detector model",
                available_models,
                index=default_model_index,
                format_func=lambda path: path.name,
                help="The local detector used for this run.",
            )
            regions_config_path = st.selectbox(
                "Race-area profile",
                [matching_regions_profile, None]
                if matching_regions_profile is not None
                else [None],
                format_func=lambda path: (
                    "Use the matching race-area profile"
                    if path is not None
                    else "Use the full video frame"
                ),
                help="Only a profile explicitly named for this source video is offered.",
            )
            calibration_config_path = st.selectbox(
                "Camera calibration",
                [matching_calibration_profile, None]
                if matching_calibration_profile is not None
                else [None],
                format_func=lambda path: (
                    "Use the matching camera calibration"
                    if path is not None
                    else "No calibration (camera-only analysis)"
                ),
                help="Calibration is specific to one camera setup and enables the local spatial map.",
            )
            safety_zone_config_path = st.selectbox(
                "Safety-zone profile",
                [matching_safety_zone_profile, None]
                if matching_safety_zone_profile is not None
                else [None],
                format_func=lambda path: (
                    "Use the matching local-track safety zones"
                    if path is not None
                    else "No safety zones for this run"
                ),
                help="Zones are optional, camera-specific local-track polygons. TrackSight never reuses zones from another video.",
            )
            start_analysis = st.form_submit_button(
                "Run TrackSight analysis",
                type="primary",
                icon=":material/play_arrow:",
            )

    st.caption(
        "Analysis runs locally and can take several minutes. On success, TrackSight verifies "
        "the saved video, trajectories, evidence, and run summary before opening the result."
    )
    if not start_analysis:
        return

    try:
        if selected_video_path is None:
            if uploaded_video is None:
                raise ValueError("Choose a project video or upload a video before starting analysis.")
            selected_video_path = persist_uploaded_video(
                uploaded_video.name,
                uploaded_video.getvalue(),
                RAW_VIDEO_DIRECTORY / "dashboard_uploads",
            )
        output_path = make_dashboard_output_path(TRACKING_OUTPUT_DIRECTORY, run_label)
        command = build_tracking_command(
            PROJECT_ROOT,
            selected_video_path,
            model_path,
            output_path,
            regions_config_path=regions_config_path,
            calibration_config_path=calibration_config_path,
            safety_zone_config_path=safety_zone_config_path,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        st.error(str(error), icon=":material/error:")
        return

    with st.status("TrackSight analysis is running", expanded=True) as status:
        st.write(f"Preparing {selected_video_path.name} with {model_path.name}.")
        st.write("Detecting, tracking, and recording review evidence. Keep this page open.")
        try:
            run_tracking_command(command, PROJECT_ROOT)
            st.write("Verifying the completed video and its saved evidence artifacts.")
            verification = validate_completed_run(output_path)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            status.update(
                label="TrackSight analysis did not complete",
                state="error",
                expanded=True,
            )
            st.error(str(error), icon=":material/error:")
            return
        status.write(
            f"Verified {verification['track_segments']} track segments and "
            f"{verification['observations']} detector positions."
        )
        status.update(
            label="TrackSight analysis completed and verified",
            state="complete",
            expanded=False,
        )

    st.session_state["dashboard_pending_review_run"] = output_path.with_name(
        f"{output_path.stem}_trajectories.json"
    )
    st.rerun()


def build_trajectory_dataframe(trajectories, trusted_track_ids=None):
    rows = []
    for track_id, trajectory in trajectories.items():
        if trusted_track_ids is not None and int(track_id) not in trusted_track_ids:
            continue
        for point in trajectory:
            motion = point.get("motion", {})
            track_position = point.get("track_position")
            center = point.get("center")
            camera_x = (
                center[0]
                if isinstance(center, (list, tuple)) and len(center) == 2
                else None
            )
            camera_y = (
                center[1]
                if isinstance(center, (list, tuple)) and len(center) == 2
                else None
            )
            rows.append(
                {
                    "track_id": int(track_id),
                    "frame": point["frame_index"],
                    "time_seconds": point["time_seconds"],
                    "confidence": point["confidence"],
                    "camera_x": camera_x,
                    "camera_y": camera_y,
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


def build_race_control_timeline_dataframe(review_events, risk_events):
    """Create one time-ordered review row per saved evidence event.

    Observed and predicted evidence stay in separate timeline lanes because
    their measurement values use different valid normalizations. The timeline
    is for locating review material, not for ranking unlike measurements.
    """
    rows = []
    for event_index, event in enumerate(risk_events):
        rows.append(
            {
                "event_key": f"predicted-{event_index}",
                "review_event_id": make_review_event_id(
                    "Predicted risk",
                    event["track_ids"],
                    event["highest_risk_frame"],
                ),
                "evidence_type": "Predicted risk",
                "track_ids": ", ".join(map(str, event["track_ids"])),
                "track_id_values": list(event["track_ids"]),
                "frame": event["highest_risk_frame"],
                "time_seconds": event["highest_risk_time_seconds"],
                "measurement_value": event[
                    "predicted_separation_vehicle_widths"
                ],
                "measurement_basis": "average projected vehicle widths",
                "clip": event.get("marshal_briefing_clip_path")
                or event.get("review_clip_path"),
            }
        )
    for event_index, event in enumerate(review_events):
        rows.append(
            {
                "event_key": f"observed-{event_index}",
                "review_event_id": make_review_event_id(
                    "Observed proximity",
                    event["track_ids"],
                    event["closest_frame"],
                ),
                "evidence_type": "Observed proximity",
                "track_ids": ", ".join(map(str, event["track_ids"])),
                "track_id_values": list(event["track_ids"]),
                "frame": event["closest_frame"],
                "time_seconds": event["closest_time_seconds"],
                "measurement_value": event["closest_normalized_distance"],
                "measurement_basis": "average detected-box diagonals",
                "clip": event.get("marshal_briefing_clip_path")
                or event.get("review_clip_path"),
            }
        )
    return pd.DataFrame(rows)


def filter_race_control_timeline(
    event_dataframe,
    selected_evidence_types,
    selected_track_ids,
    selected_time_range,
):
    """Filter timeline rows by evidence type, involved vehicle, and video time."""
    if event_dataframe.empty:
        return event_dataframe
    filtered_events = event_dataframe[
        event_dataframe["evidence_type"].isin(selected_evidence_types)
    ]
    if selected_track_ids:
        selected_track_id_set = set(selected_track_ids)
        filtered_events = filtered_events[
            filtered_events["track_id_values"].map(
                lambda event_track_ids: bool(
                    selected_track_id_set.intersection(event_track_ids)
                )
            )
        ]
    start_time, end_time = selected_time_range
    return filtered_events[
        filtered_events["time_seconds"].between(start_time, end_time)
    ].sort_values("time_seconds")


def get_default_timeline_event_key(event_dataframe):
    """Choose the first timeline event with a saved clip when possible."""
    for _, event in event_dataframe.iterrows():
        clip_path = event["clip"]
        if isinstance(clip_path, str) and Path(clip_path).is_file():
            return event["event_key"]
    return event_dataframe.iloc[0]["event_key"]


def get_source_evidence_event(event_key, review_events, risk_events):
    """Return the persisted source record behind one dashboard timeline row."""
    evidence_prefix, separator, event_index_text = str(event_key).partition("-")
    if not separator or not event_index_text.isdigit():
        raise ValueError("The selected evidence event key is invalid.")
    event_index = int(event_index_text)
    if evidence_prefix == "predicted":
        return risk_events[event_index]
    if evidence_prefix == "observed":
        return review_events[event_index]
    raise ValueError("The selected evidence event type is unsupported.")


def get_human_review_event_summary(event):
    """Return the source-evidence fields a decision must match to remain valid."""
    return {
        "evidence_type": event["evidence_type"],
        "track_ids": list(event["track_id_values"]),
        "frame": int(event["frame"]),
        "time_seconds": float(event["time_seconds"]),
        "measurement_value": float(event["measurement_value"]),
        "measurement_basis": event["measurement_basis"],
    }


def build_human_review_queue(event_dataframe, decisions):
    """Attach only still-matching human decisions to the source evidence rows."""
    queue_rows = []
    saved_decisions = decisions.get("decisions", {})
    for _, event in event_dataframe.iterrows():
        event_summary = get_human_review_event_summary(event)
        saved_decision = saved_decisions.get(event["review_event_id"])
        if is_current_human_review_decision(saved_decision, event_summary):
            outcome = saved_decision["outcome"]
            reviewer_name = saved_decision["reviewer_name"]
            updated_at = saved_decision["updated_at"]
        else:
            outcome = "Unreviewed"
            reviewer_name = "—"
            updated_at = "—"
        queue_rows.append(
            {
                **event.to_dict(),
                "review_status": outcome,
                "reviewer_name": reviewer_name,
                "review_updated_at": updated_at,
            }
        )
    return pd.DataFrame(queue_rows).sort_values(
        ["review_status", "time_seconds"],
        key=lambda series: (
            series.map(
                {
                    "Escalate": 0,
                    "Needs review": 1,
                    "Unreviewed": 2,
                    "Cleared": 3,
                }
            )
            if series.name == "review_status"
            else series
        ),
    )


def get_human_review_reason(event):
    """Explain why an evidence item was surfaced without turning it into a ruling."""
    vehicle_label = f"Track IDs {event['track_ids']}"
    if event["evidence_type"] == "Predicted risk":
        return (
            f"{vehicle_label} had a predicted closest separation of "
            f"{event['measurement_value']:.3f} {event['measurement_basis']} "
            f"at {event['time_seconds']:.2f}s."
        )
    return (
        f"{vehicle_label} had an observed centre separation of "
        f"{event['measurement_value']:.3f} {event['measurement_basis']} "
        f"at {event['time_seconds']:.2f}s."
    )


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


def render_race_control_timeline(
    artifacts,
    review_events,
    risk_events,
    run_is_quality_assessed,
    spatial_analysis_enabled,
    run_summary,
):
    """Let a reviewer find saved evidence along one completed video timeline."""
    st.header("Race-control timeline", icon=":material/timeline:")
    if not run_is_quality_assessed:
        st.warning(
            "This legacy run was created before TrackSight added track-quality controls. Rerun the video before using its timeline.",
            icon=":material/error:",
        )
        return
    if not spatial_analysis_enabled:
        st.info(
            "The timeline is unavailable because this moving camera has no stable spatial calibration.",
            icon=":material/videocam_off:",
        )
        return

    event_dataframe = build_race_control_timeline_dataframe(
        review_events,
        risk_events,
    )
    if event_dataframe.empty:
        st.info("This run did not generate evidence events for the timeline.")
        return

    video_duration_seconds = (
        run_summary["processed_frame_count"] / run_summary["frames_per_second"]
        if run_summary["frames_per_second"]
        else event_dataframe["time_seconds"].max()
    )
    evidence_types = ["Predicted risk", "Observed proximity"]
    available_track_ids = sorted(
        {
            track_id
            for event_track_ids in event_dataframe["track_id_values"]
            for track_id in event_track_ids
        }
    )
    selected_evidence_types = st.multiselect(
        "Evidence types",
        options=evidence_types,
        default=["Predicted risk"],
        key="timeline_evidence_types",
        help="The timeline starts with candidate-risk events. Add observed proximity when you need broader context; the two types use different valid units.",
    )
    selected_track_ids = st.multiselect(
        "Vehicles involved",
        options=available_track_ids,
        default=[],
        key="timeline_track_ids",
        help="Leave empty to show events involving any vehicle.",
    )
    selected_time_range = st.slider(
        "Video time window",
        min_value=0.0,
        max_value=float(video_duration_seconds),
        value=(0.0, float(video_duration_seconds)),
        step=0.1,
        format="%.1f s",
        key="timeline_time_range",
        help="Narrow the completed video to the period you want to review.",
    )
    filtered_events = filter_race_control_timeline(
        event_dataframe,
        selected_evidence_types,
        selected_track_ids,
        selected_time_range,
    )
    if filtered_events.empty:
        st.info("No evidence events match these filters.")
        return

    st.caption(
        "The timeline starts with candidate-risk events. Add observed proximity for broader context, then use the filters to locate an evidence cluster and choose the exact labelled event below. This timeline does not confirm contact, blame, or a rule violation."
    )
    timeline_chart = (
        alt.Chart(filtered_events)
        .mark_point(filled=True, size=180)
        .encode(
            x=alt.X(
                "time_seconds:Q",
                title="Video time (seconds)",
                scale=alt.Scale(domain=[0, video_duration_seconds]),
            ),
            y=alt.Y("evidence_type:N", title=None, sort=evidence_types),
            color=alt.Color(
                "evidence_type:N",
                title="Evidence type",
                scale=alt.Scale(
                    domain=evidence_types,
                    range=["#ff9f1c", "#4cc9f0"],
                ),
            ),
            shape=alt.Shape("evidence_type:N", title="Evidence type"),
            tooltip=[
                alt.Tooltip("evidence_type:N", title="Evidence"),
                alt.Tooltip("track_ids_label:N", title="Vehicles"),
                alt.Tooltip("time_seconds:Q", title="Time (s)", format=".2f"),
                alt.Tooltip("frame:Q", title="Frame"),
                alt.Tooltip(
                    "measurement_value:Q",
                    title="Separation",
                    format=".3f",
                ),
                alt.Tooltip("measurement_basis:N", title="Measurement basis"),
            ],
        )
        .properties(height=180)
    )
    st.altair_chart(
        timeline_chart,
        width="stretch",
        alt="Timeline of observed and predicted vehicle proximity evidence across the completed race video.",
    )
    available_event_keys = filtered_events["event_key"].tolist()
    selected_event_key = get_default_timeline_event_key(filtered_events)
    selected_event = st.selectbox(
        "Open timeline event",
        options=available_event_keys,
        index=available_event_keys.index(selected_event_key),
        key="timeline_event_selector",
        format_func=lambda event_key: (
            f"{filtered_events.loc[filtered_events['event_key'].eq(event_key), 'evidence_type'].iloc[0]} "
            f"— vehicles {filtered_events.loc[filtered_events['event_key'].eq(event_key), 'track_ids'].iloc[0]} at "
            f"{filtered_events.loc[filtered_events['event_key'].eq(event_key), 'time_seconds'].iloc[0]:.2f}s"
        ),
    )
    selected_event = filtered_events[
        filtered_events["event_key"].eq(selected_event)
    ].iloc[0]
    st.caption(
        f"{selected_event['evidence_type']} at {selected_event['time_seconds']:.2f}s: "
        f"{selected_event['measurement_value']:.3f} "
        f"{selected_event['measurement_basis']}."
    )
    clip_path = selected_event["clip"]
    if isinstance(clip_path, str) and Path(clip_path).is_file():
        st.video(
            clip_path,
            alt="Annotated camera clip supporting the selected race-control timeline event.",
        )
    else:
        st.info(
            "No separate short clip was saved for this lower-priority event. The full annotated camera video opens at its evidence time.",
            icon=":material/info:",
        )
        st.video(
            str(artifacts["annotated_video"]),
            start_time=max(0.0, selected_event["time_seconds"] - 2.0),
            alt="Annotated camera video starting shortly before the selected timeline event.",
        )


def render_incident_replay(
    artifacts,
    trajectories,
    review_events,
    risk_events,
    run_is_quality_assessed,
    spatial_analysis_enabled,
):
    """Present one saved evidence event as an explainable two-vehicle replay."""
    st.header("Decision-ready incident replay", icon=":material/play_circle:")
    st.warning(HUMAN_REVIEW_DISCLAIMER, icon=":material/warning:")
    if not run_is_quality_assessed:
        st.warning(
            "This legacy run was created before TrackSight added track-quality controls. Rerun the video before using incident replay.",
            icon=":material/error:",
        )
        return
    if not spatial_analysis_enabled:
        st.info(
            "Incident replay is unavailable because this moving camera has no stable spatial calibration. "
            "The annotated camera view remains available, but TrackSight will not present uncalibrated movement as spatial safety evidence.",
            icon=":material/videocam_off:",
        )
        return

    event_dataframe = build_race_control_timeline_dataframe(
        review_events,
        risk_events,
    )
    if event_dataframe.empty:
        st.info("This run did not generate an evidence event to replay.")
        return
    event_dataframe = event_dataframe.sort_values("time_seconds")
    event_keys = event_dataframe["event_key"].tolist()
    default_event_key = get_default_timeline_event_key(event_dataframe)
    selected_event_key = st.selectbox(
        "Evidence event to replay",
        options=event_keys,
        index=event_keys.index(default_event_key),
        key="incident_replay_event_selector",
        format_func=lambda event_key: (
            f"{event_dataframe.loc[event_dataframe['event_key'].eq(event_key), 'evidence_type'].iloc[0]} "
            f"— vehicles {event_dataframe.loc[event_dataframe['event_key'].eq(event_key), 'track_ids'].iloc[0]} at "
            f"{event_dataframe.loc[event_dataframe['event_key'].eq(event_key), 'time_seconds'].iloc[0]:.2f}s"
        ),
        help="Each option is a saved evidence record, not an automated incident decision.",
    )
    selected_event = event_dataframe[
        event_dataframe["event_key"].eq(selected_event_key)
    ].iloc[0]

    decision = None
    try:
        decisions = load_human_review_decisions(artifacts["human_review_decisions_json"])
        saved_decision = decisions["decisions"].get(selected_event["review_event_id"])
        if is_current_human_review_decision(
            saved_decision,
            get_human_review_event_summary(selected_event),
        ):
            decision = saved_decision
    except ValueError:
        st.info(
            "A saved human-review decision file could not be read. The original evidence replay remains available.",
            icon=":material/info:",
        )

    decision_status = decision["outcome"] if decision else "Unreviewed"
    with st.container(horizontal=True):
        st.metric("Evidence type", selected_event["evidence_type"], border=True)
        st.metric("Video time", f"{selected_event['time_seconds']:.2f} s", border=True)
        st.metric("Vehicles", selected_event["track_ids"], border=True)
        st.metric(
            "Human review",
            decision_status,
            border=True,
            help="A human workflow status. It is never an automated TrackSight verdict.",
        )

    with st.container(border=True):
        st.subheader("Why TrackSight surfaced this", icon=":material/manage_search:")
        st.write(get_human_review_reason(selected_event))
        st.caption(
            "The replay uses a two-second observed-history window and one second after the saved event. "
            "It shows only actual stored observations; it does not interpolate across a detector gap."
        )
        if decision:
            st.success(
                f"Human decision: {decision['outcome']} by {decision['reviewer_name']}.",
                icon=":material/fact_check:",
            )
            st.write(decision["review_note"])
        else:
            st.info(
                "No current human decision is recorded. Use Human review decisions to record a named, auditable outcome.",
                icon=":material/edit_note:",
            )

    source_event = get_source_evidence_event(
        selected_event["event_key"],
        review_events,
        risk_events,
    )
    reconstruction = build_approach_reconstruction(
        trajectories,
        source_event,
        selected_event["evidence_type"],
    )
    reconstruction_summary = reconstruction["summary"]
    with st.container(border=True):
        st.subheader(
            "Evidence-window approach reconstruction",
            icon=":material/timeline:",
        )
        st.caption(
            "This is a factual account of the saved evidence window, not an incident reconstruction or a finding of contact.",
        )
        with st.container(horizontal=True):
            st.metric(
                "Paired real observations",
                reconstruction_summary["observation_count"],
                border=True,
            )
            st.metric(
                "Missing pair frames",
                reconstruction_summary["missing_pair_frames"],
                border=True,
                help="A missing pair frame is omitted from the trace; TrackSight never fills it with an estimated position.",
            )
            if reconstruction_summary["status"] == "available":
                trend_labels = {
                    "net_closing": "Net closing",
                    "net_separating": "Net separating",
                    "no_material_net_change": "No material change",
                }
                st.metric(
                    "Observed net trend",
                    trend_labels[reconstruction_summary["net_trend"]],
                    border=True,
                    help="Net trend compares the first and last paired real observations in the saved evidence window.",
                )
        timeline_dataframe = pd.DataFrame(reconstruction["timeline"])
        st.dataframe(
            timeline_dataframe,
            column_config={
                "moment": "Saved evidence moment",
                "frame": st.column_config.NumberColumn("Frame", format="%d"),
                "time_seconds": st.column_config.NumberColumn("Video time (s)", format="%.2f"),
                "basis": "Why it is shown",
            },
            hide_index=True,
            alt="Start, anchor, and end moments from the selected saved evidence window.",
        )
        if reconstruction_summary["status"] == "available":
            trace_dataframe = pd.DataFrame(reconstruction["trace"])
            trace_chart = (
                alt.Chart(trace_dataframe)
                .mark_line(point=True)
                .encode(
                    x=alt.X("time_seconds:Q", title="Video time (s)"),
                    y=alt.Y(
                        "separation_vehicle_widths:Q",
                        title="Observed separation (average projected vehicle widths)",
                    ),
                    tooltip=[
                        alt.Tooltip("frame:Q", title="Frame"),
                        alt.Tooltip("time_seconds:Q", title="Video time (s)", format=".2f"),
                        alt.Tooltip(
                            "separation_vehicle_widths:Q",
                            title="Observed separation (vehicle widths)",
                            format=".3f",
                        ),
                    ],
                )
            )
            st.altair_chart(
                trace_chart.properties(height=240),
                width="stretch",
                alt="Observed separation between the selected two vehicle ID segments across paired real calibrated source frames.",
            )
            st.caption(
                "The chart is only comparable within this selected pair and evidence window. Lower values mean the saved projected positions were closer; they do not prove contact or fault."
            )
        else:
            st.info(
                "TrackSight withholds the paired separation trace because this evidence window has no paired real calibrated observations.",
                icon=":material/visibility_off:",
            )
        st.info(reconstruction["caveat"], icon=":material/info:")

    replay_points = build_incident_replay_points(
        trajectories,
        selected_event["track_id_values"],
        selected_event["time_seconds"],
        selected_event["frame"],
    )
    replay_summary = build_incident_replay_summary(
        replay_points,
        selected_event["track_id_values"],
    )
    if replay_summary["missing_track_ids"]:
        st.warning(
            "No real saved observation was available in this replay window for vehicle ID(s) "
            f"{', '.join(map(str, replay_summary['missing_track_ids']))}. "
            "TrackSight leaves that trail incomplete instead of inventing a position.",
            icon=":material/visibility_off:",
        )
    elif replay_summary["contains_observation_gaps"]:
        st.info(
            "A vehicle has one or more observation gaps in this replay. Separate line segments show what was observed without drawing through the gap.",
            icon=":material/info:",
        )

    replay_dataframe = pd.DataFrame(replay_points)
    if not replay_dataframe.empty:
        coordinate_columns = (
            ("track_x", "track_y", "Local track X", "Local track Y")
            if replay_summary["has_calibrated_positions"]
            else ("camera_x", "camera_y", "Camera X", "Camera Y")
        )
        x_column, y_column, x_title, y_title = coordinate_columns
        path_chart = (
            alt.Chart(replay_dataframe)
            .mark_line(point=False)
            .encode(
                x=alt.X(f"{x_column}:Q", title=x_title),
                y=alt.Y(
                    f"{y_column}:Q",
                    title=y_title,
                    scale=alt.Scale(reverse=True),
                ),
                color=alt.Color("track_id:N", title="Vehicle ID"),
                detail=alt.Detail("replay_segment:N"),
                order=alt.Order("frame:Q"),
                tooltip=[
                    alt.Tooltip("track_id:N", title="Vehicle ID"),
                    alt.Tooltip("time_seconds:Q", title="Video time (s)", format=".2f"),
                    alt.Tooltip("confidence:Q", title="Detector confidence", format=".2f"),
                ],
            )
        )
        event_points = replay_dataframe[replay_dataframe["is_event_frame"]]
        if not event_points.empty:
            path_chart = path_chart + (
                alt.Chart(event_points)
                .mark_point(shape="diamond", filled=True, size=160, color="#ffd166")
                .encode(
                    x=alt.X(f"{x_column}:Q"),
                    y=alt.Y(f"{y_column}:Q"),
                    tooltip=[
                        alt.Tooltip("track_id:N", title="Vehicle ID"),
                        alt.Tooltip("time_seconds:Q", title="Evidence time (s)", format=".2f"),
                    ],
                )
            )
        st.subheader("Focused two-vehicle movement", icon=":material/route:")
        st.altair_chart(
            path_chart.properties(height=360),
            width="stretch",
            alt="Observed movement trails for only the two vehicles involved in the selected evidence event.",
        )
        st.caption(
            "The yellow diamond marks a saved observation at the event frame. The movement chart contains only the selected pair; the time-aligned camera replay below retains surrounding traffic for context."
        )
    else:
        st.warning(
            "This saved event has no real trajectory observations in its replay window. The supporting camera clip remains the authoritative review material.",
            icon=":material/visibility_off:",
        )

    context_points = build_incident_replay_context_points(
        trajectories,
        selected_event["track_id_values"],
        selected_event["frame"],
    )
    if context_points:
        context_dataframe = pd.DataFrame(context_points)
        surrounding_traffic = context_dataframe[
            ~context_dataframe["is_selected"]
        ]
        selected_pair = context_dataframe[context_dataframe["is_selected"]]
        context_chart = alt.Chart(surrounding_traffic).mark_circle(
            filled=True,
            size=55,
            opacity=0.55,
            color="#9aa0a6",
        ).encode(
            x=alt.X("track_x:Q", title="Local track X"),
            y=alt.Y("track_y:Q", title="Local track Y", scale=alt.Scale(reverse=True)),
            tooltip=[alt.Tooltip("track_id:N", title="Vehicle ID")],
        )
        selected_pair_chart = alt.Chart(selected_pair).mark_circle(
            filled=True,
            size=220,
            color="#ffd166",
            stroke="#5a3d00",
            strokeWidth=2,
        ).encode(
            x=alt.X("track_x:Q"),
            y=alt.Y("track_y:Q", scale=alt.Scale(reverse=True)),
            tooltip=[alt.Tooltip("track_id:N", title="Selected vehicle ID")],
        )
        selected_pair_labels = alt.Chart(selected_pair).mark_text(
            dy=-16,
            color="#ffd166",
            fontWeight="bold",
        ).encode(
            x=alt.X("track_x:Q"),
            y=alt.Y("track_y:Q", scale=alt.Scale(reverse=True)),
            text=alt.Text("track_id_label:N"),
        )
        st.subheader("Traffic context at the evidence moment", icon=":material/groups:")
        st.altair_chart(
            (context_chart + selected_pair_chart + selected_pair_labels).properties(height=260),
            width="stretch",
            alt="Real projected vehicle positions at the evidence frame, with the selected evidence pair highlighted.",
        )
        st.caption(
            "Gray markers are other real projected detector positions at the exact saved evidence frame. Gold markers are the selected pair. Missing detections are not placed on this map."
        )
    else:
        st.info(
            "No real projected vehicle positions were saved at this exact evidence frame, so TrackSight withholds the traffic-context snapshot.",
            icon=":material/visibility_off:",
        )

    replay_start_time = max(0.0, selected_event["time_seconds"] - 2.0)
    st.subheader("Time-aligned race-control replay", icon=":material/sync:")
    st.caption(
        f"Both panels begin at the same saved source-video time: {replay_start_time:.2f}s. "
        "The camera view keeps traffic context; the local track replay shows confirmed projected vehicle positions."
    )
    if not artifacts["annotated_video"].is_file():
        st.error(
            "The annotated camera video required for this replay is missing.",
            icon=":material/error:",
        )
        return
    with st.container(horizontal=True):
        with st.container(border=True):
            st.markdown("**Annotated camera**")
            st.video(
                str(artifacts["annotated_video"]),
                start_time=replay_start_time,
                alt="Annotated camera replay aligned to the selected evidence time window.",
            )
        with st.container(border=True):
            st.markdown("**Calibrated local track**")
            if artifacts["top_down_video"].is_file():
                st.video(
                    str(artifacts["top_down_video"]),
                    start_time=replay_start_time,
                    alt="Confirmed vehicle footpoints in the calibrated local-track replay aligned to the selected evidence time window.",
                )
            else:
                st.info(
                    "This run has no top-down video. The exact-frame traffic snapshot above remains available from saved calibrated observations.",
                    icon=":material/videocam_off:",
                )
    st.caption(
        "Both videos begin at the same source timestamp. Use their normal player controls to continue reviewing the surrounding context; the saved focused clip remains available in Evidence queue and Marshal briefing."
    )


def render_evidence_quality(trajectories, run_summary):
    """Show which real observations were eligible for safety evidence and why."""
    st.header("Evidence quality and coverage", icon=":material/verified:")
    coverage = build_evidence_coverage(run_summary, trajectories)
    with st.container(horizontal=True):
        st.metric(
            "Safety-eligible ID segments",
            f"{coverage['trusted_track_segments']} / {coverage['all_track_segments']}",
            border=True,
            help="An ID segment must meet the saved consecutive real-observation requirement before TrackSight may use it for safety evidence.",
        )
        st.metric(
            "Frames with trusted observations",
            f"{coverage['trusted_frame_ratio']:.0%}",
            border=True,
            help="Share of processed frames containing at least one real observation from a safety-eligible ID segment. This is not detector accuracy.",
        )
        st.metric(
            "Trusted real observations",
            coverage["trusted_observations"],
            border=True,
            help="Saved detector observations from safety-eligible segments. Estimated tracker positions are excluded.",
        )
        st.metric(
            "Projected trusted frames",
            (
                f"{coverage['projected_trusted_frame_ratio']:.0%}"
                if coverage["spatial_analysis_enabled"]
                else "Unavailable"
            ),
            border=True,
            help="Share of processed frames containing a trusted observation with a calibrated local-track position.",
        )

    st.caption(
        "These are evidence-eligibility measures, not detector accuracy, vehicle counts, or a guarantee that every car was observed in every frame. "
        "They make clear which saved observations TrackSight was allowed to use for safety review."
    )
    coverage_dataframe = pd.DataFrame(
        [
            {
                "Evidence input": "Any real detector observation",
                "Frames": coverage["all_observation_frames"],
                "Share of processed video": (
                    f"{coverage['all_observation_frames'] / coverage['processed_frame_count']:.1%}"
                    if coverage["processed_frame_count"]
                    else "0.0%"
                ),
                "Use": "Saved for audit, but not automatically safety-eligible.",
            },
            {
                "Evidence input": "Trusted real observation",
                "Frames": coverage["trusted_observation_frames"],
                "Share of processed video": f"{coverage['trusted_frame_ratio']:.1%}",
                "Use": "Passed the continuity gate and may support safety evidence.",
            },
            {
                "Evidence input": "Trusted calibrated observation",
                "Frames": coverage["projected_trusted_frames"],
                "Share of processed video": f"{coverage['projected_trusted_frame_ratio']:.1%}",
                "Use": "Also has a local-track position for spatial review.",
            },
        ]
    )
    st.dataframe(
        coverage_dataframe,
        hide_index=True,
        alt="Evidence eligibility coverage across the completed TrackSight video run.",
    )
    st.subheader("What TrackSight excluded or withheld", icon=":material/shield:")
    for reason in coverage["exclusion_reasons"]:
        st.info(reason, icon=":material/info:")


def render_candidate_hotspots(
    artifacts,
    trajectories,
    review_events,
    risk_events,
    run_is_quality_assessed,
    spatial_analysis_enabled,
):
    """Group repeated calibrated candidate events into reviewer-facing locations."""
    st.header("Candidate-risk hotspots", icon=":material/location_searching:")
    st.warning(
        "A hotspot is a concentration of saved candidate events for review. It does not establish dangerous driving, fault, a crash, or a track defect.",
        icon=":material/warning:",
    )
    if not run_is_quality_assessed:
        st.warning(
            "This legacy run was created before TrackSight added track-quality controls. Rerun the video before using hotspot analysis.",
            icon=":material/error:",
        )
        return
    if not spatial_analysis_enabled:
        st.info(
            "Hotspot analysis is unavailable because this run has no usable camera calibration. "
            "TrackSight does not group uncalibrated image positions into apparent track locations.",
            icon=":material/videocam_off:",
        )
        return

    event_dataframe = build_race_control_timeline_dataframe(
        review_events,
        risk_events,
    )
    if event_dataframe.empty:
        st.info("This run did not generate evidence events to map.")
        return
    evidence_type = st.segmented_control(
        "Evidence to map",
        ["Predicted risk", "Observed proximity"],
        default="Predicted risk",
        key="hotspot_evidence_type",
        help="The two evidence types use different valid measurements, so TrackSight never groups them together.",
    )
    selected_events = event_dataframe[
        event_dataframe["evidence_type"].eq(evidence_type)
    ].sort_values("time_seconds")
    if selected_events.empty:
        st.info(f"This run has no {evidence_type.lower()} events to map.")
        return
    event_locations = build_calibrated_event_locations(
        selected_events.to_dict("records"),
        trajectories,
    )
    if not event_locations:
        st.info(
            "No selected evidence event has two saved calibrated vehicle positions at its anchor frame. "
            "TrackSight therefore withholds the hotspot map rather than placing an uncertain marker.",
            icon=":material/visibility_off:",
        )
        return
    neighborhood_vehicle_widths = st.slider(
        "Hotspot neighborhood",
        min_value=2.0,
        max_value=12.0,
        value=6.0,
        step=0.5,
        format="%.1f average vehicle widths",
        key="hotspot_neighborhood_vehicle_widths",
        help="Controls only how nearby saved candidate events are grouped for review. It is not a risk threshold or a physical track measurement.",
    )
    hotspots, radius_track_units = build_candidate_hotspots(
        event_locations,
        neighborhood_vehicle_widths=neighborhood_vehicle_widths,
    )
    event_location_dataframe = pd.DataFrame(event_locations)
    with st.container(horizontal=True):
        st.metric("Mapped candidate events", len(event_locations), border=True)
        st.metric("Recurring hotspots", len(hotspots), border=True)
        st.metric(
            "Grouping radius",
            f"{neighborhood_vehicle_widths:.1f} vehicle widths",
            border=True,
            help="Derived from the median projected vehicle width among the selected evidence events.",
        )
        st.metric(
            "Local radius",
            f"{radius_track_units:.1f} units",
            border=True,
            help="A local calibrated coordinate distance, not metres unless the camera profile has separately been given a real-world scale.",
        )
    st.caption(
        "Each small marker is one saved evidence event at the midpoint of its two real calibrated vehicle positions. "
        "Large numbered markers identify locations where at least two selected candidates fall within the review neighborhood."
    )
    event_points = (
        alt.Chart(event_location_dataframe)
        .mark_point(filled=True, size=70, opacity=0.45, color="#9aa0a6")
        .encode(
            x=alt.X("track_x:Q", title="Local track X"),
            y=alt.Y("track_y:Q", title="Local track Y", scale=alt.Scale(reverse=True)),
            tooltip=[
                alt.Tooltip("track_ids_label:N", title="Vehicles"),
                alt.Tooltip("time_seconds:Q", title="Video time (s)", format=".2f"),
                alt.Tooltip("measurement_value:Q", title="Separation", format=".3f"),
            ],
        )
    )
    chart_layers = [event_points]
    if hotspots:
        hotspot_dataframe = pd.DataFrame(hotspots)
        hotspot_points = (
            alt.Chart(hotspot_dataframe)
            .mark_circle(filled=True, opacity=0.8, color="#ff9f1c")
            .encode(
                x=alt.X("track_x:Q"),
                y=alt.Y("track_y:Q", scale=alt.Scale(reverse=True)),
                size=alt.Size(
                    "candidate_event_count:Q",
                    title="Candidate events",
                    scale=alt.Scale(range=[350, 1400]),
                ),
                tooltip=[
                    alt.Tooltip("candidate_event_count:Q", title="Candidate events"),
                    alt.Tooltip("earliest_time_seconds:Q", title="First event (s)", format=".2f"),
                    alt.Tooltip("latest_time_seconds:Q", title="Last event (s)", format=".2f"),
                    alt.Tooltip("vehicle_ids_label:N", title="Vehicle IDs"),
                ],
            )
        )
        hotspot_counts = (
            alt.Chart(hotspot_dataframe)
            .mark_text(color="black", fontWeight="bold")
            .encode(
                x=alt.X("track_x:Q"),
                y=alt.Y("track_y:Q", scale=alt.Scale(reverse=True)),
                text=alt.Text("candidate_event_count:Q", format="d"),
            )
        )
        chart_layers.extend([hotspot_points, hotspot_counts])
    st.altair_chart(
        alt.layer(*chart_layers).properties(height=460),
        width="stretch",
        alt="Calibrated local-track positions of saved candidate events and their recurring review hotspots.",
    )
    if not hotspots:
        st.info(
            "No recurring hotspot was found at this review grouping scale. The individual candidate markers remain available for review.",
            icon=":material/info:",
        )
        return

    hotspot_ids = [hotspot["hotspot_id"] for hotspot in hotspots]
    selected_hotspot_id = st.selectbox(
        "Open hotspot",
        options=hotspot_ids,
        key="hotspot_selector",
        format_func=lambda hotspot_id: (
            f"{next(hotspot for hotspot in hotspots if hotspot['hotspot_id'] == hotspot_id)['candidate_event_count']} "
            f"candidate events — {next(hotspot for hotspot in hotspots if hotspot['hotspot_id'] == hotspot_id)['earliest_time_seconds']:.2f}s "
            f"to {next(hotspot for hotspot in hotspots if hotspot['hotspot_id'] == hotspot_id)['latest_time_seconds']:.2f}s"
        ),
    )
    selected_hotspot = next(
        hotspot for hotspot in hotspots if hotspot["hotspot_id"] == selected_hotspot_id
    )
    selected_event_rows = event_dataframe[
        event_dataframe["event_key"].isin(selected_hotspot["event_keys"])
    ].sort_values("time_seconds")
    with st.container(border=True):
        st.subheader("Evidence behind this hotspot", icon=":material/manage_search:")
        st.write(
            f"This review location contains {selected_hotspot['candidate_event_count']} saved "
            f"{evidence_type.lower()} event(s), involving vehicle ID(s) "
            f"{', '.join(map(str, selected_hotspot['vehicle_ids']))}."
        )
        st.dataframe(
            selected_event_rows[
                [
                    "track_ids",
                    "time_seconds",
                    "frame",
                    "measurement_value",
                    "measurement_basis",
                ]
            ],
            column_config={
                "track_ids": "Vehicles",
                "time_seconds": st.column_config.NumberColumn(
                    "Video time (s)", format="%.2f"
                ),
                "frame": "Frame",
                "measurement_value": st.column_config.NumberColumn(
                    "Separation", format="%.3f"
                ),
                "measurement_basis": "Measurement basis",
            },
            hide_index=True,
            alt="Saved evidence events grouped at the selected candidate-risk hotspot.",
        )
        hotspot_event_keys = selected_event_rows["event_key"].tolist()
        selected_event_key = st.selectbox(
            "Open supporting event",
            options=hotspot_event_keys,
            key="hotspot_event_selector",
            format_func=lambda event_key: (
                f"Vehicles {selected_event_rows.loc[selected_event_rows['event_key'].eq(event_key), 'track_ids'].iloc[0]} at "
                f"{selected_event_rows.loc[selected_event_rows['event_key'].eq(event_key), 'time_seconds'].iloc[0]:.2f}s"
            ),
        )
        selected_event = selected_event_rows[
            selected_event_rows["event_key"].eq(selected_event_key)
        ].iloc[0]
        clip_path = selected_event["clip"]
        if isinstance(clip_path, str) and Path(clip_path).is_file():
            st.video(
                clip_path,
                alt="Annotated camera clip supporting an event in the selected candidate-risk hotspot.",
            )
        else:
            st.info(
                "No separate clip was saved for this event. The full annotated camera video opens shortly before the evidence time.",
                icon=":material/info:",
            )
            st.video(
                str(artifacts["annotated_video"]),
                start_time=max(0.0, selected_event["time_seconds"] - 2.0),
                alt="Annotated camera video starting shortly before an event in the selected candidate-risk hotspot.",
            )


def render_incident_threads(
    artifacts,
    review_events,
    risk_events,
    run_summary,
    run_is_quality_assessed,
    spatial_analysis_enabled,
    safety_zone_evidence,
):
    """Turn related saved evidence into compact, reviewer-selected cases."""
    st.header("Incident threads", icon=":material/account_tree:")
    st.warning(
        "An incident thread is a time-bounded review case built from related saved evidence. "
        "It does not establish an incident, fault, contact, or a rule violation.",
        icon=":material/warning:",
    )
    if not run_is_quality_assessed:
        st.warning(
            "This legacy run was created before TrackSight added track-quality controls. Rerun the video before using incident threads.",
            icon=":material/error:",
        )
        return
    if not spatial_analysis_enabled:
        st.info(
            "Incident threads are unavailable because this moving camera has no stable spatial calibration. "
            "TrackSight will not turn uncalibrated image positions into a spatial safety case.",
            icon=":material/videocam_off:",
        )
        return
    event_dataframe = build_race_control_timeline_dataframe(
        review_events,
        risk_events,
    )
    if event_dataframe.empty:
        st.info("This run did not generate saved evidence to group into review cases.")
        return

    thread_gap_seconds = st.slider(
        "Maximum gap inside one review case",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.25,
        format="%.2f seconds",
        key="incident_thread_gap_seconds",
        help="This changes only how related saved evidence is grouped for review. It is not a risk threshold.",
    )
    threads = build_incident_threads(
        event_dataframe.to_dict("records"),
        gap_seconds=thread_gap_seconds,
    )
    linked_threads = [thread for thread in threads if thread["event_count"] >= 2]
    if linked_threads:
        case_scope = st.segmented_control(
            "Cases to display",
            options=["Linked cases (2+ records)", "All saved cases"],
            default="Linked cases (2+ records)",
            required=True,
            key="incident_thread_case_scope_control",
            help=(
                "Linked cases are the focused review queue: the same two vehicles have more than one "
                "nearby saved evidence record. All saved cases retains single-record observations for audit."
            ),
        )
        visible_threads = linked_threads if case_scope == "Linked cases (2+ records)" else threads
    else:
        visible_threads = threads
        st.info(
            "No vehicle pair has multiple nearby saved evidence records in this run. "
            "Showing all saved cases for audit.",
            icon=":material/info:",
        )
    with st.container(horizontal=True):
        st.metric("Linked review cases", len(linked_threads), border=True)
        st.metric("All saved cases", len(threads), border=True)
        st.metric("Saved evidence records", len(event_dataframe), border=True)
        st.metric(
            "Grouping gap",
            f"{thread_gap_seconds:.2f} s",
            border=True,
            help="The maximum time between vehicle-linked evidence records in one case.",
        )
    st.caption(
        "The focused queue shows only vehicle pairs with repeated nearby evidence. Each case can contain "
        "observed and predicted evidence, but only for the same two vehicle IDs. The complete single-record "
        "audit trail remains available through All saved cases. A six-second cap prevents repeated evidence "
        "from becoming one misleading case."
        if linked_threads
        else "Each saved case concerns one exact vehicle pair. A six-second cap prevents repeated evidence "
        "from becoming one misleading case."
    )
    thread_rows = pd.DataFrame(
        [
            {
                "thread_id": thread["thread_id"],
                "vehicles": thread["track_ids_label"],
                "start_seconds": thread["earliest_time_seconds"],
                "end_seconds": thread["latest_time_seconds"],
                "evidence_records": thread["event_count"],
                "predicted_records": thread["predicted_risk_event_count"],
                "observed_records": thread["observed_proximity_event_count"],
            }
            for thread in visible_threads
        ]
    )
    st.dataframe(
        thread_rows.drop(columns="thread_id"),
        column_config={
            "vehicles": "Vehicle IDs",
            "start_seconds": st.column_config.NumberColumn("Start (s)", format="%.2f"),
            "end_seconds": st.column_config.NumberColumn("End (s)", format="%.2f"),
            "evidence_records": "Evidence records",
            "predicted_records": "Predicted records",
            "observed_records": "Observed records",
        },
        hide_index=True,
        alt="Time-bounded TrackSight review cases built from vehicle-linked saved evidence.",
    )
    selected_thread_id = st.selectbox(
        "Open review case",
        options=[thread["thread_id"] for thread in visible_threads],
        key=f"incident_thread_selector_{'linked' if linked_threads and case_scope == 'Linked cases (2+ records)' else 'all'}",
        format_func=lambda thread_id: (
            f"Vehicles {next(thread for thread in visible_threads if thread['thread_id'] == thread_id)['track_ids_label']} — "
            f"{next(thread for thread in visible_threads if thread['thread_id'] == thread_id)['earliest_time_seconds']:.2f}s to "
            f"{next(thread for thread in visible_threads if thread['thread_id'] == thread_id)['latest_time_seconds']:.2f}s"
        ),
    )
    selected_thread = next(
        thread for thread in visible_threads if thread["thread_id"] == selected_thread_id
    )
    selected_event_rows = event_dataframe[
        event_dataframe["event_key"].isin(selected_thread["event_keys"])
    ].sort_values("time_seconds")
    decision_path = artifacts["human_review_decisions_json"]
    try:
        saved_decisions = load_human_review_decisions(decision_path)
    except ValueError as error:
        saved_decisions = {"decisions": {}}
        st.error(
            f"The saved human-review file cannot be used: {error}",
            icon=":material/error:",
        )
    current_event_summaries = {
        event["review_event_id"]: get_human_review_event_summary(event)
        for _, event in event_dataframe.iterrows()
    }
    current_decisions = filter_current_human_review_decisions(
        saved_decisions,
        current_event_summaries,
    )
    with st.container(border=True):
        st.subheader("Case evidence", icon=":material/manage_search:")
        st.write(
            f"This case contains {selected_thread['event_count']} saved evidence record(s) involving vehicle ID(s) "
            f"{selected_thread['track_ids_label']} from {selected_thread['earliest_time_seconds']:.2f}s to "
            f"{selected_thread['latest_time_seconds']:.2f}s."
        )
        st.dataframe(
            selected_event_rows[
                [
                    "evidence_type",
                    "track_ids",
                    "time_seconds",
                    "frame",
                    "measurement_value",
                    "measurement_basis",
                ]
            ],
            column_config={
                "evidence_type": "Evidence type",
                "track_ids": "Vehicles",
                "time_seconds": st.column_config.NumberColumn("Video time (s)", format="%.2f"),
                "frame": "Frame",
                "measurement_value": st.column_config.NumberColumn("Value", format="%.3f"),
                "measurement_basis": "Measurement basis",
            },
            hide_index=True,
            alt="Source evidence records for the selected TrackSight review case.",
        )
        clip_path = selected_thread["preferred_clip"]
        if isinstance(clip_path, str) and Path(clip_path).is_file():
            st.video(
                clip_path,
                alt="Annotated camera clip supporting the selected TrackSight review case.",
            )
        else:
            st.info(
                "No separate short clip was saved for this case. The full annotated camera video opens shortly before the first evidence time.",
                icon=":material/info:",
            )
            st.video(
                str(artifacts["annotated_video"]),
                start_time=max(0.0, selected_thread["earliest_time_seconds"] - 2.0),
                alt="Annotated camera video starting shortly before the selected TrackSight review case.",
            )
        packet = build_incident_packet(
            selected_thread,
            run_summary,
            current_decisions,
            safety_zone_evidence=safety_zone_evidence,
        )
        st.download_button(
            "Download incident packet",
            data=packet,
            file_name=f"{artifacts['run_name']}_{selected_thread['thread_id']}_incident_packet.zip",
            mime="application/zip",
            help="Downloads this selected review case, its source evidence, current human decisions, optional zone context, supporting clips, and SHA-256 manifest without local machine paths.",
            icon=":material/download:",
            key=f"download_incident_packet_{selected_thread['thread_id']}",
        )


def render_safety_zones(
    artifacts,
    safety_zone_evidence,
    run_is_quality_assessed,
    spatial_analysis_enabled,
):
    """Show vehicles observed inside explicitly configured local-track zones."""
    st.header("Safety zones", icon=":material/fence:")
    st.warning(
        "Safety-zone observations show only that a trusted, real detection was inside a reviewer-configured local-track polygon. "
        "They do not establish an infringement, fault, or unsafe driving.",
        icon=":material/warning:",
    )
    if not run_is_quality_assessed:
        st.warning(
            "This legacy run was created before TrackSight added track-quality controls. Rerun the video before using safety zones.",
            icon=":material/error:",
        )
        return
    if not spatial_analysis_enabled:
        st.info(
            "Safety zones require a usable camera calibration because they are defined in local track units.",
            icon=":material/videocam_off:",
        )
        return
    if safety_zone_evidence is None:
        st.info(
            "No safety-zone profile was used for this completed run. A profile is optional and camera-specific; run the pipeline with --safety-zone-config <profile.json> to record zone observations.",
            icon=":material/info:",
        )
        return
    if safety_zone_evidence.get("schema_version") != SAFETY_ZONE_EVIDENCE_SCHEMA_VERSION:
        st.error(
            "The saved safety-zone evidence uses an unsupported format. Reprocess the run with the current pipeline.",
            icon=":material/error:",
        )
        return
    zones = safety_zone_evidence.get("zones", [])
    observations = safety_zone_evidence.get("observations", [])
    with st.container(horizontal=True):
        st.metric("Configured zones", len(zones), border=True)
        st.metric("Saved zone observations", len(observations), border=True)
        st.metric(
            "Vehicles observed in zones",
            len({observation["track_id"] for observation in observations}),
            border=True,
        )
    st.caption(
        "Zone membership uses only saved calibrated positions. A gap in real detections creates separate observations rather than an invented continuous stay."
    )
    polygon_rows = []
    for zone in zones:
        polygon = zone["polygon"]
        for vertex_index, point in enumerate(polygon + polygon[:1]):
            polygon_rows.append(
                {
                    "zone_label": zone["label"],
                    "zone_category": zone["category"],
                    "vertex_index": vertex_index,
                    "track_x": point[0],
                    "track_y": point[1],
                }
            )
    zone_chart = (
        alt.Chart(pd.DataFrame(polygon_rows))
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("track_x:Q", title="Local track X"),
            y=alt.Y("track_y:Q", title="Local track Y", scale=alt.Scale(reverse=True)),
            color=alt.Color("zone_label:N", title="Safety zone"),
            detail=alt.Detail("zone_label:N"),
            order=alt.Order("vertex_index:Q"),
            tooltip=[
                alt.Tooltip("zone_label:N", title="Safety zone"),
                alt.Tooltip("zone_category:N", title="Category"),
            ],
        )
    )
    chart_layers = [zone_chart]
    if observations:
        observation_dataframe = pd.DataFrame(observations)
        chart_layers.append(
            alt.Chart(observation_dataframe)
            .mark_point(filled=True, size=100, color="#ff9f1c")
            .encode(
                x=alt.X("entry_track_x:Q"),
                y=alt.Y("entry_track_y:Q", scale=alt.Scale(reverse=True)),
                tooltip=[
                    alt.Tooltip("zone_label:N", title="Safety zone"),
                    alt.Tooltip("track_id:N", title="Vehicle ID"),
                    alt.Tooltip("start_seconds:Q", title="Entry time (s)", format=".2f"),
                    alt.Tooltip("frame_count:Q", title="Observed frames"),
                ],
            )
        )
    st.altair_chart(
        alt.layer(*chart_layers).properties(height=460),
        width="stretch",
        alt="Configured local-track safety zones and real vehicle entry observations when available.",
    )
    if not observations:
        st.info(
            "No trusted vehicle observation entered a configured safety zone in this completed run.",
            icon=":material/check_circle:",
        )
        return
    observation_dataframe = pd.DataFrame(observations)
    st.dataframe(
        observation_dataframe[
            [
                "zone_label",
                "zone_category",
                "track_id",
                "start_seconds",
                "end_seconds",
                "frame_count",
            ]
        ],
        column_config={
            "zone_label": "Safety zone",
            "zone_category": "Category",
            "track_id": "Vehicle ID",
            "start_seconds": st.column_config.NumberColumn("Start (s)", format="%.2f"),
            "end_seconds": st.column_config.NumberColumn("End (s)", format="%.2f"),
            "frame_count": "Observed frames",
        },
        hide_index=True,
        alt="Trusted vehicle observations recorded inside configured safety zones.",
    )
    selected_observation_index = st.selectbox(
        "Open safety-zone observation",
        options=observation_dataframe.index.tolist(),
        key="safety_zone_observation_selector",
        format_func=lambda index: (
            f"{observation_dataframe.loc[index, 'zone_label']} — vehicle "
            f"{observation_dataframe.loc[index, 'track_id']} at "
            f"{observation_dataframe.loc[index, 'start_seconds']:.2f}s"
        ),
    )
    selected_observation = observation_dataframe.loc[selected_observation_index]
    st.video(
        str(artifacts["annotated_video"]),
        start_time=max(0.0, selected_observation["start_seconds"] - 2.0),
        alt="Annotated camera video starting shortly before the selected safety-zone observation.",
    )


def render_safety_candidates(
    artifacts,
    safety_candidate_evidence,
    run_summary,
    run_is_quality_assessed,
):
    """Present calibrated low-displacement and restricted-area candidates."""
    st.header("Safety candidates", icon=":material/health_and_safety:")
    st.warning(
        "These are evidence-backed prompts for human review. They do not confirm a stopped vehicle, a track exit, a hazard, fault, or a rule breach.",
        icon=":material/warning:",
    )
    if not run_is_quality_assessed:
        st.info("Reprocess this legacy run before using the safety-candidate analysis.")
        return
    if not run_summary.get("spatial_analysis_enabled", False):
        st.info(
            "Safety candidates require a usable camera calibration because they compare saved positions in local track units.",
            icon=":material/videocam_off:",
        )
        return
    if safety_candidate_evidence is None:
        st.info(
            "This completed run predates safety-candidate analysis. Reprocess the video to create low-displacement and restricted-area evidence.",
            icon=":material/info:",
        )
        return
    if (
        safety_candidate_evidence.get("schema_version")
        != SAFETY_CANDIDATE_EVIDENCE_SCHEMA_VERSION
    ):
        st.error(
            "The saved safety-candidate evidence uses an unsupported format. Reprocess the run before review.",
            icon=":material/error:",
        )
        return

    stationary_candidates = safety_candidate_evidence[
        "stationary_vehicle_candidates"
    ]
    restricted_candidates = safety_candidate_evidence[
        "restricted_area_entry_candidates"
    ]
    active_racing_area_candidates = safety_candidate_evidence.get(
        "stationary_in_active_racing_area_candidates", []
    )
    against_direction_candidates = safety_candidate_evidence.get(
        "against_expected_direction_candidates", []
    )
    with st.container(horizontal=True):
        st.metric("Low-displacement candidates", len(stationary_candidates), border=True)
        st.metric("Restricted-area candidates", len(restricted_candidates), border=True)
        st.metric("Active-area low-displacement", len(active_racing_area_candidates), border=True)
        st.metric("Against-direction prompts", len(against_direction_candidates), border=True)
        st.metric("Total candidate prompts", len(get_safety_candidates(safety_candidate_evidence)), border=True)
    st.caption(
        "Low displacement is measured relative to each saved vehicle segment's projected width. Zone policies are optional, camera-specific local-track configuration—not rules inferred from an image."
    )
    candidates = get_safety_candidates(safety_candidate_evidence)
    if not candidates:
        st.success(
            "No safety candidates were saved for this completed calibrated run.",
            icon=":material/check_circle:",
        )
        return

    candidate_scope = st.segmented_control(
        "Candidate type",
        ["All", "Low displacement", "Restricted-area entry", "Active-area low displacement", "Against configured direction"],
        default="All",
        key="safety_candidate_scope",
    )
    visible_candidates = [
        candidate
        for candidate in candidates
        if candidate_scope == "All"
        or (
            candidate_scope == "Low displacement"
            and candidate["candidate_type"] == "stationary_vehicle"
        )
        or (
            candidate_scope == "Restricted-area entry"
            and candidate["candidate_type"] == "restricted_area_entry"
        )
        or (
            candidate_scope == "Active-area low displacement"
            and candidate["candidate_type"] == "stationary_in_active_racing_area"
        )
        or (
            candidate_scope == "Against configured direction"
            and candidate["candidate_type"] == "against_expected_direction"
        )
    ]
    candidate_rows = []
    for index, candidate in enumerate(visible_candidates):
        candidate_rows.append(
            {
                "candidate_key": f"{candidate['candidate_type']}-{index}",
                "candidate_type": {
                    "stationary_vehicle": "Low-displacement vehicle",
                    "restricted_area_entry": "Restricted-area entry",
                    "stationary_in_active_racing_area": "Low displacement on active racing area",
                    "against_expected_direction": "Against configured direction",
                }.get(candidate["candidate_type"], candidate["candidate_type"]),
                "vehicle_id": candidate["track_id"],
                "time_seconds": candidate["anchor_time_seconds"],
                "duration_seconds": candidate["duration_seconds"],
                "observed_frames": candidate["frame_count"],
                "zone": candidate.get("zone_label", "—"),
                "reason": candidate["review_reason"],
            }
        )
    candidate_dataframe = pd.DataFrame(candidate_rows)
    st.dataframe(
        candidate_dataframe.drop(columns="candidate_key"),
        column_config={
            "candidate_type": "Candidate type",
            "vehicle_id": "Vehicle ID",
            "time_seconds": st.column_config.NumberColumn("Evidence time (s)", format="%.2f"),
            "duration_seconds": st.column_config.NumberColumn("Observed duration (s)", format="%.2f"),
            "observed_frames": "Real consecutive frames",
            "zone": "Reviewer-defined zone",
            "reason": "Why TrackSight surfaced it",
        },
        hide_index=True,
        alt="Saved low-displacement vehicle and reviewer-defined restricted-area candidates for human safety review.",
    )
    selected_candidate_index = st.selectbox(
        "Open safety candidate",
        options=list(range(len(visible_candidates))),
        format_func=lambda index: (
            f"Vehicle {visible_candidates[index]['track_id']} — "
            f"{visible_candidates[index]['candidate_type'].replace('_', ' ')} at "
            f"{visible_candidates[index]['anchor_time_seconds']:.2f}s"
        ),
        key="safety_candidate_selector",
    )
    selected_candidate = visible_candidates[selected_candidate_index]
    briefing = (
        load_saved_json(artifacts["marshal_briefing_json"])
        if artifacts["marshal_briefing_json"].is_file()
        else {}
    )
    matching_highlight = next(
        (
            highlight
            for highlight in briefing.get("safety_candidate_highlights", [])
            if highlight.get("evidence_type") == selected_candidate["candidate_type"]
            and highlight.get("frame") == selected_candidate["anchor_frame"]
            and highlight.get("track_ids") == selected_candidate["track_ids"]
        ),
        None,
    )
    with st.container(border=True):
        st.subheader("Evidence-grounded reviewer briefing", icon=":material/summarize:")
        st.write(selected_candidate["review_reason"])
        st.caption(f"Measurement basis: {selected_candidate['measurement_basis']}.")
        if matching_highlight is not None:
            st.info(matching_highlight["evidence_grounding"], icon=":material/verified:")
        else:
            st.info(
                "This candidate was saved from trusted real calibrated observations. No additional briefing highlight was saved, so the full candidate record remains the source of truth.",
                icon=":material/info:",
            )
        st.write(selected_candidate["recommended_human_action"])
        source_checks = pd.DataFrame(
            [
                {"evidence source": "Camera calibration", "status": "Available", "detail": "Positions are expressed in this camera's local-track coordinates."},
                {"evidence source": "ID continuity", "status": "Passed", "detail": "This candidate uses a segment that passed the real-observation continuity gate."},
                {"evidence source": "Candidate rule", "status": "Saved", "detail": selected_candidate["measurement_basis"]},
                {"evidence source": "Human action", "status": "Required", "detail": "TrackSight does not make an operational or rules decision."},
            ]
        )
        st.dataframe(
            source_checks,
            column_config={
                "evidence source": "Evidence source",
                "status": "Status",
                "detail": "What was used",
            },
            hide_index=True,
            alt="Calibration, continuity, measurement, and human-review safeguards supporting the selected safety candidate.",
        )
        clip_path = selected_candidate.get("marshal_briefing_clip_path")
        if isinstance(clip_path, str) and Path(clip_path).is_file():
            st.video(
                clip_path,
                alt="Annotated camera clip supporting the selected TrackSight safety candidate.",
            )
        else:
            st.info(
                "No focused clip was saved for this candidate. The full annotated camera video opens shortly before the candidate time.",
                icon=":material/info:",
            )
            st.video(
                str(artifacts["annotated_video"]),
                start_time=max(0.0, selected_candidate["anchor_time_seconds"] - 2.0),
                alt="Annotated camera video starting shortly before the selected TrackSight safety candidate.",
            )


def render_human_review_workflow(
    artifacts,
    review_events,
    risk_events,
    run_is_quality_assessed,
    spatial_analysis_enabled,
):
    """Let a human record an auditable decision about saved evidence."""
    st.header("Human review decisions", icon=":material/fact_check:")
    st.warning(HUMAN_REVIEW_DISCLAIMER, icon=":material/warning:")
    if not run_is_quality_assessed:
        st.warning(
            "This legacy run was created before TrackSight added track-quality controls. Rerun the video before recording a review.",
            icon=":material/error:",
        )
        return
    if not spatial_analysis_enabled:
        st.info(
            "Human review decisions are unavailable because this moving camera has no stable spatial calibration.",
            icon=":material/videocam_off:",
        )
        return

    event_dataframe = build_race_control_timeline_dataframe(
        review_events,
        risk_events,
    )
    if event_dataframe.empty:
        st.info("This run did not generate evidence events to review.")
        return
    decision_path = artifacts["human_review_decisions_json"]
    try:
        decisions = load_human_review_decisions(decision_path)
    except ValueError as error:
        st.error(
            f"The saved human-review file cannot be used: {error}",
            icon=":material/error:",
        )
        return

    review_queue = build_human_review_queue(event_dataframe, decisions)
    current_decision_count = (review_queue["review_status"] != "Unreviewed").sum()
    with st.container(horizontal=True):
        st.metric(
            "Evidence events",
            len(review_queue),
            help="Saved TrackSight evidence items available for a human reviewer.",
            border=True,
        )
        st.metric(
            "Reviewed events",
            current_decision_count,
            help="Current evidence items with a saved human review decision.",
            border=True,
        )
        st.metric(
            "Escalated events",
            (review_queue["review_status"] == "Escalate").sum(),
            help="Human decisions marked for escalation; not confirmed incidents.",
            border=True,
        )

    st.caption(
        "A saved decision is reused only if its event details still exactly match this completed run. Reprocessing changed evidence therefore returns to Unreviewed."
    )
    st.dataframe(
        review_queue[
            [
                "evidence_type",
                "track_ids",
                "time_seconds",
                "frame",
                "review_status",
                "reviewer_name",
                "review_updated_at",
            ]
        ],
        column_config={
            "evidence_type": "Evidence type",
            "track_ids": "Vehicles",
            "time_seconds": st.column_config.NumberColumn(
                "Video time (s)", format="%.2f"
            ),
            "frame": "Frame",
            "review_status": "Human review status",
            "reviewer_name": "Reviewer",
            "review_updated_at": "Last updated (UTC)",
        },
        hide_index=True,
        height=280,
        alt="Human review status for each saved TrackSight evidence event.",
    )

    event_ids = review_queue["review_event_id"].tolist()
    selected_event_id = st.selectbox(
        "Open evidence for human review",
        options=event_ids,
        key="human_review_event_selector",
        format_func=lambda event_id: (
            f"{review_queue.loc[review_queue['review_event_id'].eq(event_id), 'evidence_type'].iloc[0]} "
            f"— vehicles {review_queue.loc[review_queue['review_event_id'].eq(event_id), 'track_ids'].iloc[0]} at "
            f"{review_queue.loc[review_queue['review_event_id'].eq(event_id), 'time_seconds'].iloc[0]:.2f}s "
            f"[{review_queue.loc[review_queue['review_event_id'].eq(event_id), 'review_status'].iloc[0]}]"
        ),
    )
    selected_event = review_queue[
        review_queue["review_event_id"].eq(selected_event_id)
    ].iloc[0]
    event_summary = get_human_review_event_summary(selected_event)
    existing_decision = decisions["decisions"].get(selected_event_id)
    if not is_current_human_review_decision(existing_decision, event_summary):
        existing_decision = {}

    with st.container(border=True):
        st.subheader("Evidence context", icon=":material/manage_search:")
        st.write(get_human_review_reason(selected_event))
        st.caption(
            "This explains why TrackSight surfaced the item. It is evidence for human review, not a decision about fault or an incident."
        )
        clip_path = selected_event["clip"]
        if isinstance(clip_path, str) and Path(clip_path).is_file():
            st.video(
                clip_path,
                alt="Annotated camera clip supporting the selected human-review event.",
            )
        else:
            st.info(
                "No short clip was saved for this event. The annotated camera video opens shortly before its evidence time.",
                icon=":material/info:",
            )
            st.video(
                str(artifacts["annotated_video"]),
                start_time=max(0.0, selected_event["time_seconds"] - 2.0),
                alt="Annotated camera video starting shortly before the selected human-review event.",
            )

    with st.form(f"human_review_form_{selected_event_id}"):
        st.subheader("Record human review", icon=":material/edit_note:")
        reviewer_name = st.text_input(
            "Reviewer name",
            value=existing_decision.get("reviewer_name", ""),
            max_chars=100,
            help="Required so the saved decision has accountable human provenance.",
        )
        existing_outcome = existing_decision.get("outcome", "Needs review")
        outcome = st.selectbox(
            "Review outcome",
            options=HUMAN_REVIEW_OUTCOMES,
            index=HUMAN_REVIEW_OUTCOMES.index(existing_outcome),
            help="Cleared and Escalate are human workflow labels, not automated findings by TrackSight.",
        )
        review_note = st.text_area(
            "Review note",
            value=existing_decision.get("review_note", ""),
            max_chars=2000,
            placeholder="State the evidence you checked and why this outcome is appropriate.",
            help="Required so another reviewer can understand the human decision.",
        )
        save_decision = st.form_submit_button(
            "Save human review",
            type="primary",
            icon=":material/save:",
        )
    if save_decision:
        try:
            updated_decisions = save_human_review_decision(
                decisions,
                selected_event_id,
                event_summary,
                outcome,
                reviewer_name,
                review_note,
            )
            write_human_review_decisions(updated_decisions, decision_path)
        except ValueError as error:
            st.error(str(error), icon=":material/error:")
        else:
            st.session_state["human_review_saved_message"] = (
                f"Saved {outcome.lower()} for vehicles {selected_event['track_ids']} at "
                f"{selected_event['time_seconds']:.2f}s."
            )
            st.rerun()

    saved_message = st.session_state.pop("human_review_saved_message", None)
    if saved_message:
        st.success(saved_message, icon=":material/check_circle:")

    current_event_summaries = {
        event["review_event_id"]: get_human_review_event_summary(event)
        for _, event in event_dataframe.iterrows()
    }
    current_decisions = filter_current_human_review_decisions(
        decisions,
        current_event_summaries,
    )
    if current_decisions["decisions"]:
        report = build_human_review_report(
            artifacts["run_name"],
            current_decisions,
        )
        st.download_button(
            "Download human-review decision report",
            data=json.dumps(report, indent=2),
            file_name=f"{artifacts['run_name']}_human_review_report.json",
            mime="application/json",
            help="Downloads the saved human decisions without local computer paths.",
            icon=":material/download:",
        )


def render_marshal_handover(
    artifacts,
    trajectories,
    review_events,
    risk_events,
    safety_candidate_evidence,
    run_summary,
    run_is_quality_assessed,
):
    """Show the next reviewer a concise, decision-safe session handover."""
    st.header("Marshal handover", icon=":material/handoff:")
    if not run_is_quality_assessed:
        st.warning("This legacy run is not eligible for a Marshal handover.", icon=":material/error:")
        return
    timeline = build_race_control_timeline_dataframe(review_events, risk_events)
    decisions = load_human_review_decisions(artifacts["human_review_decisions_json"])
    review_queue = build_human_review_queue(timeline, decisions) if not timeline.empty else pd.DataFrame()
    handover = build_marshal_handover(
        run_summary,
        trajectories,
        review_queue.to_dict("records"),
        safety_candidate_evidence,
    )
    st.warning(handover["disclaimer"], icon=":material/warning:")
    with st.container(horizontal=True):
        st.metric("Session duration", f"{handover['session']['duration_seconds']:.1f} s", border=True)
        st.metric("Unresolved evidence", handover["review_status"]["unreviewed"] + handover["review_status"]["needs_review"] + handover["review_status"]["escalated"], border=True)
        st.metric("Cleared by reviewer", handover["review_status"]["cleared"], border=True)
        st.metric("Safety-eligible segments", handover["coverage"]["trusted_track_segments"], border=True)
    st.subheader("Next three review items")
    next_items = handover["next_review_items"]
    if next_items:
        st.dataframe(
            pd.DataFrame(next_items),
            column_config={
                "review_status": "Review status",
                "evidence_type": "Evidence type",
                "track_ids": "Vehicle IDs",
                "frame": "Frame",
                "time_seconds": st.column_config.NumberColumn("Video time (s)", format="%.2f"),
                "measurement_basis": "Measurement basis",
                "has_supporting_clip": "Supporting clip saved",
            },
            hide_index=True,
            alt="The next unresolved TrackSight evidence items for a human Marshal to review.",
        )
    else:
        st.success("No unresolved saved proximity or predicted-risk evidence remains in this run.", icon=":material/check_circle:")
    st.subheader("Coverage and withheld scope")
    st.dataframe(
        pd.DataFrame([{"limitation or safeguard": item} for item in handover["limitations"]]),
        hide_index=True,
        alt="Camera, continuity, and evidence limitations that the next Marshal must keep in scope.",
    )
    st.subheader("Configured policy prompts")
    st.dataframe(
        pd.DataFrame([
            {"policy prompt": label.replace("_", " "), "count": count}
            for label, count in handover["policy_candidate_counts"].items()
        ]),
        hide_index=True,
        alt="Review-only candidate prompts created by optional local track policy configuration.",
    )
    st.download_button(
        "Download Marshal handover",
        data=json.dumps(handover, indent=2),
        file_name=f"{artifacts['run_name']}_marshal_handover.json",
        mime="application/json",
        help="Downloads the current review-status overlay and immutable evidence coverage without local computer paths.",
        icon=":material/download:",
    )


def render_evaluation_and_limits(
    artifacts, trajectories, run_summary, safety_candidate_evidence, run_is_quality_assessed
):
    """Show scope and output evidence without presenting unmeasured accuracy claims."""
    st.header("Evaluation and limits", icon=":material/fact_check:")
    if not run_is_quality_assessed:
        st.warning("This legacy run is not eligible for a transparent evaluation report.", icon=":material/error:")
        return
    report = build_evaluation_report(run_summary, trajectories, safety_candidate_evidence)
    st.warning(report["disclaimer"], icon=":material/warning:")
    with st.container(horizontal=True):
        st.metric("Detector positions", report["recorded_outputs"]["detector_positions"], border=True)
        st.metric("ID segments", report["recorded_outputs"]["tracker_id_segments"], border=True)
        st.metric("Safety-eligible segments", report["recorded_outputs"]["safety_eligible_segments"], border=True)
        st.metric("Review prompts", report["recorded_outputs"]["policy_candidate_prompts"], border=True)
    st.subheader("Completed run context")
    st.table({
        "Source video": report["evaluated_run_context"]["source_video"],
        "Camera mode": report["evaluated_run_context"]["camera_mode"],
        "Spatial analysis": report["evaluated_run_context"]["spatial_analysis_status"],
        "Processed frames": report["evaluated_run_context"]["processed_frames"],
    })
    st.subheader("Safeguards")
    for safeguard in report["safeguards"]:
        st.info(safeguard, icon=":material/verified:")
    st.subheader("Known limitations and withheld claims")
    for limitation in report["known_limitations"]:
        st.caption(f"• {limitation}")
    st.download_button(
        "Download evaluation and limits report",
        data=json.dumps(report, indent=2),
        file_name=f"{artifacts['run_name']}_evaluation_and_limits.json",
        mime="application/json",
        help="Downloads an honest run-scope report; it contains no invented accuracy metrics.",
        icon=":material/download:",
    )


def _current_review_queue(artifacts, review_events, risk_events):
    """Return saved events with only current human review decisions overlaid."""
    timeline = build_race_control_timeline_dataframe(review_events, risk_events)
    decisions = load_human_review_decisions(artifacts["human_review_decisions_json"])
    return (
        build_human_review_queue(timeline, decisions)
        if not timeline.empty else pd.DataFrame()
    )


def render_learning_queue(artifacts, review_events, risk_events, safety_candidate_evidence):
    """Prepare evidence for human annotation without calling it ground truth."""
    st.header("Learning queue", icon=":material/model_training:")
    st.warning(LEARNING_QUEUE_DISCLAIMER, icon=":material/warning:")
    review_queue = _current_review_queue(artifacts, review_events, risk_events)
    tasks = build_learning_queue(
        review_queue.to_dict("records"), get_safety_candidates(safety_candidate_evidence)
    )
    with st.container(horizontal=True):
        st.metric("Annotation tasks", len(tasks), border=True)
        st.metric("Evidence-event tasks", sum(task["task_type"] == "evidence_event" for task in tasks), border=True)
        st.metric("Policy-prompt tasks", sum(task["task_type"] == "policy_candidate" for task in tasks), border=True)
    st.caption("Cleared evidence is excluded. Every remaining task needs independent human box/identity/outcome annotation before it can be used for training or benchmarking.")
    if tasks:
        st.dataframe(
            pd.DataFrame(tasks), hide_index=True,
            column_config={
                "task_id": "Task ID", "task_type": "Source", "labeling_status": "Labelling status",
                "evidence_type": "Evidence type", "track_ids": "Vehicle IDs", "frame": "Frame",
                "time_seconds": st.column_config.NumberColumn("Video time (s)", format="%.2f"),
                "why_selected": "Why selected", "supporting_clip": "Supporting clip",
            },
            alt="Human labelling tasks exported from unresolved TrackSight evidence and policy prompts.",
        )
    else:
        st.success("No unresolved evidence or policy prompts need labelling in this run.", icon=":material/check_circle:")
    export = build_learning_export(artifacts["run_name"], tasks)
    st.download_button(
        "Download labelling manifest", data=json.dumps(export, indent=2),
        file_name=f"{artifacts['run_name']}_human_labelling_queue.json", mime="application/json",
        icon=":material/download:", help="Exports annotation tasks and portable clip names, not labels or ground truth.",
    )


def render_telemetry_correlation(artifacts, review_events, risk_events):
    """Align an optional user-supplied telemetry CSV to immutable video evidence."""
    st.header("Telemetry correlation", icon=":material/sensors:")
    st.warning(TELEMETRY_DISCLAIMER, icon=":material/warning:")
    st.caption("Upload a CSV whose `time_seconds` column uses the same clock as the video. Other numeric columns can be named freely, for example speed, brake, throttle, steering, or GPS fields.")
    telemetry_file = st.file_uploader("Telemetry CSV", type=["csv"], key="telemetry_csv")
    if telemetry_file is None:
        st.info("No telemetry file is attached. TrackSight will not invent vehicle data.", icon=":material/info:")
        return
    try:
        telemetry_rows = parse_telemetry_csv(telemetry_file.getvalue().decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as error:
        st.error(str(error), icon=":material/error:")
        return
    review_queue = _current_review_queue(artifacts, review_events, risk_events)
    event_rows = review_queue.to_dict("records")
    if not event_rows:
        st.info(
            "This run has no saved evidence events to align with telemetry.",
            icon=":material/info:",
        )
        return
    correlations = correlate_telemetry_events(event_rows, telemetry_rows)
    st.session_state["telemetry_correlations"] = correlations
    st.session_state["telemetry_correlation_run_name"] = artifacts["run_name"]
    aligned = [row for row in correlations if row["alignment_status"] == "aligned"]
    with st.container(horizontal=True):
        st.metric("Telemetry samples", len(telemetry_rows), border=True)
        st.metric("Saved evidence events", len(event_rows), border=True)
        st.metric("Aligned within 0.50 s", len(aligned), border=True)
    st.dataframe(
        pd.DataFrame(correlations), hide_index=True,
        column_config={
            "event_key": "Evidence event", "event_time_seconds": st.column_config.NumberColumn("Evidence time (s)", format="%.3f"),
            "telemetry_time_seconds": st.column_config.NumberColumn("Telemetry time (s)", format="%.3f"),
            "time_delta_seconds": st.column_config.NumberColumn("Alignment difference (s)", format="%.3f"),
            "telemetry_values": st.column_config.JsonColumn("Telemetry values"), "alignment_status": "Alignment status",
        }, alt="Telemetry samples aligned to saved TrackSight evidence only when their timestamps are within the configured window.",
    )
    st.download_button(
        "Download telemetry correlation", data=json.dumps({"disclaimer": TELEMETRY_DISCLAIMER, "correlations": correlations}, indent=2),
        file_name=f"{artifacts['run_name']}_telemetry_correlation.json", mime="application/json", icon=":material/download:",
    )


def render_multi_camera_review(artifacts, review_events, risk_events):
    """Review a selected evidence time across reviewer-synchronised camera files."""
    st.header("Multi-camera review", icon=":material/video_camera_back:")
    st.warning(MULTI_CAMERA_DISCLAIMER, icon=":material/warning:")
    st.caption("Add camera files and their manually verified time offsets in a manifest. The current annotated camera remains the evidence source; additional videos provide context only.")
    example_manifest_path = CONFIG_DIRECTORY / "multi_camera_manifest.example.json"
    if example_manifest_path.is_file():
        st.download_button(
            "Download multi-camera manifest template",
            data=example_manifest_path.read_bytes(),
            file_name=example_manifest_path.name,
            mime="application/json",
            icon=":material/download:",
            help="Download a template, replace the example path and offset, then upload it here.",
        )
    manifest_file = st.file_uploader("Multi-camera manifest JSON", type=["json"], key="multi_camera_manifest")
    if manifest_file is None:
        st.info("No camera manifest is attached. TrackSight is intentionally showing only the recorded source camera.", icon=":material/info:")
        return
    try:
        manifest = validate_multi_camera_manifest(json.loads(manifest_file.getvalue()))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        st.error(str(error), icon=":material/error:")
        return
    st.session_state["multi_camera_manifest"] = manifest
    timeline = build_race_control_timeline_dataframe(review_events, risk_events)
    if timeline.empty:
        st.info("This run has no saved evidence event to synchronize across cameras.", icon=":material/info:")
        return
    selected_index = st.selectbox(
        "Evidence moment", timeline.index.tolist(),
        format_func=lambda index: f"{timeline.loc[index, 'evidence_type']} at {timeline.loc[index, 'time_seconds']:.2f}s",
        key="multi_camera_event",
    )
    event = timeline.loc[selected_index]
    playback = build_multi_camera_playback(event["time_seconds"], manifest)
    st.video(str(artifacts["annotated_video"]), start_time=max(0.0, float(event["time_seconds"]) - 2.0), alt="Primary annotated camera video at the selected evidence moment.")
    for camera in playback:
        with st.container(border=True):
            st.subheader(camera["label"])
            st.caption(f"Configured offset: {camera['video_time_offset_seconds']:+.2f}s; requested playback: {camera['playback_time_seconds']:.2f}s.")
            if camera["video_available"]:
                st.video(camera["video_path"], start_time=camera["playback_time_seconds"], alt=f"Supporting camera {camera['label']} at the configured synchronized evidence time.")
            else:
                st.info("The configured camera file is not available on this computer, so no video is shown.", icon=":material/videocam_off:")


def render_race_control_export(artifacts, review_events, risk_events, run_summary):
    """Offer a chronological, portable review export for race control."""
    st.header("Race-control export", icon=":material/file_download:")
    st.warning(RACE_CONTROL_EXPORT_DISCLAIMER, icon=":material/warning:")
    queue = _current_review_queue(artifacts, review_events, risk_events)
    report = build_race_control_timeline_export(artifacts["run_name"], run_summary, queue.to_dict("records"))
    with st.container(horizontal=True):
        st.metric("Timeline entries", len(report["timeline"]), border=True)
        st.metric("Unreviewed", sum(item["review_status"] == "Unreviewed" for item in report["timeline"]), border=True)
        st.metric("Escalated", sum(item["review_status"] == "Escalate" for item in report["timeline"]), border=True)
    if report["timeline"]:
        st.subheader("Timeline preview")
        st.dataframe(
            pd.DataFrame(report["timeline"]),
            hide_index=True,
            column_config={
                "evidence_type": "Evidence type",
                "track_ids": "Vehicle IDs",
                "frame": "Frame",
                "time_seconds": st.column_config.NumberColumn("Video time (s)", format="%.2f"),
                "measurement_value": st.column_config.NumberColumn("Measured separation", format="%.3f"),
                "measurement_basis": "Measurement basis",
                "review_status": "Human review status",
                "supporting_clip": "Supporting clip",
            },
            alt="Chronological race-control evidence export preview with current human-review statuses.",
        )
    else:
        st.info("This run has no saved evidence events to export.", icon=":material/info:")
    st.download_button(
        "Download race-control timeline", data=json.dumps(report, indent=2),
        file_name=f"{artifacts['run_name']}_race_control_timeline.json", mime="application/json", icon=":material/download:",
        help="Exports chronological source evidence and current human-review status without local paths.",
    )


def _session_telemetry_correlations(artifacts):
    """Return only telemetry aligned during this browser session for this run."""
    if st.session_state.get("telemetry_correlation_run_name") != artifacts["run_name"]:
        return []
    return st.session_state.get("telemetry_correlations", [])


def render_safety_command_center(
    artifacts, review_events, risk_events, run_summary, run_is_quality_assessed
):
    """Bring one quality-gated case, its evidence, and its human action into one view."""
    st.header("Safety command center", icon=":material/assistant_direction:")
    st.warning(DECISION_LADDER_DISCLAIMER, icon=":material/warning:")
    if not run_is_quality_assessed:
        st.info("A completed quality-assessed run is required before TrackSight can build a review recommendation.", icon=":material/info:")
        return
    timeline = _current_review_queue(artifacts, review_events, risk_events)
    if timeline.empty:
        st.info("This run has no saved evidence case to bring into the command center.", icon=":material/info:")
        return
    threads = build_incident_threads(timeline.to_dict("records"))
    cases = build_review_priority_queue(threads, run_summary)
    selected_case_id = st.selectbox(
        "Review case",
        [case["case_id"] for case in cases],
        format_func=lambda case_id: next(
            f"{case['priority']} — vehicles {case['track_ids_label']} at {case['start_time_seconds']:.2f}s"
            for case in cases if case["case_id"] == case_id
        ),
        key="safety_command_center_case",
    )
    selected_case = next(case for case in cases if case["case_id"] == selected_case_id)
    case_rows = timeline[timeline["event_key"].isin(selected_case["source_event_keys"])].sort_values("time_seconds")
    ladder = build_safety_decision_ladder(
        selected_case,
        run_summary,
        case_rows.to_dict("records"),
        artifacts["annotated_video"].is_file(),
    )
    with st.container(horizontal=True):
        st.metric("Decision ladder", ladder["status"], border=True)
        st.metric("Vehicles", selected_case["track_ids_label"], border=True)
        st.metric("Evidence records", len(case_rows), border=True)
        st.metric("Video time", f"{selected_case['start_time_seconds']:.2f} s", border=True)
    st.caption(ladder["recommended_action"])
    st.subheader("Why this recommendation", icon=":material/fact_check:")
    st.dataframe(
        pd.DataFrame(ladder["checks"]),
        column_config={
            "evidence_source": "Evidence source",
            "status": "Recorded status",
            "detail": "What this means",
        },
        hide_index=True,
        alt="Decision-ladder checks for the selected TrackSight evidence case.",
    )
    with st.container(border=True):
        st.subheader("Replay and evidence", icon=":material/play_circle:")
        st.dataframe(
            case_rows[["evidence_type", "track_ids", "time_seconds", "frame", "measurement_basis", "review_status"]],
            column_config={
                "evidence_type": "Evidence type",
                "track_ids": "Vehicle IDs",
                "time_seconds": st.column_config.NumberColumn("Video time (s)", format="%.2f"),
                "frame": "Frame",
                "measurement_basis": "Measurement basis",
                "review_status": "Human review status",
            },
            hide_index=True,
            alt="Time-local source evidence and current human-review statuses for the selected command-center case.",
        )
        clip_path = selected_case.get("preferred_clip")
        if isinstance(clip_path, str) and Path(clip_path).is_file():
            st.video(clip_path, alt="Annotated supporting clip for the selected command-center case.")
        elif artifacts["annotated_video"].is_file():
            st.video(
                str(artifacts["annotated_video"]),
                start_time=max(0.0, selected_case["start_time_seconds"] - 2.0),
                alt="Annotated primary camera video starting shortly before the selected command-center case.",
            )
        else:
            st.error("The recorded annotated video is unavailable, so replay-dependent review is withheld.", icon=":material/error:")
    st.subheader("Evidence correlation", icon=":material/hub:")
    st.caption(EVENT_CORRELATION_DISCLAIMER)
    correlation = build_event_correlation(
        case_rows.to_dict("records"),
        run_summary,
        artifacts["annotated_video"],
        _session_telemetry_correlations(artifacts),
        st.session_state.get("multi_camera_manifest"),
    )
    st.dataframe(
        pd.DataFrame(correlation),
        column_config={"source": "Evidence source", "status": "Availability", "detail": "Review context"},
        hide_index=True,
        alt="Available and unavailable evidence sources correlated to the selected TrackSight review case.",
    )
    st.download_button(
        "Download after-action report",
        data=build_after_action_report_pdf(
            artifacts["run_name"], selected_case, ladder, correlation, case_rows.to_dict("records")
        ),
        file_name=f"{artifacts['run_name']}_{selected_case['case_id']}_after_action_report.pdf",
        mime="application/pdf",
        icon=":material/picture_as_pdf:",
        help="Downloads a concise, path-free handoff report for the selected evidence case.",
    )


def render_recurring_review_map(selected_trajectory_path, available_runs):
    """Show same-calibration recurring review locations across selected completed sessions."""
    st.header("Recurring review map", icon=":material/map:")
    st.warning(
        "This view groups repeated candidate-evidence locations for human review. It does not identify a dangerous corner, prove a track defect, or rank drivers.",
        icon=":material/warning:",
    )
    baseline = build_review_workspace(get_run_artifacts(selected_trajectory_path))
    compatible_paths = []
    for candidate_path in available_runs:
        if candidate_path == selected_trajectory_path:
            continue
        candidate_summary = load_saved_json(get_run_artifacts(candidate_path)["run_summary_json"])
        candidate_snapshot = build_session_snapshot(
            get_run_artifacts(candidate_path)["run_name"], candidate_summary, [], []
        )
        if build_session_comparison(baseline["snapshot"], candidate_snapshot)["comparable"]:
            compatible_paths.append(candidate_path)
    if not compatible_paths:
        st.info("No other completed run shares this run's recorded camera calibration, so recurring locations are withheld.", icon=":material/info:")
        return
    selected_paths = st.multiselect(
        "Compatible completed runs to include",
        compatible_paths,
        default=compatible_paths,
        format_func=lambda path: path.name.removesuffix("_trajectories.json"),
        key="recurring_review_map_runs",
        help="Only runs with the same saved calibration are available because local-track coordinates from different cameras are not comparable.",
    )
    if not selected_paths:
        st.info("Choose at least one compatible completed run to compare with the current run.", icon=":material/info:")
        return
    workspaces = [baseline] + [build_review_workspace(get_run_artifacts(path)) for path in selected_paths]
    recurring, locations, message = build_multi_run_risk_map([workspace["snapshot"] for workspace in workspaces])
    st.caption(message)
    if not recurring:
        st.info("No same-type evidence display group occurred across all selected sessions within the adaptive neighborhood.", icon=":material/info:")
        return
    recurring_frame = pd.DataFrame(recurring)
    chart = (
        alt.Chart(recurring_frame)
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X("track_x:Q", title="Local track X"),
            y=alt.Y("track_y:Q", title="Local track Y", scale=alt.Scale(reverse=True)),
            color=alt.Color("evidence_type:N", title="Evidence type"),
            size=alt.Size("event_count:Q", title="Evidence records"),
            tooltip=[
                alt.Tooltip("evidence_type:N", title="Evidence type"),
                alt.Tooltip("run_count:Q", title="Completed runs"),
                alt.Tooltip("event_count:Q", title="Evidence records"),
                alt.Tooltip("track_x:Q", title="Local X", format=".1f"),
                alt.Tooltip("track_y:Q", title="Local Y", format=".1f"),
            ],
        )
        .properties(height=440)
    )
    st.altair_chart(chart, width="stretch", alt="Recurring same-calibration evidence display groups across selected completed TrackSight sessions.")
    st.dataframe(
        recurring_frame[["evidence_type", "run_count", "event_count", "track_x", "track_y"]],
        column_config={
            "evidence_type": "Evidence type",
            "run_count": "Completed runs",
            "event_count": "Evidence records",
            "track_x": st.column_config.NumberColumn("Local track X", format="%.1f"),
            "track_y": st.column_config.NumberColumn("Local track Y", format="%.1f"),
        },
        hide_index=True,
        alt="Recurring candidate-evidence display groups across compatible calibrated TrackSight sessions.",
    )


def render_reviewer_corrections(artifacts, review_events, risk_events):
    """Record human feedback for later independent annotation, without altering evidence."""
    st.header("Reviewer corrections", icon=":material/rate_review:")
    st.warning(REVIEW_CORRECTIONS_DISCLAIMER, icon=":material/warning:")
    timeline = _current_review_queue(artifacts, review_events, risk_events)
    if timeline.empty:
        st.info("This run has no evidence item to correct.", icon=":material/info:")
        return
    try:
        corrections = load_review_corrections(artifacts["review_corrections_json"])
    except ValueError as error:
        st.error(str(error), icon=":material/error:")
        return
    selected_key = st.selectbox(
        "Evidence item to correct",
        timeline["event_key"].tolist(),
        format_func=lambda event_key: (
            lambda row: f"{row['evidence_type']} — vehicles {row['track_ids']} at {row['time_seconds']:.2f}s"
        )(timeline.loc[timeline["event_key"] == event_key].iloc[0]),
        key="review_correction_event",
    )
    selected_event = timeline.loc[timeline["event_key"] == selected_key].iloc[0]
    with st.form("review_correction_form"):
        reviewer_name = st.text_input("Reviewer name", max_chars=100)
        correction_type = st.selectbox("Correction type", CORRECTION_TYPES)
        annotation_note = st.text_area(
            "What should a human annotator verify?",
            max_chars=2000,
            placeholder="Describe the frame-level issue without assuming a model label is correct.",
        )
        saved = st.form_submit_button("Save reviewer correction", type="primary", icon=":material/save:")
    if saved:
        try:
            corrections = append_review_correction(
                corrections, selected_event, correction_type, reviewer_name, annotation_note
            )
            write_review_corrections(corrections, artifacts["review_corrections_json"])
        except ValueError as error:
            st.error(str(error), icon=":material/error:")
        else:
            st.success("Reviewer correction saved separately from the immutable evidence. It still needs independent annotation before training.", icon=":material/check_circle:")
            st.rerun()
    if corrections["corrections"]:
        correction_frame = pd.DataFrame(corrections["corrections"])
        correction_frame["event"] = correction_frame["evidence_event"].map(
            lambda event: f"{event['evidence_type']} — vehicle IDs {', '.join(map(str, event['track_ids']))} at {event['time_seconds']:.2f}s"
        )
        st.dataframe(
            correction_frame[["recorded_at", "reviewer_name", "correction_type", "event", "annotation_note"]],
            column_config={
                "recorded_at": "Recorded at",
                "reviewer_name": "Reviewer",
                "correction_type": "Correction type",
                "event": "Evidence item",
                "annotation_note": "Annotation note",
            },
            hide_index=True,
            alt="Append-only human reviewer corrections associated with saved TrackSight evidence items.",
        )
    else:
        st.info("No reviewer correction has been recorded for this run yet.", icon=":material/info:")
    st.download_button(
        "Download correction handoff",
        data=json.dumps(build_correction_learning_export(artifacts["run_name"], corrections), indent=2),
        file_name=f"{artifacts['run_name']}_reviewer_corrections.json",
        mime="application/json",
        icon=":material/download:",
        help="Exports human notes for independent annotation; it does not create training labels.",
    )


def _load_audit_records(artifacts):
    """Load human-controlled records that are intentionally separate from evidence."""
    return (
        load_human_review_decisions(artifacts["human_review_decisions_json"]),
        load_race_control_action_log(artifacts["race_control_action_log_json"]),
        load_review_corrections(artifacts["review_corrections_json"]),
        load_second_opinions(artifacts["second_opinions_json"]),
    )


def render_decision_ledger(artifacts, review_events, risk_events):
    """Show the current decision history without changing source evidence."""
    st.header("Decision ledger", icon=":material/history:")
    st.warning(DECISION_LEDGER_DISCLAIMER, icon=":material/warning:")
    try:
        decisions, action_log, corrections, second_opinions = _load_audit_records(artifacts)
    except ValueError as error:
        st.error(str(error), icon=":material/error:")
        return
    timeline = build_race_control_timeline_dataframe(review_events, risk_events)
    current_event_summaries = {
        event["review_event_id"]: get_human_review_event_summary(event)
        for _, event in timeline.iterrows()
    }
    current_decisions = filter_current_human_review_decisions(
        decisions, current_event_summaries
    )
    stale_decision_count = len(decisions["decisions"]) - len(current_decisions["decisions"])
    ledger = build_decision_ledger(
        current_decisions, action_log, corrections, second_opinions
    )
    with st.container(horizontal=True):
        st.metric("Decision versions", sum(len(item.get("history", [])) or 1 for item in decisions["decisions"].values()), border=True)
        st.metric("Race-control actions", len(action_log["actions"]), border=True)
        st.metric("Reviewer corrections", len(corrections["corrections"]), border=True)
        st.metric("Independent opinions", len(second_opinions["opinions"]), border=True)
    if stale_decision_count:
        st.info(
            f"{stale_decision_count} stored human decision(s) are withheld because the source evidence changed after reprocessing.",
            icon=":material/info:",
        )
    if not ledger:
        st.info("No human review record has been saved for this run yet.", icon=":material/info:")
        return
    st.dataframe(
        pd.DataFrame(ledger),
        column_config={
            "recorded_at": "Recorded at",
            "record_type": "Record type",
            "event_reference": "Evidence reference",
            "reviewer": "Reviewer",
            "decision_or_action": "Decision or action",
            "note": "Reviewer note",
            "recording_mode": "Audit status",
        },
        hide_index=True,
        alt="Chronological human review, race-control action, correction, and independent second-opinion records for this TrackSight run.",
    )
    st.download_button(
        "Download decision ledger",
        data=json.dumps({
            "ledger_type": "tracksight_decision_ledger",
            "run_name": artifacts["run_name"],
            "disclaimer": DECISION_LEDGER_DISCLAIMER,
            "records": ledger,
        }, indent=2),
        file_name=f"{artifacts['run_name']}_decision_ledger.json",
        mime="application/json",
        icon=":material/download:",
        help="Downloads saved human-controlled records without local computer paths.",
    )


def render_evidence_integrity(artifacts):
    """Show output availability first and calculate hashes only on a human request."""
    st.header("Evidence integrity", icon=":material/verified_user:")
    st.warning(EVIDENCE_INTEGRITY_DISCLAIMER, icon=":material/warning:")
    state_key = f"evidence_integrity_{artifacts['run_name']}"
    if st.button(
        "Calculate saved-output hashes",
        icon=":material/fingerprint:",
        key=f"calculate_{state_key}",
        help="Reads each saved output file to calculate SHA-256 checksums. This can take time for a long video.",
    ):
        st.session_state[state_key] = build_evidence_integrity(artifacts, include_hashes=True)
    rows = st.session_state.get(state_key)
    if rows is None:
        rows = build_evidence_integrity(artifacts, include_hashes=False)
        st.caption("File presence is shown below. Select Calculate saved-output hashes to create a point-in-time checksum manifest.")
    present_count = sum(row["status"] == "Present" for row in rows)
    missing_required = sum(row["status"] == "Missing" for row in rows)
    with st.container(horizontal=True):
        st.metric("Present output artifacts", present_count, border=True)
        st.metric("Missing required artifacts", missing_required, border=True)
        st.metric("Hashes", "Calculated" if any(len(str(row["sha256"])) == 64 for row in rows) else "Not calculated", border=True)
    st.dataframe(
        pd.DataFrame(rows),
        column_config={
            "artifact": "Artifact",
            "status": "Availability",
            "file_name": "File name",
            "size_bytes": st.column_config.NumberColumn("Size (bytes)", format="%d"),
            "sha256": "SHA-256",
            "required_for_saved_run": "Required for saved run",
        },
        hide_index=True,
        alt="Saved TrackSight output files, their availability, and optional point-in-time SHA-256 checksums.",
    )
    st.download_button(
        "Download integrity manifest",
        data=json.dumps(build_integrity_manifest(artifacts["run_name"], rows), indent=2),
        file_name=f"{artifacts['run_name']}_integrity_manifest.json",
        mime="application/json",
        icon=":material/download:",
        help="Downloads portable output availability and any calculated SHA-256 checksums.",
    )


def render_operational_coverage(trajectories, run_summary):
    """Display observation coverage and withheld frames without inventing accuracy."""
    st.header("Operational coverage", icon=":material/timeline:")
    st.warning(OPERATIONAL_COVERAGE_DISCLAIMER, icon=":material/warning:")
    rows = build_operational_coverage_timeline(trajectories, run_summary)
    summary = summarise_operational_coverage(rows)
    with st.container(horizontal=True):
        st.metric("Processed frames", summary["processed_frames"], border=True)
        st.metric("Frames with positions", summary["frames_with_positions"], border=True)
        st.metric("Pair-review eligible frames", summary["pair_review_eligible_frames"], border=True)
        st.metric("Spatially withheld frames", summary["withheld_frames"], border=True)
    coverage = pd.DataFrame(rows)
    chart = (
        alt.Chart(coverage)
        .transform_fold(
            ["detector_positions", "trusted_positions"],
            as_=["observation_type", "positions"],
        )
        .mark_line()
        .encode(
            x=alt.X("time_seconds:Q", title="Video time (s)"),
            y=alt.Y("positions:Q", title="Recorded positions"),
            color=alt.Color("observation_type:N", title="Observation type"),
            tooltip=[
                alt.Tooltip("time_seconds:Q", title="Video time (s)", format=".2f"),
                alt.Tooltip("observation_type:N", title="Observation type"),
                alt.Tooltip("positions:Q", title="Recorded positions"),
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(
        chart,
        width="stretch",
        alt="Detector and continuity-qualified vehicle positions over the completed video timeline.",
    )
    st.dataframe(
        coverage,
        column_config={
            "frame": "Frame",
            "time_seconds": st.column_config.NumberColumn("Video time (s)", format="%.2f"),
            "detector_positions": "Detector positions",
            "trusted_positions": "Trusted positions",
            "untrusted_positions": "Untrusted positions",
            "mean_detector_confidence": st.column_config.NumberColumn("Mean detector confidence", format="%.2f"),
            "operational_status": "Recorded review status",
        },
        hide_index=True,
        alt="Frame-by-frame detector positions and ID-continuity eligibility across the completed TrackSight video.",
    )


def render_independent_second_review(artifacts, review_events, risk_events):
    """Collect a second named opinion only after a primary human decision exists."""
    st.header("Independent second review", icon=":material/groups:")
    st.warning(SECOND_OPINION_DISCLAIMER, icon=":material/warning:")
    try:
        decisions = load_human_review_decisions(artifacts["human_review_decisions_json"])
        opinions = load_second_opinions(artifacts["second_opinions_json"])
    except ValueError as error:
        st.error(str(error), icon=":material/error:")
        return
    timeline = build_race_control_timeline_dataframe(review_events, risk_events)
    if timeline.empty:
        st.info("This run has no saved evidence item for independent review.", icon=":material/info:")
        return
    review_queue = build_human_review_queue(timeline, decisions)
    candidates = review_queue[review_queue["review_status"] != "Unreviewed"]
    if candidates.empty:
        st.info("Record a named primary human-review decision before requesting an independent second opinion.", icon=":material/info:")
        return
    selected_event_id = st.selectbox(
        "Primary decision to independently review",
        candidates["review_event_id"].tolist(),
        format_func=lambda event_id: (
            lambda row: f"{row['review_status']} — {row['evidence_type']} for vehicles {row['track_ids']} at {row['time_seconds']:.2f}s"
        )(candidates.loc[candidates["review_event_id"] == event_id].iloc[0]),
        key="independent_second_review_event",
    )
    event = candidates.loc[candidates["review_event_id"] == selected_event_id].iloc[0]
    primary_decision = decisions["decisions"][selected_event_id]
    st.caption(
        f"Primary decision: {primary_decision['outcome']} by {primary_decision['reviewer_name']}. A second opinion cannot overwrite it."
    )
    with st.form(f"independent_second_review_form_{selected_event_id}"):
        reviewer_name = st.text_input("Independent reviewer name", max_chars=100)
        opinion_type = st.selectbox("Independent review outcome", SECOND_OPINION_TYPES)
        note = st.text_area(
            "Independent review note",
            max_chars=2000,
            placeholder="State what you reviewed and why you agree, disagree, or need more evidence.",
        )
        save_opinion = st.form_submit_button("Append independent opinion", type="primary", icon=":material/person_add:")
    if save_opinion:
        try:
            opinions = append_second_opinion(
                opinions, event, primary_decision, opinion_type, reviewer_name, note
            )
            write_second_opinions(opinions, artifacts["second_opinions_json"])
        except ValueError as error:
            st.error(str(error), icon=":material/error:")
        else:
            st.success("Independent opinion appended. The primary decision and source evidence were not changed.", icon=":material/check_circle:")
            st.rerun()
    matching = [
        opinion for opinion in opinions["opinions"]
        if opinion["evidence_event"]["review_event_id"] == selected_event_id
    ]
    if matching:
        st.dataframe(
            pd.DataFrame(matching)[["recorded_at", "reviewer_name", "opinion_type", "note"]],
            column_config={
                "recorded_at": "Recorded at",
                "reviewer_name": "Independent reviewer",
                "opinion_type": "Opinion",
                "note": "Review note",
            },
            hide_index=True,
            alt="Append-only independent second-review opinions for the selected TrackSight evidence item.",
        )
    else:
        st.info("No independent second opinion has been recorded for this evidence item.", icon=":material/info:")


def render_grounded_case_assistant(artifacts, review_events, risk_events, run_summary):
    """Explain a selected case through fixed questions grounded in saved records."""
    st.header("Grounded case assistant", icon=":material/quiz:")
    st.warning(GROUNDED_ASSISTANT_DISCLAIMER, icon=":material/warning:")
    timeline = _current_review_queue(artifacts, review_events, risk_events)
    if timeline.empty:
        st.info("This run has no saved evidence case to explain.", icon=":material/info:")
        return
    cases = build_review_priority_queue(
        build_incident_threads(timeline.to_dict("records")), run_summary
    )
    selected_case_id = st.selectbox(
        "Evidence case",
        [case["case_id"] for case in cases],
        format_func=lambda case_id: next(
            f"Vehicles {case['track_ids_label']} at {case['start_time_seconds']:.2f}s"
            for case in cases if case["case_id"] == case_id
        ),
        key="grounded_case_assistant_case",
    )
    case = next(case for case in cases if case["case_id"] == selected_case_id)
    case_rows = timeline[timeline["event_key"].isin(case["source_event_keys"])].sort_values("time_seconds")
    ladder = build_safety_decision_ladder(
        case, run_summary, case_rows.to_dict("records"), artifacts["annotated_video"].is_file()
    )
    correlation = build_event_correlation(
        case_rows.to_dict("records"), run_summary, artifacts["annotated_video"],
        _session_telemetry_correlations(artifacts), st.session_state.get("multi_camera_manifest"),
    )
    try:
        decisions, action_log, corrections, second_opinions = _load_audit_records(artifacts)
    except ValueError as error:
        st.error(str(error), icon=":material/error:")
        return
    current_event_summaries = {
        event["review_event_id"]: get_human_review_event_summary(event)
        for _, event in timeline.iterrows()
    }
    current_decisions = filter_current_human_review_decisions(
        decisions, current_event_summaries
    )
    ledger = build_decision_ledger(
        current_decisions, action_log, corrections, second_opinions
    )
    case_references = {
        case["case_id"],
        *case["source_event_keys"],
        *case["source_review_event_ids"],
    }
    case_ledger = [
        record for record in ledger
        if record["event_reference"] in case_references
    ]
    question = st.selectbox("Ask a saved-evidence question", GROUNDED_QUESTIONS, key="grounded_case_assistant_question")
    answer = build_grounded_case_answer(
        question, case, ladder, correlation, case_rows.to_dict("records"), case_ledger
    )
    with st.container(border=True):
        st.subheader("Grounded answer", icon=":material/fact_check:")
        st.write(answer["answer"])
        st.caption("Sources used: " + ", ".join(answer["sources"]))


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
    package_path = artifacts["marshal_evidence_package"]
    if package_path.is_file():
        st.download_button(
            "Download Marshal evidence package",
            data=package_path.read_bytes(),
            file_name=package_path.name,
            mime="application/zip",
            help=(
                "Downloads this run's portable briefing, evidence records, "
                "trajectory data, and supporting clips without local directory paths."
            ),
            key="download_marshal_evidence_package",
        )
    else:
        st.error(
            "This completed run is missing its portable evidence package. Reprocess the run before sharing it.",
            icon=":material/error:",
        )
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
        st.metric(
            "Safety candidates",
            summary.get("stationary_vehicle_candidate_count", 0)
            + summary.get("restricted_area_entry_candidate_count", 0)
            + summary.get("stationary_in_active_racing_area_candidate_count", 0)
            + summary.get("against_expected_direction_candidate_count", 0),
            border=True,
        )
    st.caption(
        "Proximity lists stay separate because their measurements use different valid normalizations. Safety candidates are separate again because they use local-track movement or explicit zone-policy evidence."
    )
    for title, highlights in (
        ("Predicted-risk review highlights", briefing["predicted_risk_highlights"]),
        ("Observed-proximity review highlights", briefing["observed_proximity_highlights"]),
        ("Safety-candidate review highlights", briefing.get("safety_candidate_highlights", [])),
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
                measurement = highlight["measurement"]
                if measurement["value"] is None:
                    st.caption(f"Evidence basis: {measurement['basis']}.")
                else:
                    st.caption(
                        f"Measurement: {measurement['value']:.3f} "
                        f"{measurement['basis']}."
                    )
                if highlight.get("evidence_grounding"):
                    st.info(
                        highlight["evidence_grounding"],
                        icon=":material/verified:",
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


def build_review_workspace(artifacts):
    """Load one completed run into the conservative dashboard review shape."""
    trajectories = load_saved_json(artifacts["trajectory_json"])
    run_summary = load_saved_json(artifacts["run_summary_json"])
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
    timeline = build_race_control_timeline_dataframe(review_events, risk_events)
    threads = build_incident_threads(timeline.to_dict("records")) if not timeline.empty else []
    locations = build_calibrated_event_locations(
        timeline.to_dict("records"), trajectories
    ) if run_summary.get("spatial_analysis_enabled") else []
    hotspots, _ = build_candidate_hotspots(locations) if locations else ([], None)
    snapshot = build_session_snapshot(
        artifacts["run_name"], run_summary, locations, hotspots
    )
    return {
        "trajectories": trajectories,
        "summary": run_summary,
        "review_events": review_events,
        "risk_events": risk_events,
        "timeline": timeline,
        "threads": threads,
        "locations": locations,
        "hotspots": hotspots,
        "snapshot": snapshot,
    }


def render_camera_trust_monitor(run_summary):
    """Show exactly which camera and identity safeguards a run recorded."""
    st.header("Camera trust monitor", icon=":material/video_settings:")
    monitor = build_camera_trust_monitor(run_summary)
    st.warning(
        "This view reports recorded evidence safeguards, not detector accuracy or a probability that the camera is correct.",
        icon=":material/warning:",
    )
    with st.container(horizontal=True):
        st.metric("Review mode", monitor["overall_status"], border=True)
        st.metric(
            "Calibration",
            "Available" if monitor["spatial_analysis_enabled"] else "Unavailable",
            border=True,
        )
        st.metric(
            "Trusted ID segments",
            f"{monitor['trusted_segments']} / {monitor['all_segments']}",
            border=True,
        )
        st.metric("Camera mode", (run_summary.get("run_provenance") or {}).get("camera_mode", "unknown"), border=True)
    st.dataframe(
        pd.DataFrame(monitor["checks"]),
        column_config={
            "check": "Safeguard",
            "status": "Saved status",
            "detail": "Why it matters",
        },
        hide_index=True,
        alt="Recorded camera calibration, camera mode, ID continuity, and stabilization safeguards for the selected TrackSight run.",
    )
    stabilization = monitor["stabilization"]
    if stabilization["recorded"] and "updated_frames" in stabilization:
        with st.container(horizontal=True):
            st.metric("Stabilization updates", stabilization["updated_frames"], border=True)
            st.metric("Reliable transform retained", stabilization["retained_frames"], border=True)
        st.caption(
            "When the current moving frame lacks enough reliable background evidence, TrackSight retains the last reliable transform instead of fabricating a new camera correction."
        )
    elif not stabilization["recorded"]:
        st.info(stabilization["detail"], icon=":material/info:")
    else:
        st.success(stabilization["detail"], icon=":material/check_circle:")


def render_live_review_simulation(
    artifacts,
    review_events,
    risk_events,
    run_summary,
    run_is_quality_assessed,
):
    """Replay saved outputs as a clearly labelled race-control timing demo."""
    st.header("Live race-control simulation", icon=":material/live_tv:")
    st.warning(
        "Replay simulation only: this view reads saved TrackSight output and does not claim live inference, live telemetry, or an automatic race-control decision.",
        icon=":material/warning:",
    )
    if not run_is_quality_assessed or not artifacts["annotated_video"].is_file():
        st.info("A completed verified annotated video is required for the replay simulation.")
        return
    timeline = build_race_control_timeline_dataframe(review_events, risk_events)
    if timeline.empty:
        st.info("This run did not save candidate evidence to surface during the replay.")
        return
    fps = run_summary.get("frames_per_second", 0.0)
    duration_seconds = (
        run_summary.get("processed_frame_count", 0) / fps if fps else 0.0
    )
    replay_time = st.slider(
        "Replay clock",
        min_value=0.0,
        max_value=max(duration_seconds, 0.1),
        value=0.0,
        step=0.1,
        format="%.1f seconds",
        key="live_review_replay_clock",
        help="Move the replay clock to see only evidence that would have been available by that source-video time.",
    )
    surfaced = timeline[timeline["time_seconds"] <= replay_time].sort_values("time_seconds")
    upcoming = timeline[timeline["time_seconds"] > replay_time].sort_values("time_seconds")
    with st.container(horizontal=True):
        st.metric("Replay time", f"{replay_time:.1f} s", border=True)
        st.metric("Evidence surfaced", len(surfaced), border=True)
        st.metric("Still ahead", len(upcoming), border=True)
    playback_data = timeline.copy()
    playback_data["availability"] = playback_data["time_seconds"].map(
        lambda time_seconds: "Surfaced" if time_seconds <= replay_time else "Upcoming"
    )
    playback_chart = (
        alt.Chart(playback_data)
        .mark_point(filled=True, size=90)
        .encode(
            x=alt.X("time_seconds:Q", title="Source-video time (s)"),
            y=alt.Y("evidence_type:N", title="Evidence stream"),
            color=alt.Color("evidence_type:N", title="Evidence type"),
            opacity=alt.condition(
                alt.datum.availability == "Surfaced", alt.value(1.0), alt.value(0.18)
            ),
            tooltip=[
                alt.Tooltip("availability:N", title="Replay availability"),
                alt.Tooltip("evidence_type:N", title="Evidence type"),
                alt.Tooltip("track_ids:N", title="Vehicle IDs"),
                alt.Tooltip("time_seconds:Q", title="Video time (s)", format=".2f"),
            ],
        )
        .properties(height=210)
    )
    st.altair_chart(
        playback_chart,
        width="stretch",
        alt="Saved candidate evidence across the video, with events up to the selected replay clock shown as surfaced.",
    )
    if upcoming.empty:
        st.success("The replay has reached the end of this completed video.", icon=":material/check_circle:")
    else:
        next_event = upcoming.iloc[0]
        st.info(
            f"Next saved evidence: {next_event['evidence_type']} involving vehicle ID(s) {next_event['track_ids']} at {next_event['time_seconds']:.2f}s.",
            icon=":material/schedule:",
        )
    st.video(
        str(artifacts["annotated_video"]),
        start_time=replay_time,
        alt="Annotated TrackSight camera video starting at the selected replay-simulation time.",
    )
    if not surfaced.empty:
        st.dataframe(
            surfaced[["evidence_type", "track_ids", "time_seconds", "frame", "measurement_basis"]],
            column_config={
                "evidence_type": "Evidence type",
                "track_ids": "Vehicle IDs",
                "time_seconds": st.column_config.NumberColumn("Surfaced at (s)", format="%.2f"),
                "frame": "Frame",
                "measurement_basis": "Measurement basis",
            },
            hide_index=True,
            alt="Evidence records that have surfaced by the selected replay-simulation time.",
        )


def render_race_control_desk(
    artifacts,
    trajectories,
    review_events,
    risk_events,
    run_summary,
    run_is_quality_assessed,
    spatial_analysis_enabled,
):
    """Offer an auditable human work queue without automated race decisions."""
    st.header("Race-control desk", icon=":material/admin_panel_settings:")
    st.warning(
        "TrackSight prioritises what a human should inspect. It never confirms contact, fault, a rule breach, or an incident outcome.",
        icon=":material/warning:",
    )
    if not run_is_quality_assessed or not spatial_analysis_enabled:
        st.info(
            "The race-control desk needs a completed quality-assessed run with camera calibration. This prevents uncalibrated image positions from becoming operational review cases.",
            icon=":material/videocam_off:",
        )
        return
    timeline = build_race_control_timeline_dataframe(review_events, risk_events)
    if timeline.empty:
        st.info("This completed run has no saved evidence records to prioritise.")
        return
    cases = build_review_priority_queue(
        build_incident_threads(timeline.to_dict("records")), run_summary
    )
    with st.container(horizontal=True):
        st.metric("Review now", sum(case["priority"] == REVIEW_NOW for case in cases), border=True)
        st.metric("Monitor", sum(case["priority"] == MONITOR for case in cases), border=True)
        st.metric("Insufficient evidence", sum(case["priority"] == INSUFFICIENT_EVIDENCE for case in cases), border=True)
    st.caption(
        "Priority comes from the continuity-quality gate and repeated time-local evidence for the exact same pair. It is deliberately not a severity score."
    )
    case_scope = st.segmented_control(
        "Cases to display",
        ["Actionable cases", "All cases"],
        default="Actionable cases",
        key="race_control_case_scope",
        help="Actionable cases have passed TrackSight's saved evidence-quality gate. All cases retains withheld records for audit.",
    )
    visible_cases = (
        [case for case in cases if case["priority"] != INSUFFICIENT_EVIDENCE]
        if case_scope == "Actionable cases"
        else cases
    )
    if not visible_cases:
        visible_cases = cases
        st.info("No case passed the quality gate, so all saved cases are shown for audit.")
    queue = pd.DataFrame([
        {
            "priority": case["priority"],
            "vehicles": case["track_ids_label"],
            "start_seconds": case["start_time_seconds"],
            "end_seconds": case["end_time_seconds"],
            "evidence_records": case["event_count"],
            "reason": case["reason"],
        }
        for case in visible_cases
    ])
    st.dataframe(
        queue,
        column_config={
            "priority": "Review priority",
            "vehicles": "Vehicle IDs",
            "start_seconds": st.column_config.NumberColumn("Start (s)", format="%.2f"),
            "end_seconds": st.column_config.NumberColumn("End (s)", format="%.2f"),
            "evidence_records": "Evidence records",
            "reason": "Why it is queued",
        },
        hide_index=True,
        alt="Quality-gated TrackSight review priorities, ordered for human race-control review.",
    )
    selected_case_id = st.selectbox(
        "Open priority case",
        [case["case_id"] for case in visible_cases],
        format_func=lambda case_id: next(
            f"{case['priority']} — vehicles {case['track_ids_label']} at {case['start_time_seconds']:.2f}s"
            for case in visible_cases if case["case_id"] == case_id
        ),
        key="race_control_priority_case",
    )
    selected_case = next(case for case in visible_cases if case["case_id"] == selected_case_id)
    selected_rows = timeline[timeline["event_key"].isin(selected_case["source_event_keys"])].sort_values("time_seconds")
    evidence_chain = build_evidence_chain(
        selected_case,
        selected_rows.to_dict("records"),
        run_summary,
        artifacts["annotated_video"].is_file(),
    )
    with st.container(border=True):
        st.subheader("Case review material", icon=":material/manage_search:")
        st.write(selected_case["reason"])
        st.caption(selected_case["recommended_human_action"])
        st.dataframe(
            selected_rows[["evidence_type", "track_ids", "time_seconds", "frame", "measurement_basis"]],
            column_config={
                "evidence_type": "Evidence type", "track_ids": "Vehicles",
                "time_seconds": st.column_config.NumberColumn("Video time (s)", format="%.2f"),
                "frame": "Frame", "measurement_basis": "Measurement basis",
            }, hide_index=True,
            alt="Saved source evidence records for the selected race-control priority case.",
        )
        clip_path = selected_case.get("preferred_clip")
        if isinstance(clip_path, str) and Path(clip_path).is_file():
            st.video(clip_path, alt="Annotated camera clip supporting the selected race-control priority case.")
        else:
            st.info("No separate clip was saved. The full annotated camera video starts shortly before this review case.")
            st.video(str(artifacts["annotated_video"]), start_time=max(0.0, selected_case["start_time_seconds"] - 2.0), alt="Annotated camera video supporting the selected race-control priority case.")
    st.subheader("Evidence chain", icon=":material/account_tree:")
    st.caption(
        "This is the review trace for this case. A withheld or missing stage is visible rather than filled with an inferred result."
    )
    chain_dataframe = pd.DataFrame(evidence_chain)
    st.dataframe(
        chain_dataframe[["order", "stage", "status", "detail"]],
        column_config={
            "order": "Step",
            "stage": "Evidence stage",
            "status": "Status",
            "detail": "What this stage means",
        },
        hide_index=True,
        alt="Ordered evidence chain showing source video, calibration, ID continuity, evidence records, replay, and required human action for the selected case.",
    )
    try:
        action_log = load_race_control_action_log(artifacts["race_control_action_log_json"])
    except ValueError as error:
        st.error(f"The saved race-control action log cannot be used: {error}", icon=":material/error:")
        return
    with st.form(f"race_control_action_{selected_case_id}"):
        st.subheader("Record human race-control action", icon=":material/edit_note:")
        operator = st.text_input("Operator name", max_chars=100)
        action = st.selectbox("Action", RACE_CONTROL_ACTIONS)
        note = st.text_area("Action note", max_chars=2000, placeholder="State what was reviewed and why this human action was taken.")
        st.caption("Required review checklist")
        review_checklist = {
            "supporting_video_reviewed": st.checkbox(
                "I reviewed the supporting camera video or saved clip.",
                key=f"video_check_{selected_case_id}",
            ),
            "identity_continuity_considered": st.checkbox(
                "I considered the saved vehicle-ID continuity evidence.",
                key=f"identity_check_{selected_case_id}",
            ),
            "camera_trust_reviewed": st.checkbox(
                "I reviewed the camera-trust status for this run.",
                key=f"camera_check_{selected_case_id}",
            ),
        }
        save_action = st.form_submit_button("Append action to log", type="primary", icon=":material/add_task:")
    if save_action:
        try:
            action_log = append_race_control_action(
                action_log,
                selected_case,
                action,
                operator,
                note,
                review_checklist,
            )
            write_race_control_action_log(action_log, artifacts["race_control_action_log_json"])
        except ValueError as error:
            st.error(str(error), icon=":material/error:")
        else:
            st.success("Human action appended. Existing evidence and prior actions were not changed.")
            st.rerun()
    st.subheader("Append-only action history", icon=":material/history:")
    if action_log["actions"]:
        action_history = pd.DataFrame(action_log["actions"])
        saved_checklists = action_history.get(
            "review_checklist",
            pd.Series([None] * len(action_history), index=action_history.index),
        )
        action_history["review_checks"] = saved_checklists.map(
            lambda checklist: (
                "Complete"
                if isinstance(checklist, dict)
                and all(checklist.get(check) is True for check in REQUIRED_REVIEW_CHECKS)
                else "Not recorded (legacy)"
            )
        )
        st.dataframe(
            action_history[[
                "recorded_at",
                "action",
                "track_ids_label",
                "priority_snapshot",
                "operator_name",
                "review_checks",
                "action_note",
            ]],
            column_config={
                "recorded_at": "Recorded at",
                "action": "Human action",
                "track_ids_label": "Vehicle IDs",
                "priority_snapshot": "Case priority",
                "operator_name": "Operator",
                "review_checks": "Required review checks",
                "action_note": "Operator note",
            },
            hide_index=True,
            alt="Timestamped human race-control actions, including whether the mandatory video, ID-continuity, and camera-trust checks were recorded.",
        )
    else:
        st.info("No human race-control action has been recorded for this run.")
    st.download_button("Download action log", data=json.dumps(build_race_control_action_report(artifacts["run_name"], action_log), indent=2), file_name=f"{artifacts['run_name']}_race_control_action_log.json", mime="application/json", icon=":material/download:")
    st.download_button("Download one-page race-control report", data=build_race_control_report_pdf(artifacts["run_name"], run_summary, cases, action_log), file_name=f"{artifacts['run_name']}_race_control_review_report.pdf", mime="application/pdf", icon=":material/picture_as_pdf:")
    st.caption(RACE_CONTROL_LOG_DISCLAIMER)


def render_session_comparison(selected_trajectory_path, available_runs):
    """Compare saved local evidence only across one explicitly compatible camera view."""
    st.header("Session comparison", icon=":material/compare_arrows:")
    st.caption("Compare two completed sessions only when they share the recorded camera calibration. This view reports evidence counts and recurring locations; it does not rank driver performance or declare a dangerous section of track.")
    alternatives = [path for path in available_runs if path != selected_trajectory_path]
    if not alternatives:
        st.info("Create another completed calibrated run before comparing sessions.")
        return
    baseline = build_review_workspace(get_run_artifacts(selected_trajectory_path))
    compatible_paths = []
    for candidate_path in alternatives:
        candidate_artifacts = get_run_artifacts(candidate_path)
        candidate_summary = load_saved_json(candidate_artifacts["run_summary_json"])
        candidate_snapshot = build_session_snapshot(
            candidate_artifacts["run_name"], candidate_summary, [], []
        )
        if build_session_comparison(baseline["snapshot"], candidate_snapshot)["comparable"]:
            compatible_paths.append(candidate_path)
    if not compatible_paths:
        st.info(
            "No other completed run shares this run's recorded camera calibration. "
            "TrackSight will not compare unrelated local-track coordinate systems.",
            icon=":material/info:",
        )
        return
    st.caption(
        "Only completed runs with the same recorded camera calibration are offered. "
        "Runs from another camera remain available separately for their own review."
    )
    comparison_path = st.selectbox(
        "Comparison run",
        compatible_paths,
        format_func=lambda path: path.name.removesuffix("_trajectories.json"),
        key="session_comparison_run",
    )
    comparison = build_review_workspace(get_run_artifacts(comparison_path))
    report = build_session_comparison(baseline["snapshot"], comparison["snapshot"])
    if not report["comparable"]:
        st.info(report["message"], icon=":material/info:")
        return
    st.success(report["message"], icon=":material/check_circle:")
    st.dataframe(pd.DataFrame(report["metrics"]), hide_index=True, alt="Auditable count comparison for two compatible calibrated TrackSight sessions.")
    recurring, locations, map_message = build_multi_run_risk_map([baseline["snapshot"], comparison["snapshot"]])
    st.caption(map_message)
    if locations:
        location_frame = pd.DataFrame(locations)
        chart = alt.Chart(location_frame).mark_point(filled=True, size=85, opacity=0.65).encode(
            x=alt.X("track_x:Q", title="Local track X"),
            y=alt.Y("track_y:Q", title="Local track Y", scale=alt.Scale(reverse=True)),
            color=alt.Color("run_name:N", title="Completed run"),
            shape=alt.Shape("evidence_type:N", title="Evidence type"),
            tooltip=["run_name:N", "evidence_type:N", alt.Tooltip("time_seconds:Q", title="Video time (s)", format=".2f")],
        ).properties(height=430)
        st.altair_chart(chart, width="stretch", alt="Quality-gated local evidence positions from two compatible completed TrackSight sessions.")
    if recurring:
        st.dataframe(pd.DataFrame(recurring)[["evidence_type", "run_count", "event_count", "track_x", "track_y"]], hide_index=True, alt="Recurring local evidence display groups found across compatible completed sessions.")
    else:
        st.info("No same-type local evidence location occurred in both selected runs within the adaptive display neighborhood.")


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
            "Low-displacement candidates": run_summary.get(
                "stationary_vehicle_candidate_count", "not recorded"
            ),
            "Restricted-area candidates": run_summary.get(
                "restricted_area_entry_candidate_count", "not recorded"
            ),
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
                "Stationary-candidate duration": run_provenance.get(
                    "minimum_stationary_seconds", "not recorded"
                ),
                "Stationary-candidate spread": run_provenance.get(
                    "maximum_stationary_spread_vehicle_widths", "not recorded"
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
        ("Download safety-candidate data", "safety_candidate_evidence_json"),
        ("Download Marshal briefing", "marshal_briefing_json"),
        ("Download Marshal evidence package", "marshal_evidence_package"),
        ("Download run summary", "run_summary_json"),
    ):
        artifact_path = artifacts[artifact_key]
        if artifact_path.is_file():
            st.download_button(
                label,
                data=artifact_path.read_bytes(),
                file_name=artifact_path.name,
                mime=(
                    "application/zip"
                    if artifact_key == "marshal_evidence_package"
                    else "application/json"
                ),
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
    pending_review_run = st.session_state.pop("dashboard_pending_review_run", None)
    if pending_review_run is not None:
        st.session_state["dashboard_workspace"] = "Review completed run"
        st.session_state["dashboard_selected_run"] = Path(pending_review_run)

    default_workspace = (
        "Review completed run" if available_runs else "Run analysis"
    )
    workspace_arguments = {}
    if "dashboard_workspace" not in st.session_state:
        workspace_arguments["default"] = default_workspace
    workspace = st.segmented_control(
        "Workspace",
        ["Run analysis", "Review completed run"],
        key="dashboard_workspace",
        **workspace_arguments,
    )
    if workspace == "Run analysis":
        render_run_analysis_launcher()
        return
    if not available_runs:
        st.info(
            "There is no completed run to review yet. Use Run analysis to create one."
        )
        return

    if st.session_state.get("dashboard_selected_run") not in available_runs:
        st.session_state["dashboard_selected_run"] = available_runs[0]

    with st.sidebar:
        st.header("Run selection", icon=":material/folder_open:")
        selected_trajectory_path = st.selectbox(
            "Completed run",
            available_runs,
            format_func=lambda path: path.name.removesuffix("_trajectories.json"),
            key="dashboard_selected_run",
        )
        st.caption("Review a completed, verified TrackSight run.")

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
    safety_zone_evidence = (
        load_saved_json(artifacts["safety_zone_evidence_json"])
        if artifacts["safety_zone_evidence_json"].is_file()
        else None
    )
    safety_candidate_evidence = (
        load_saved_json(artifacts["safety_candidate_evidence_json"])
        if artifacts["safety_candidate_evidence_json"].is_file()
        else None
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
    available_views = get_available_review_views(
        artifacts,
        review_events,
        risk_events,
        run_summary,
        safety_zone_evidence,
        safety_candidate_evidence,
        len(available_runs),
    )
    st.caption(
        "This menu only includes tools supported by the selected run's saved evidence."
    )
    view = st.selectbox(
        "Review view",
        available_views,
        format_func=lambda view_name: REVIEW_VIEW_LABELS[view_name],
        key=f"dashboard_review_view_{artifacts['run_name']}",
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
    elif view == "Race-control timeline":
        render_race_control_timeline(
            artifacts,
            review_events,
            risk_events,
            run_is_quality_assessed,
            run_summary.get("spatial_analysis_enabled", False),
            run_summary,
        )
    elif view == "Incident replay":
        render_incident_replay(
            artifacts,
            trajectories,
            review_events,
            risk_events,
            run_is_quality_assessed,
            run_summary.get("spatial_analysis_enabled", False),
        )
    elif view == "Evidence quality":
        render_evidence_quality(trajectories, run_summary)
    elif view == "Camera trust monitor":
        render_camera_trust_monitor(run_summary)
    elif view == "Safety command center":
        render_safety_command_center(
            artifacts,
            review_events,
            risk_events,
            run_summary,
            run_is_quality_assessed,
        )
    elif view == "Candidate hotspots":
        render_candidate_hotspots(
            artifacts,
            trajectories,
            review_events,
            risk_events,
            run_is_quality_assessed,
            run_summary.get("spatial_analysis_enabled", False),
        )
    elif view == "Incident threads":
        render_incident_threads(
            artifacts,
            review_events,
            risk_events,
            run_summary,
            run_is_quality_assessed,
            run_summary.get("spatial_analysis_enabled", False),
            safety_zone_evidence,
        )
    elif view == "Live race-control simulation":
        render_live_review_simulation(
            artifacts,
            review_events,
            risk_events,
            run_summary,
            run_is_quality_assessed,
        )
    elif view == "Race-control desk":
        render_race_control_desk(
            artifacts,
            trajectories,
            review_events,
            risk_events,
            run_summary,
            run_is_quality_assessed,
            run_summary.get("spatial_analysis_enabled", False),
        )
    elif view == "Session comparison":
        render_session_comparison(selected_trajectory_path, available_runs)
    elif view == "Recurring review map":
        render_recurring_review_map(selected_trajectory_path, available_runs)
    elif view == "Safety zones":
        render_safety_zones(
            artifacts,
            safety_zone_evidence,
            run_is_quality_assessed,
            run_summary.get("spatial_analysis_enabled", False),
        )
    elif view == "Safety candidates":
        render_safety_candidates(
            artifacts,
            safety_candidate_evidence,
            run_summary,
            run_is_quality_assessed,
        )
    elif view == "Learning queue":
        render_learning_queue(
            artifacts, review_events, risk_events, safety_candidate_evidence
        )
    elif view == "Reviewer corrections":
        render_reviewer_corrections(artifacts, review_events, risk_events)
    elif view == "Independent second review":
        render_independent_second_review(artifacts, review_events, risk_events)
    elif view == "Decision ledger":
        render_decision_ledger(artifacts, review_events, risk_events)
    elif view == "Evidence integrity":
        render_evidence_integrity(artifacts)
    elif view == "Operational coverage":
        render_operational_coverage(trajectories, run_summary)
    elif view == "Grounded case assistant":
        render_grounded_case_assistant(
            artifacts, review_events, risk_events, run_summary
        )
    elif view == "Telemetry correlation":
        render_telemetry_correlation(artifacts, review_events, risk_events)
    elif view == "Multi-camera review":
        render_multi_camera_review(artifacts, review_events, risk_events)
    elif view == "Race-control export":
        render_race_control_export(artifacts, review_events, risk_events, run_summary)
    elif view == "Human review decisions":
        render_human_review_workflow(
            artifacts,
            review_events,
            risk_events,
            run_is_quality_assessed,
            run_summary.get("spatial_analysis_enabled", False),
        )
    elif view == "Marshal briefing":
        render_marshal_briefing(artifacts, run_is_quality_assessed)
    elif view == "Marshal handover":
        render_marshal_handover(
            artifacts,
            trajectories,
            review_events,
            risk_events,
            safety_candidate_evidence,
            run_summary,
            run_is_quality_assessed,
        )
    elif view == "Evaluation and limits":
        render_evaluation_and_limits(
            artifacts,
            trajectories,
            run_summary,
            safety_candidate_evidence,
            run_is_quality_assessed,
        )
    else:
        render_raw_data(artifacts, run_summary)


if __name__ == "__main__":
    main()
