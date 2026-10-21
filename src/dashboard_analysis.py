"""Safe helpers for starting one TrackSight run from the local dashboard."""

import hashlib
import re
import subprocess
import sys
from pathlib import Path


SUPPORTED_VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
RUN_LABEL_PATTERN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")


def list_available_videos(video_directory):
    """Return locally available videos in a stable, readable order."""
    video_directory = Path(video_directory)
    if not video_directory.is_dir():
        return []
    return sorted(
        (
            video_path
            for video_path in video_directory.iterdir()
            if video_path.is_file() and video_path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES
        ),
        key=lambda video_path: video_path.name.lower(),
    )


def get_matching_profile_path(video_path, config_directory, profile_suffix=""):
    """Return only the configuration explicitly named for this source video."""
    video_path = Path(video_path)
    profile_path = Path(config_directory) / f"{video_path.stem}{profile_suffix}.json"
    return profile_path if profile_path.is_file() else None


def normalize_run_label(run_label):
    """Turn a human run label into a safe, predictable output-name component."""
    normalized_label = re.sub(r"[^a-z0-9]+", "_", run_label.lower()).strip("_")
    if not RUN_LABEL_PATTERN.fullmatch(normalized_label):
        raise ValueError(
            "Run label must contain at least one letter or number."
        )
    return normalized_label


def make_dashboard_output_path(output_directory, run_label):
    """Reserve a non-overwriting annotated-video path inside tracking outputs."""
    output_directory = Path(output_directory)
    normalized_label = normalize_run_label(run_label)
    output_path = output_directory / f"annotated_{normalized_label}.mp4"
    existing_artifacts = list(output_directory.glob(f"annotated_{normalized_label}*"))
    if existing_artifacts:
        raise FileExistsError(
            "That run label already has saved output. Choose a new run label so "
            "existing evidence is preserved."
        )
    return output_path


def persist_uploaded_video(upload_name, upload_bytes, upload_directory):
    """Store one uploaded video under a content-addressed local filename."""
    upload_name = Path(upload_name).name
    suffix = Path(upload_name).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_SUFFIXES:
        raise ValueError("Upload a supported video file: MP4, WebM, MOV, MKV, or AVI.")
    if not upload_bytes:
        raise ValueError("The uploaded video is empty.")
    safe_stem = re.sub(r"[^a-z0-9]+", "_", Path(upload_name).stem.lower()).strip("_")
    if not safe_stem:
        safe_stem = "uploaded_video"
    file_digest = hashlib.sha256(upload_bytes).hexdigest()[:12]
    upload_path = Path(upload_directory) / f"{safe_stem}_{file_digest}{suffix}"
    try:
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        if not upload_path.is_file():
            upload_path.write_bytes(upload_bytes)
    except OSError as error:
        raise RuntimeError(f"Could not save the uploaded video: {error}") from error
    return upload_path


def build_tracking_command(
    project_root,
    video_path,
    model_path,
    output_path,
    regions_config_path=None,
    calibration_config_path=None,
    safety_zone_config_path=None,
    python_executable=None,
):
    """Build a shell-free command for the established TrackSight runner."""
    project_root = Path(project_root).resolve()
    video_path = Path(video_path).resolve()
    model_path = Path(model_path).resolve()
    output_path = Path(output_path).resolve()
    for input_path, label in ((video_path, "video"), (model_path, "model")):
        if not input_path.is_file():
            raise FileNotFoundError(f"Could not find selected {label}: {input_path.name}")
    if output_path.parent != (project_root / "outputs" / "tracking").resolve():
        raise ValueError("Dashboard analysis output must stay in outputs/tracking.")
    command = [
        python_executable or sys.executable,
        "-m",
        "src.run_video_tracking",
        str(video_path),
        "--model-path",
        str(model_path),
        "--output-path",
        str(output_path),
    ]
    for config_path, command_option, label in (
        (regions_config_path, "--regions-config", "race-area profile"),
        (calibration_config_path, "--calibration-config", "calibration profile"),
        (safety_zone_config_path, "--safety-zone-config", "safety-zone profile"),
    ):
        if config_path is not None:
            config_path = Path(config_path).resolve()
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"Could not find selected {label}: {config_path.name}"
                )
            command.extend((command_option, str(config_path)))
    return command


def run_tracking_command(command, project_root, command_runner=subprocess.run):
    """Run the existing CLI pipeline and surface a concise, actionable failure."""
    try:
        completed_process = command_runner(
            command,
            cwd=Path(project_root),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as error:
        raise RuntimeError(f"Could not start local TrackSight analysis: {error}") from error
    if completed_process.returncode != 0:
        error_output = (completed_process.stderr or "").strip()
        if len(error_output) > 3000:
            error_output = error_output[-3000:]
        raise RuntimeError(error_output or "TrackSight analysis stopped without an error message.")
    return completed_process
