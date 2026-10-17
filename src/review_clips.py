from pathlib import Path

import cv2

from src.video_output import create_working_video_writer, finalize_browser_video


def select_review_events(
    review_evidence,
    fps,
    pre_event_seconds,
    post_event_seconds,
    max_review_clips,
    anchor_frame_key="closest_frame",
    priority_key="closest_normalized_distance",
):
    if fps <= 0:
        raise ValueError("Video frames per second must be greater than zero.")
    if pre_event_seconds < 0 or post_event_seconds < 0:
        raise ValueError("Review-clip durations cannot be negative.")
    if max_review_clips < 1:
        raise ValueError("At least one review clip must be allowed.")

    pre_event_frames = round(pre_event_seconds * fps)
    post_event_frames = round(post_event_seconds * fps)
    selected_events = []
    selected_windows = []
    for event in sorted(
        review_evidence,
        key=lambda event: event[priority_key],
    ):
        start_frame = max(0, event[anchor_frame_key] - pre_event_frames)
        end_frame = event[anchor_frame_key] + post_event_frames
        overlaps_selected_window = any(
            start_frame <= selected_end and end_frame >= selected_start
            for selected_start, selected_end in selected_windows
        )
        if overlaps_selected_window:
            continue
        selected_events.append((event, start_frame, end_frame))
        selected_windows.append((start_frame, end_frame))
        if len(selected_events) == max_review_clips:
            break
    return selected_events


def create_review_clips(
    annotated_video_path,
    review_evidence,
    output_directory,
    pre_event_seconds,
    post_event_seconds,
    max_review_clips,
    anchor_frame_key="closest_frame",
    priority_key="closest_normalized_distance",
    clip_prefix="review",
):
    video_capture = cv2.VideoCapture(str(annotated_video_path))
    if not video_capture.isOpened():
        raise FileNotFoundError(
            f"Could not open annotated video for review clips: {annotated_video_path}"
        )
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    selected_events = select_review_events(
        review_evidence,
        fps,
        pre_event_seconds,
        post_event_seconds,
        max_review_clips,
        anchor_frame_key,
        priority_key,
    )
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    review_clip_paths = []
    for clip_number, (event, start_frame, end_frame) in enumerate(
        selected_events,
        start=1,
    ):
        track_id_a, track_id_b = event["track_ids"]
        review_clip_path = output_directory / (
            f"{clip_prefix}_{clip_number:02d}_tracks_{track_id_a}_{track_id_b}_"
            f"frame_{event[anchor_frame_key]}.mp4"
        )
        video_writer, working_clip_path = create_working_video_writer(
            review_clip_path,
            fps,
            (frame_width, frame_height),
        )
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(start_frame, end_frame + 1):
            frame_read_success, frame = video_capture.read()
            if not frame_read_success:
                break
            video_writer.write(frame)
        video_writer.release()
        finalize_browser_video(working_clip_path, review_clip_path)
        event["review_clip_path"] = str(review_clip_path)
        review_clip_paths.append(review_clip_path)
    video_capture.release()
    return review_clip_paths
