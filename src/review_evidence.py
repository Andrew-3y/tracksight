import json
from itertools import combinations
from math import hypot


def get_box_diagonal(point):
    box_width = point.get("box_width")
    box_height = point.get("box_height")
    if box_width is None or box_height is None:
        return None
    return hypot(box_width, box_height)


def get_center_distance(point_a, point_b):
    center_x_a, center_y_a = point_a["center"]
    center_x_b, center_y_b = point_b["center"]
    return hypot(center_x_b - center_x_a, center_y_b - center_y_a)


def find_close_approach_evidence(
    trajectories,
    proximity_ratio,
    minimum_consecutive_frames,
):
    if proximity_ratio <= 0:
        raise ValueError("Proximity ratio must be greater than zero.")
    if minimum_consecutive_frames < 1:
        raise ValueError("Minimum consecutive frames must be at least one.")

    points_by_frame = {}
    for track_id, trajectory in trajectories.items():
        for point in trajectory:
            frame_index = point["frame_index"]
            points_by_frame.setdefault(frame_index, []).append((track_id, point))

    active_events = {}
    completed_events = []
    for frame_index in sorted(points_by_frame):
        close_pairs = {}
        for (track_id_a, point_a), (track_id_b, point_b) in combinations(
            points_by_frame[frame_index],
            2,
        ):
            diagonal_a = get_box_diagonal(point_a)
            diagonal_b = get_box_diagonal(point_b)
            if diagonal_a is None or diagonal_b is None:
                continue
            average_diagonal = (diagonal_a + diagonal_b) / 2
            if average_diagonal == 0:
                continue
            center_distance = get_center_distance(point_a, point_b)
            normalized_distance = center_distance / average_diagonal
            if normalized_distance > proximity_ratio:
                continue
            pair_key = tuple(sorted((track_id_a, track_id_b)))
            close_pairs[pair_key] = {
                "frame_index": frame_index,
                "time_seconds": point_a["time_seconds"],
                "center_distance_pixels": center_distance,
                "normalized_distance": normalized_distance,
            }

        next_active_events = {}
        for pair_key, close_pair in close_pairs.items():
            previous_event = active_events.get(pair_key)
            if previous_event is not None and previous_event["end_frame"] == frame_index - 1:
                event = previous_event
                event["end_frame"] = frame_index
                event["end_seconds"] = close_pair["time_seconds"]
                event["frame_count"] += 1
            else:
                if previous_event is not None:
                    completed_events.append(previous_event)
                event = {
                    "track_ids": list(pair_key),
                    "start_frame": frame_index,
                    "end_frame": frame_index,
                    "start_seconds": close_pair["time_seconds"],
                    "end_seconds": close_pair["time_seconds"],
                    "frame_count": 1,
                    "closest_frame": frame_index,
                    "closest_time_seconds": close_pair["time_seconds"],
                    "closest_center_distance_pixels": close_pair["center_distance_pixels"],
                    "closest_normalized_distance": close_pair["normalized_distance"],
                }
            if close_pair["normalized_distance"] < event["closest_normalized_distance"]:
                event["closest_frame"] = frame_index
                event["closest_time_seconds"] = close_pair["time_seconds"]
                event["closest_center_distance_pixels"] = close_pair["center_distance_pixels"]
                event["closest_normalized_distance"] = close_pair["normalized_distance"]
            next_active_events[pair_key] = event

        for pair_key, event in active_events.items():
            if pair_key not in next_active_events:
                completed_events.append(event)
        active_events = next_active_events

    completed_events.extend(active_events.values())
    return [
        event
        for event in completed_events
        if event["frame_count"] >= minimum_consecutive_frames
    ]


def save_review_evidence(review_evidence, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(review_evidence, output_file, indent=2)
    return output_path
