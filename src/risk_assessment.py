import json
from itertools import combinations
from math import hypot


def calculate_predicted_closest_approach(
    point_a,
    point_b,
    prediction_horizon_seconds,
):
    if prediction_horizon_seconds <= 0:
        raise ValueError("Prediction horizon must be positive.")

    position_a_x, position_a_y = point_a["track_position"]
    position_b_x, position_b_y = point_b["track_position"]
    velocity_a_x = point_a["motion"]["velocity_x_track_units_per_second"]
    velocity_a_y = point_a["motion"]["velocity_y_track_units_per_second"]
    velocity_b_x = point_b["motion"]["velocity_x_track_units_per_second"]
    velocity_b_y = point_b["motion"]["velocity_y_track_units_per_second"]

    relative_position_x = position_b_x - position_a_x
    relative_position_y = position_b_y - position_a_y
    relative_velocity_x = velocity_b_x - velocity_a_x
    relative_velocity_y = velocity_b_y - velocity_a_y
    relative_speed_squared = (
        relative_velocity_x**2 + relative_velocity_y**2
    )

    if relative_speed_squared == 0:
        time_to_closest_approach = 0.0
    else:
        time_to_closest_approach = -(
            relative_position_x * relative_velocity_x
            + relative_position_y * relative_velocity_y
        ) / relative_speed_squared
        time_to_closest_approach = min(
            prediction_horizon_seconds,
            max(0.0, time_to_closest_approach),
        )

    predicted_relative_x = (
        relative_position_x
        + relative_velocity_x * time_to_closest_approach
    )
    predicted_relative_y = (
        relative_position_y
        + relative_velocity_y * time_to_closest_approach
    )
    predicted_separation = hypot(predicted_relative_x, predicted_relative_y)
    current_separation = hypot(relative_position_x, relative_position_y)
    if current_separation == 0:
        closing_speed = 0.0
    else:
        closing_speed = max(
            0.0,
            -(
                relative_position_x * relative_velocity_x
                + relative_position_y * relative_velocity_y
            ) / current_separation,
        )

    average_vehicle_width = (
        point_a["track_width_units"] + point_b["track_width_units"]
    ) / 2
    if average_vehicle_width <= 0:
        raise ValueError("Vehicle track widths must be positive.")
    return {
        "time_to_closest_approach_seconds": time_to_closest_approach,
        "current_separation_track_units": current_separation,
        "predicted_separation_track_units": predicted_separation,
        "predicted_separation_vehicle_widths": (
            predicted_separation / average_vehicle_width
        ),
        "closing_speed_track_units_per_second": closing_speed,
    }


def find_predicted_close_approach_evidence(
    trajectories,
    prediction_horizon_seconds,
    maximum_separation_vehicle_widths,
    minimum_consecutive_frames,
):
    if maximum_separation_vehicle_widths <= 0:
        raise ValueError("Maximum predicted separation must be positive.")
    if minimum_consecutive_frames < 1:
        raise ValueError("Minimum consecutive frames must be at least one.")

    points_by_frame = {}
    for track_id, trajectory in trajectories.items():
        for point in trajectory:
            if (
                point.get("motion", {}).get("status") != "measured"
                or "track_width_units" not in point
            ):
                continue
            points_by_frame.setdefault(point["frame_index"], []).append(
                (track_id, point)
            )

    active_events = {}
    completed_events = []
    for frame_index in sorted(points_by_frame):
        frame_points = points_by_frame[frame_index]
        next_active_events = {}
        for (track_id_a, point_a), (track_id_b, point_b) in combinations(
            frame_points,
            2,
        ):
            metrics = calculate_predicted_closest_approach(
                point_a,
                point_b,
                prediction_horizon_seconds,
            )
            if (
                metrics["predicted_separation_vehicle_widths"]
                > maximum_separation_vehicle_widths
            ):
                continue

            track_ids = tuple(sorted((track_id_a, track_id_b)))
            previous_event = active_events.get(track_ids)
            if (
                previous_event is not None
                and previous_event["end_frame"] == frame_index - 1
            ):
                event = previous_event
                event["end_frame"] = frame_index
                event["end_time_seconds"] = point_a["time_seconds"]
                event["frame_count"] += 1
            else:
                if previous_event is not None:
                    completed_events.append(previous_event)
                event = {
                    "risk_type": "predicted_close_approach",
                    "track_ids": list(track_ids),
                    "start_frame": frame_index,
                    "end_frame": frame_index,
                    "start_time_seconds": point_a["time_seconds"],
                    "end_time_seconds": point_a["time_seconds"],
                    "frame_count": 1,
                    "highest_risk_frame": frame_index,
                    "highest_risk_time_seconds": point_a["time_seconds"],
                    **metrics,
                }

            if (
                metrics["predicted_separation_vehicle_widths"]
                < event["predicted_separation_vehicle_widths"]
            ):
                event["highest_risk_frame"] = frame_index
                event["highest_risk_time_seconds"] = point_a["time_seconds"]
                event.update(metrics)
            next_active_events[track_ids] = event

        for track_ids, active_event in active_events.items():
            if track_ids not in next_active_events:
                completed_events.append(active_event)
        active_events = next_active_events

    completed_events.extend(active_events.values())
    return [
        event
        for event in completed_events
        if event["frame_count"] >= minimum_consecutive_frames
    ]


def save_risk_evidence(risk_evidence, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(risk_evidence, output_file, indent=2)
    return output_path
