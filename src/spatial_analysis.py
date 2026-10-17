from math import atan2, degrees, hypot


def calculate_track_motion(previous_point, current_point):
    previous_x, previous_y = previous_point["track_position"]
    current_x, current_y = current_point["track_position"]
    elapsed_seconds = current_point["time_seconds"] - previous_point["time_seconds"]
    if elapsed_seconds <= 0:
        raise ValueError("Motion points must have increasing timestamps.")

    horizontal_change = current_x - previous_x
    vertical_change = current_y - previous_y
    distance_track_units = hypot(horizontal_change, vertical_change)
    return {
        "status": "measured",
        "elapsed_seconds": elapsed_seconds,
        "distance_track_units": distance_track_units,
        "speed_track_units_per_second": distance_track_units / elapsed_seconds,
        "velocity_x_track_units_per_second": horizontal_change / elapsed_seconds,
        "velocity_y_track_units_per_second": vertical_change / elapsed_seconds,
        "heading_degrees": degrees(atan2(vertical_change, horizontal_change)),
    }


def add_motion_measurements(trajectories, maximum_gap_seconds):
    if maximum_gap_seconds <= 0:
        raise ValueError("Maximum motion gap must be positive.")

    for trajectory in trajectories.values():
        previous_calibrated_point = None
        for point in trajectory:
            if "track_position" not in point:
                point["motion"] = {"status": "unavailable_no_calibration"}
                previous_calibrated_point = None
                continue

            if previous_calibrated_point is None:
                point["motion"] = {"status": "insufficient_history"}
                previous_calibrated_point = point
                continue

            elapsed_seconds = (
                point["time_seconds"]
                - previous_calibrated_point["time_seconds"]
            )
            if elapsed_seconds > maximum_gap_seconds:
                point["motion"] = {
                    "status": "gap_too_large",
                    "elapsed_seconds": elapsed_seconds,
                }
            else:
                point["motion"] = calculate_track_motion(
                    previous_calibrated_point,
                    point,
                )
            previous_calibrated_point = point

    return trajectories
