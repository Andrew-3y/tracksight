import cv2
import numpy as np


TRACK_COLORS = [
    (0, 255, 0),
    (255, 255, 0),
    (0, 165, 255),
    (255, 0, 255),
    (255, 255, 255),
]


def get_track_color(track_id):
    return TRACK_COLORS[(track_id - 1) % len(TRACK_COLORS)]


def create_top_down_canvas(track_map_size):
    """Create a neutral local-map frame instead of warping broadcast imagery."""
    map_width, map_height = track_map_size
    canvas = np.full((map_height, map_width, 3), (20, 24, 30), dtype=np.uint8)
    cv2.putText(
        canvas,
        "TrackSight local map — vehicle footpoints only",
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (230, 230, 230),
        2,
    )
    cv2.putText(
        canvas,
        "Confirmed detector observations only",
        (24, map_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (190, 190, 190),
        1,
    )
    return canvas


def draw_top_down_marker(top_down_frame, track_id, track_position):
    map_height, map_width = top_down_frame.shape[:2]
    position_x, position_y = track_position
    marker_x = min(max(round(position_x), 0), map_width - 1)
    marker_y = min(max(round(position_y), 0), map_height - 1)
    color = get_track_color(track_id)
    cv2.circle(top_down_frame, (marker_x, marker_y), 8, color, -1)
    cv2.putText(
        top_down_frame,
        f"ID {track_id}",
        (marker_x + 10, max(20, marker_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )
    return top_down_frame
