from math import sqrt

import numpy as np
from ultralytics.trackers.utils.matching import linear_assignment

from src.detect_vehicles import calculate_iou
from src.reidentification import calculate_appearance_distance


def get_box_center(box):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y

def calculate_distance(point_a, point_b):
    x_a, y_a = point_a
    x_b, y_b = point_b
    horizontal_distance = abs(x_a - x_b)
    vertical_distance = abs(y_a - y_b)
    distance = sqrt(horizontal_distance**2 + vertical_distance**2)
    return distance


def get_predicted_center(track):
    center_x, center_y = track["center"]
    velocity_x, velocity_y = track["velocity"]
    next_frame_count = track["missing_frames"] + 1
    return (
        center_x + velocity_x * next_frame_count,
        center_y + velocity_y * next_frame_count,
    )


def shift_box(box, horizontal_shift, vertical_shift):
    x1, y1, x2, y2 = box
    return (
        x1 + horizontal_shift,
        y1 + vertical_shift,
        x2 + horizontal_shift,
        y2 + vertical_shift,
    )


def get_predicted_box(track):
    velocity_x, velocity_y = track["velocity"]
    elapsed_frames = track["missing_frames"] + 1
    return shift_box(
        track["xyxy"],
        velocity_x * elapsed_frames,
        velocity_y * elapsed_frames,
    )


def calculate_match_cost(
    detection,
    track,
    max_distance,
    reid_distance_threshold,
    minimum_iou,
):
    detection_center = get_box_center(detection["xyxy"])
    predicted_center = get_predicted_center(track)
    distance = calculate_distance(detection_center, predicted_center)
    predicted_box = get_predicted_box(track)
    iou = calculate_iou(detection["xyxy"], predicted_box)
    appearance_distance = calculate_appearance_distance(
        detection.get("appearance_embedding"),
        track.get("appearance_embedding"),
    )
    has_box_overlap = iou >= minimum_iou
    has_motion_match = distance <= max_distance
    has_spatial_match = has_box_overlap or has_motion_match
    has_appearance_match = (
        appearance_distance is not None
        and appearance_distance <= reid_distance_threshold
    )
    if not has_spatial_match and not has_appearance_match:
        return None
    normalized_distance = min(1, distance / max_distance)
    geometric_cost = (1 - iou) + normalized_distance
    if appearance_distance is None:
        return geometric_cost
    return geometric_cost + appearance_distance


def add_appearance_embedding(detection, frame, reidentifier):
    if detection.get("appearance_embedding") is not None:
        return
    if frame is None or reidentifier is None:
        return
    detection["appearance_embedding"] = reidentifier.get_embedding(
        frame,
        detection["xyxy"],
    )


class Tracker:
    def __init__(self):
        self.next_track_id = 1
        self.next_track_key = 1
        self.tracks = {}
        self.inactive_tracks = {}

    def update(
        self,
        detections,
        max_distance,
        max_missing_frames,
        reid_distance_threshold=0.35,
        frame=None,
        reidentifier=None,
        min_confirmed_frames=1,
        minimum_iou=0.05,
    ):
        if min_confirmed_frames < 1:
            raise ValueError("Minimum confirmed frames must be at least one.")
        if not 0 <= minimum_iou <= 1:
            raise ValueError("Minimum IoU must be between zero and one.")
        tracked_detections = []
        matched_track_keys = set()
        matched_detection_indexes = set()
        known_track_keys = list(self.tracks)
        match_costs = np.full(
            (len(known_track_keys), len(detections)),
            np.inf,
            dtype=np.float32,
        )
        for track_index, known_track_key in enumerate(known_track_keys):
            known_track = self.tracks[known_track_key]
            for detection_index, detection in enumerate(detections):
                match_cost = calculate_match_cost(
                    detection,
                    known_track,
                    max_distance,
                    reid_distance_threshold,
                    minimum_iou,
                )
                if match_cost is not None:
                    match_costs[track_index, detection_index] = match_cost

        maximum_match_cost = 2 + reid_distance_threshold
        matches, _, _ = linear_assignment(match_costs, maximum_match_cost)
        detection_track_keys = {}
        for track_index, detection_index in matches:
            track_key = known_track_keys[track_index]
            detection_track_keys[detection_index] = track_key
            matched_detection_indexes.add(detection_index)
            matched_track_keys.add(track_key)

        inactive_matches = []
        if self.inactive_tracks:
            for detection_index, detection in enumerate(detections):
                if detection_index in matched_detection_indexes:
                    continue
                add_appearance_embedding(detection, frame, reidentifier)
                for known_track_key, known_track in self.inactive_tracks.items():
                    appearance_distance = calculate_appearance_distance(
                        detection.get("appearance_embedding"),
                        known_track.get("appearance_embedding"),
                    )
                    if (
                        appearance_distance is not None
                        and appearance_distance <= reid_distance_threshold
                    ):
                        inactive_matches.append(
                            (appearance_distance, detection_index, known_track_key)
                        )
        inactive_matches.sort()
        for _, detection_index, track_key in inactive_matches:
            if detection_index in matched_detection_indexes:
                continue
            if track_key in matched_track_keys:
                continue
            detection_track_keys[detection_index] = track_key
            matched_detection_indexes.add(detection_index)
            matched_track_keys.add(track_key)

        for detection_index, detection in enumerate(detections):
            center = get_box_center(detection["xyxy"])
            if detection_index in detection_track_keys:
                track_key = detection_track_keys[detection_index]
                if track_key in self.inactive_tracks:
                    known_track = self.inactive_tracks.pop(track_key)
                else:
                    known_track = self.tracks[track_key]
                elapsed_frames = known_track["missing_frames"] + 1
                previous_center_x, previous_center_y = known_track["center"]
                velocity = (
                    (center[0] - previous_center_x) / elapsed_frames,
                    (center[1] - previous_center_y) / elapsed_frames,
                )
                appearance_embedding = known_track.get("appearance_embedding")
                consecutive_detections = known_track["consecutive_detections"] + 1
                track_id = known_track["track_id"]
            else:
                track_key = self.next_track_key
                self.next_track_key += 1
                velocity = (0, 0)
                appearance_embedding = None
                consecutive_detections = 1
                track_id = None
                matched_track_keys.add(track_key)
            if track_id is None and consecutive_detections >= min_confirmed_frames:
                track_id = self.next_track_id
                self.next_track_id += 1
                add_appearance_embedding(detection, frame, reidentifier)
                appearance_embedding = detection.get("appearance_embedding")
            self.tracks[track_key] = {
                "center": center,
                "velocity": velocity,
                "xyxy": detection["xyxy"],
                "missing_frames": 0,
                "appearance_embedding": appearance_embedding,
                "consecutive_detections": consecutive_detections,
                "track_id": track_id,
            }
            if track_id is not None:
                tracked_detection = detection.copy()
                tracked_detection["track_id"] = track_id
                tracked_detection["is_estimated"] = False
                tracked_detections.append(tracked_detection)
        expired_track_keys = []
        for known_track_key, known_track in self.tracks.items():
            if known_track_key not in matched_track_keys:
                known_track["missing_frames"] += 1
                known_track["consecutive_detections"] = 0
                if known_track["missing_frames"] <= max_missing_frames:
                    velocity_x, velocity_y = known_track["velocity"]
                    estimated_box = shift_box(
                        known_track["xyxy"],
                        velocity_x * known_track["missing_frames"],
                        velocity_y * known_track["missing_frames"],
                    )
                    if known_track["track_id"] is not None:
                        estimated_detection = {
                            "xyxy": estimated_box,
                            "track_id": known_track["track_id"],
                            "is_estimated": True,
                        }
                        tracked_detections.append(estimated_detection)
                else:
                    expired_track_keys.append(known_track_key)
        for track_key in expired_track_keys:
            expired_track = self.tracks.pop(track_key)
            if expired_track["track_id"] is not None:
                self.inactive_tracks[track_key] = expired_track
        return tracked_detections
