from math import sqrt


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

class Tracker:
    def __init__(self):
        self.next_track_id = 1
        self.tracks = {}
    def update(self, detections, max_distance, max_missing_frames):
        tracked_detections = []
        matched_track_ids = set()
        for detection in detections:
            center = get_box_center(detection["xyxy"])
            closest_track_id = None
            closest_distance = float("inf")
            for known_track_id, known_track in self.tracks.items():
                if known_track_id in matched_track_ids:
                    continue
                distance = calculate_distance(center, known_track["center"])
                if distance < closest_distance:
                    closest_track_id = known_track_id
                    closest_distance = distance
            if closest_track_id is not None and closest_distance <= max_distance:
                track_id = closest_track_id
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
            matched_track_ids.add(track_id)
            self.tracks[track_id] = {"center": center, "xyxy": detection["xyxy"], "missing_frames": 0}
            tracked_detection = detection.copy()
            tracked_detection["track_id"] = track_id
            tracked_detection["is_estimated"] = False
            tracked_detections.append(tracked_detection)
        for known_track_id, known_track in self.tracks.items():
            if known_track_id not in matched_track_ids:
                known_track["missing_frames"] += 1
                if known_track["missing_frames"] <= max_missing_frames:
                    estimated_detection = {
                        "xyxy": known_track["xyxy"],
                        "track_id": known_track_id,
                        "is_estimated": True
                    }
                    tracked_detections.append(estimated_detection)
        return tracked_detections
