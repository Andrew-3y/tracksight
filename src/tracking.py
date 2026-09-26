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
    def update(self, detections, max_distance):
        tracked_detections = []
        for detection in detections:
            center = get_box_center(detection["xyxy"])
            closest_track_id = None
            closest_distance = float("inf")
            for known_track_id, known_track in self.tracks.items():
                distance = calculate_distance(center, known_track["center"])
                if distance < closest_distance:
                    closest_track_id = known_track_id
                    closest_distance = distance
            if closest_track_id is not None and closest_distance <= max_distance:
                track_id = closest_track_id
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
            self.tracks[track_id] = {"center": center}
            tracked_detection = detection.copy()
            tracked_detection["track_id"] = track_id
            tracked_detections.append(tracked_detection)
        return tracked_detections
