from src.tracking import Tracker
from src.trajectory import add_trajectory_points

tracker = Tracker()
max_distance = 50
max_missing_frames = 1
trajectories = {}
test_fps = 25

frame_one_detections = [
    {
        "xyxy": (100, 100, 140, 140),
        "confidence": 0.90,
        "class_id": 2,
    },
    {
        "xyxy": (300, 200, 340, 240),
        "confidence": 0.85,
        "class_id": 2,
    },
]

frame_two_detections = [
    {
        "xyxy": (112, 104, 152, 144),
        "confidence": 0.91,
        "class_id": 2,
    },
    {
        "xyxy": (290, 205, 330, 245),
        "confidence": 0.88,
        "class_id": 2,
    },
]

frame_three_detections = [
    {
        "xyxy": (120, 110, 160, 150),
        "confidence": 0.89,
        "class_id": 2,
    },
    {
        "xyxy": (282, 210, 322, 250),
        "confidence": 0.86,
        "class_id": 2,
    },
    {
        "xyxy": (600, 400, 640, 440),
        "confidence": 0.92,
        "class_id": 2,
    },
]

frame_four_detections = []

frame_five_detections = [
    {
        "xyxy": (136, 126, 176, 166),
        "confidence": 0.90,
        "class_id": 2,
    },
    {
        "xyxy": (266, 220, 306, 260),
        "confidence": 0.87,
        "class_id": 2,
    },
    {
        "xyxy": (600, 400, 640, 440),
        "confidence": 0.93,
        "class_id": 2,
    },
]

tracked_frame_one = tracker.update(frame_one_detections,max_distance, max_missing_frames)
trajectories = add_trajectory_points(trajectories, tracked_frame_one, frame_index=0, fps=test_fps)
tracked_frame_two = tracker.update(frame_two_detections, max_distance,max_missing_frames)
trajectories = add_trajectory_points(trajectories, tracked_frame_two, frame_index=1, fps=test_fps)
tracked_frame_three = tracker.update(frame_three_detections, max_distance, max_missing_frames)
trajectories = add_trajectory_points(trajectories, tracked_frame_three, frame_index=2, fps=test_fps)
tracked_frame_four = tracker.update(frame_four_detections, max_distance, max_missing_frames)
trajectories = add_trajectory_points(trajectories, tracked_frame_four, frame_index=3, fps=test_fps)
tracked_frame_five = tracker.update(frame_five_detections, max_distance, max_missing_frames)
trajectories = add_trajectory_points(trajectories, tracked_frame_five, frame_index=4, fps=test_fps)

print("Frame 1 track IDs:", [detection["track_id"] for detection in tracked_frame_one])
print("Frame 2 track IDs:", [detection["track_id"] for detection in tracked_frame_two])
print("Frame 3 track IDs:", [detection["track_id"] for detection in tracked_frame_three])
print("Frame 4 tracked detections:", tracked_frame_four)
print("Frame 5 track IDs:", [detection["track_id"] for detection in tracked_frame_five])
print("Trajectories:", trajectories)

non_overlapping_tracker = Tracker()
first_non_overlapping_detection = non_overlapping_tracker.update(
    [{"xyxy": (0, 0, 20, 20), "confidence": 0.9, "class_id": 2}],
    max_distance,
    max_missing_frames,
)
second_non_overlapping_detection = non_overlapping_tracker.update(
    [{"xyxy": (25, 0, 45, 20), "confidence": 0.9, "class_id": 2}],
    max_distance,
    max_missing_frames,
)
assert (
    first_non_overlapping_detection[0]["track_id"]
    == second_non_overlapping_detection[0]["track_id"]
)
print("Non-overlapping nearby box keeps track ID:", second_non_overlapping_detection[0]["track_id"])
