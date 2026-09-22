from pathlib import Path
import cv2

VIDEO_PATH = Path("data/raw/racecars_pexels_3774642_1920x1080_25fps.mp4")
OUTPUT_DIR = Path("outputs/frames")
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")
fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print("Frame Width:", frame_width, "pixels")
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("Frame Height:", frame_height, "pixels")
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("Frame Count:", frame_count, "frames")
duration_seconds = frame_count / fps
print("Duration:", duration_seconds , "seconds")

read_success, first_frame = cap.read()
if not read_success:
    raise RuntimeError("Could not read the first frame of the video.")
print("First frame shape:", first_frame.shape)
top_left_pixel = first_frame[0, 0]
print("Top-left pixel:", top_left_pixel)
first_frame_path = OUTPUT_DIR / "first_frame.png"
image_saved = cv2.imwrite(str(first_frame_path), first_frame)
if not image_saved:
    raise RuntimeError(f"Could not save the first frame to {first_frame_path}")