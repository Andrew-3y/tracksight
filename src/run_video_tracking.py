from pathlib import Path
from ultralytics import YOLO
from src.process_video import process_video

project_root = Path(__file__).resolve().parents[1]
video_path = project_root / "data" / "raw" / "racecars_pexels_3774642_1920x1080_25fps.mp4"  
model_path = project_root / "models" / "yolo26m.pt"
model = YOLO(model_path)
track_regions = {
    "centre": (650, 400, 1300, 800),
    "right": (1150, 400, 1920, 850),
}
output_video_path = project_root / "outputs" / "tracking" / "annotated_racecars.mp4"
max_distance = 50
max_missing_frames = 1
completed_video_path = process_video(
    model=model,
    video_path=video_path,
    output_video_path=output_video_path,
    track_regions=track_regions,
    max_distance=max_distance,
    max_missing_frames=max_missing_frames,
)
print(f"Saved annotated video to: {completed_video_path}")