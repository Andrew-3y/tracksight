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
max_distance = 75
max_missing_frames = 1
duplicate_iou_threshold = 0.3
crop_edge_margin = 5
overlap_fraction_threshold = 0.25
completed_video_path, trajectories = process_video(
    model=model,
    video_path=video_path,
    output_video_path=output_video_path,
    track_regions=track_regions,
    max_distance=max_distance,
    max_missing_frames=max_missing_frames,
    duplicate_iou_threshold=duplicate_iou_threshold,
    crop_edge_margin=crop_edge_margin,
    overlap_fraction_threshold=overlap_fraction_threshold
)
print(f"Saved annotated video to: {completed_video_path}")
print(f"Recorded {len(trajectories)} trajectory segments.")
