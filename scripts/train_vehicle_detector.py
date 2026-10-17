import argparse
from pathlib import Path

from ultralytics import YOLO

from src.training_data import validate_yolo_training_dataset


def parse_arguments():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train TrackSight's one-class vehicle detector."
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=project_root / "data" / "training" / "race_vehicle" / "yolo" / "dataset.yaml",
        help="Generated YOLO dataset configuration file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=project_root / "models" / "yolo26m.pt",
        help="Pretrained detection model to fine-tune.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of complete passes through the training images.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1280,
        help="Square training resolution in pixels.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training images processed together; 1 is safe on modest-memory GPUs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel data-loading worker processes; 0 is the most reliable setting on Windows.",
    )
    parser.add_argument(
        "--run-name",
        default="race_vehicle",
        help="Name of the separate directory created inside outputs/training.",
    )
    return parser.parse_args()


def train_vehicle_detector(
    dataset_config_path,
    model_path,
    epochs,
    image_size,
    batch_size,
    workers,
    run_name,
):
    if not dataset_config_path.is_file():
        raise FileNotFoundError(f"Could not find dataset configuration: {dataset_config_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Could not find pretrained model: {model_path}")
    if epochs < 1:
        raise ValueError("Epochs must be at least one.")
    if image_size < 32:
        raise ValueError("Image size must be at least 32 pixels.")
    if workers < 0:
        raise ValueError("Workers cannot be negative.")
    if not run_name:
        raise ValueError("Run name cannot be empty.")

    split_counts = validate_yolo_training_dataset(dataset_config_path)
    for split_name, counts in split_counts.items():
        print(
            f"Validated {split_name}: {counts['images']} images, "
            f"{counts['vehicle_boxes']} vehicle boxes"
        )

    project_root = Path(__file__).resolve().parents[1]
    training_directory = project_root / "outputs" / "training"
    model = YOLO(model_path)
    model.train(
        data=dataset_config_path,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        workers=workers,
        project=training_directory,
        name=run_name,
        exist_ok=False,
    )
    best_model_path = Path(model.trainer.best)
    if not best_model_path.is_file():
        raise RuntimeError(f"Training completed without a best model at: {best_model_path}")
    return best_model_path


def main():
    arguments = parse_arguments()
    best_model_path = train_vehicle_detector(
        dataset_config_path=arguments.dataset_config,
        model_path=arguments.model_path,
        epochs=arguments.epochs,
        image_size=arguments.image_size,
        batch_size=arguments.batch_size,
        workers=arguments.workers,
        run_name=arguments.run_name,
    )
    print(f"Saved trained TrackSight vehicle model to: {best_model_path}")


if __name__ == "__main__":
    main()
