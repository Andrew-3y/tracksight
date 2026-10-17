import argparse
import shutil
from pathlib import Path

from src.training_data import SUPPORTED_IMAGE_SUFFIXES, validate_yolo_label_file


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build a validated one-class YOLO vehicle dataset from labelled sources."
    )
    parser.add_argument(
        "--train-sources",
        nargs="+",
        required=True,
        help="Names of fully labelled sources used to train the model.",
    )
    parser.add_argument(
        "--validation-sources",
        nargs="+",
        required=True,
        help="Names of fully labelled sources used to tune the model.",
    )
    parser.add_argument(
        "--test-sources",
        nargs="+",
        required=True,
        help="Names of fully labelled sources held back for final evaluation.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Empty directory where the YOLO dataset will be built.",
    )
    return parser.parse_args()


def get_image_paths(images_directory):
    return sorted(
        path
        for path in images_directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def validate_label_file(label_path):
    """Keep the script's public helper while using the shared validation rule."""
    return validate_yolo_label_file(label_path)


def validate_source_names(train_sources, validation_sources, test_sources):
    partitions = {
        "train": set(train_sources),
        "validation": set(validation_sources),
        "test": set(test_sources),
    }
    for first_name, first_sources in partitions.items():
        for second_name, second_sources in partitions.items():
            if first_name >= second_name:
                continue
            duplicate_sources = first_sources & second_sources
            if duplicate_sources:
                raise ValueError(
                    "A source cannot appear in more than one split: "
                    f"{sorted(duplicate_sources)}"
                )


def copy_sources(source_names, split_name, sources_directory, output_directory):
    split_images_directory = output_directory / "images" / split_name
    split_labels_directory = output_directory / "labels" / split_name
    split_images_directory.mkdir(parents=True, exist_ok=True)
    split_labels_directory.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_boxes = 0
    for source_name in source_names:
        source_directory = sources_directory / source_name
        images_directory = source_directory / "images"
        labels_directory = source_directory / "labels"
        if not images_directory.is_dir():
            raise FileNotFoundError(f"Missing source images directory: {images_directory}")
        if not labels_directory.is_dir():
            raise FileNotFoundError(f"Missing source labels directory: {labels_directory}")

        image_paths = get_image_paths(images_directory)
        if not image_paths:
            raise FileNotFoundError(f"No images found in source: {source_name}")

        for image_path in image_paths:
            label_path = labels_directory / f"{image_path.stem}.txt"
            validate_label_file(label_path)
            destination_stem = f"{source_name}__{image_path.stem}"
            destination_image_path = (
                split_images_directory / f"{destination_stem}{image_path.suffix.lower()}"
            )
            destination_label_path = split_labels_directory / f"{destination_stem}.txt"
            shutil.copy2(image_path, destination_image_path)
            shutil.copy2(label_path, destination_label_path)
            copied_images += 1
            copied_boxes += len(label_path.read_text(encoding="utf-8").splitlines())

    return copied_images, copied_boxes


def write_dataset_yaml(output_directory):
    yaml_contents = "\n".join(
        [
            # The config lives in the dataset root, so "." keeps every
            # generated dataset self-contained.  A path derived from the
            # current working directory breaks consumers that resolve it
            # relative to dataset.yaml.
            "path: .",
            "train: images/train",
            "val: images/validation",
            "test: images/test",
            "names:",
            "  0: vehicle",
            "",
        ]
    )
    dataset_yaml_path = output_directory / "dataset.yaml"
    dataset_yaml_path.write_text(yaml_contents, encoding="utf-8")
    return dataset_yaml_path


def build_vehicle_dataset(
    sources_directory,
    output_directory,
    train_sources,
    validation_sources,
    test_sources,
):
    validate_source_names(train_sources, validation_sources, test_sources)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"Output directory must be empty so nothing is overwritten: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    split_counts = {}
    for split_name, source_names in (
        ("train", train_sources),
        ("validation", validation_sources),
        ("test", test_sources),
    ):
        split_counts[split_name] = copy_sources(
            source_names,
            split_name,
            sources_directory,
            output_directory,
        )

    dataset_yaml_path = write_dataset_yaml(output_directory)
    return dataset_yaml_path, split_counts


def main():
    arguments = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data" / "training" / "race_vehicle"
    output_directory = arguments.output_directory or dataset_root / "yolo"
    dataset_yaml_path, split_counts = build_vehicle_dataset(
        sources_directory=dataset_root / "sources",
        output_directory=output_directory,
        train_sources=arguments.train_sources,
        validation_sources=arguments.validation_sources,
        test_sources=arguments.test_sources,
    )
    print(f"Built dataset configuration: {dataset_yaml_path}")
    for split_name, (image_count, box_count) in split_counts.items():
        print(f"{split_name}: {image_count} images, {box_count} vehicle boxes")


if __name__ == "__main__":
    main()
