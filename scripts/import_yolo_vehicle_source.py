import argparse
import json
import shutil
from pathlib import Path

from src.training_data import SUPPORTED_IMAGE_SUFFIXES, validate_yolo_box_values


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Import a labelled YOLO dataset as one TrackSight vehicle source."
    )
    parser.add_argument(
        "source_directory",
        type=Path,
        help="Root directory of a downloaded YOLO detection dataset.",
    )
    parser.add_argument(
        "--source-name",
        required=True,
        help="Stable name for this source inside TrackSight's training-data folder.",
    )
    parser.add_argument(
        "--license",
        required=True,
        help="Reuse licence for the imported source, recorded in its provenance file.",
    )
    parser.add_argument(
        "--source-url",
        required=True,
        help="Public page or repository URL for the imported source.",
    )
    parser.add_argument(
        "--vehicle-class-ids",
        nargs="+",
        required=True,
        type=int,
        help=(
            "Source class IDs that genuinely mean a road or race vehicle. "
            "Only these are converted to TrackSight's class 0: vehicle."
        ),
    )
    parser.add_argument(
        "--preserve-provided-splits",
        action="store_true",
        help=(
            "Import standard train, valid, and test folders as separate TrackSight "
            "sources, preserving the dataset publisher's split."
        ),
    )
    return parser.parse_args()


def find_image_paths(source_directory):
    return sorted(
        path
        for path in source_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def find_matching_label_path(image_path):
    path_parts = list(image_path.parts)
    for index, part in enumerate(path_parts):
        if part.lower() == "images":
            label_path = Path(*path_parts[:index], "labels", *path_parts[index + 1 :])
            return label_path.with_suffix(".txt")
    return image_path.with_suffix(".txt")


def convert_label_lines(label_path, vehicle_class_ids):
    """Convert explicitly approved source vehicle classes to TrackSight class 0.

    A source class may be a person, barrier, logo, or other object.  Treating
    all source classes as vehicles would corrupt the final detector, so callers
    must name the source IDs that really represent vehicles.
    """
    vehicle_class_ids = set(vehicle_class_ids)
    if not vehicle_class_ids or any(class_id < 0 for class_id in vehicle_class_ids):
        raise ValueError("At least one non-negative vehicle source class ID is required.")
    converted_lines = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        values = line.split()
        source_class_id, *_ = validate_yolo_box_values(
            values,
            f"{label_path}:{line_number}",
        )
        if source_class_id in vehicle_class_ids:
            converted_lines.append("0 " + " ".join(values[1:]))
    return converted_lines


def import_yolo_vehicle_source(
    source_directory,
    destination_source_directory,
    source_license,
    source_url,
    vehicle_class_ids,
):
    if not source_directory.is_dir():
        raise FileNotFoundError(f"Could not find downloaded source: {source_directory}")
    if destination_source_directory.exists():
        raise FileExistsError(
            "Destination source already exists, so no files were overwritten: "
            f"{destination_source_directory}"
        )

    image_paths = find_image_paths(source_directory)
    if not image_paths:
        raise FileNotFoundError(f"No images found in downloaded source: {source_directory}")

    prepared_images = []
    for image_path in image_paths:
        label_path = find_matching_label_path(image_path)
        if not label_path.is_file():
            raise FileNotFoundError(
                "Every imported image must have labels; missing label file: " f"{label_path}"
            )
        prepared_images.append(
            (image_path, convert_label_lines(label_path, vehicle_class_ids))
        )

    images_directory = destination_source_directory / "images"
    labels_directory = destination_source_directory / "labels"
    images_directory.mkdir(parents=True)
    labels_directory.mkdir(parents=True)

    imported_images = 0
    imported_boxes = 0
    for image_index, (image_path, converted_lines) in enumerate(prepared_images):
        destination_stem = f"{image_index:06d}_{image_path.stem}"
        destination_image_path = images_directory / (
            destination_stem + image_path.suffix.lower()
        )
        destination_label_path = labels_directory / f"{destination_stem}.txt"
        shutil.copy2(image_path, destination_image_path)
        destination_label_path.write_text(
            "\n".join(converted_lines) + ("\n" if converted_lines else ""),
            encoding="utf-8",
        )
        imported_images += 1
        imported_boxes += len(converted_lines)

    metadata = {
        "source_directory": str(source_directory),
        "source_url": source_url,
        "license": source_license,
        "source_label_policy": (
            "Only explicitly approved source vehicle classes are converted "
            "to class 0: vehicle."
        ),
        "vehicle_source_class_ids": sorted(set(vehicle_class_ids)),
        "imported_images": imported_images,
        "imported_vehicle_boxes": imported_boxes,
    }
    metadata_path = destination_source_directory / "source.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return imported_images, imported_boxes, metadata_path


def get_provided_split_directories(source_directory):
    split_directory_names = {
        "train": "train",
        "validation": "valid",
        "test": "test",
    }
    split_directories = {}
    for split_name, directory_name in split_directory_names.items():
        split_directory = source_directory / directory_name
        if not split_directory.is_dir():
            raise FileNotFoundError(
                "Expected a standard YOLO split directory but could not find: "
                f"{split_directory}"
            )
        split_directories[split_name] = split_directory
    return split_directories


def main():
    arguments = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]
    sources_directory = (
        project_root / "data" / "training" / "race_vehicle" / "sources"
    )

    if arguments.preserve_provided_splits:
        split_directories = get_provided_split_directories(arguments.source_directory)
        imported_source_names = {}
        for split_name, split_directory in split_directories.items():
            imported_source_name = f"{arguments.source_name}_{split_name}"
            destination_source_directory = sources_directory / imported_source_name
            imported_images, imported_boxes, metadata_path = import_yolo_vehicle_source(
                source_directory=split_directory,
                destination_source_directory=destination_source_directory,
                source_license=arguments.license,
                source_url=arguments.source_url,
                vehicle_class_ids=arguments.vehicle_class_ids,
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["publisher_provided_split"] = split_name
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            imported_source_names[split_name] = imported_source_name
            print(
                f"{split_name}: imported {imported_images} images and "
                f"{imported_boxes} vehicle boxes."
            )

        print("Use these source names when building the YOLO dataset:")
        for split_name, imported_source_name in imported_source_names.items():
            print(f"{split_name}: {imported_source_name}")
        return

    destination_source_directory = sources_directory / arguments.source_name
    imported_images, imported_boxes, metadata_path = import_yolo_vehicle_source(
        source_directory=arguments.source_directory,
        destination_source_directory=destination_source_directory,
        source_license=arguments.license,
        source_url=arguments.source_url,
        vehicle_class_ids=arguments.vehicle_class_ids,
    )
    print(f"Imported {imported_images} images and {imported_boxes} vehicle boxes.")
    print(f"Recorded source provenance in: {metadata_path}")


if __name__ == "__main__":
    main()
