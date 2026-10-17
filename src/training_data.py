"""Validation for one-class YOLO datasets before TrackSight starts training."""

from pathlib import Path

from ultralytics.utils import YAML


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def calculate_box_iou(box_a, box_b):
    """Return the overlap score between two ``(x, y, width, height)`` boxes.

    The values are YOLO's normalized, centre-based coordinates.  This helper
    is used only for a *review warning*: two different cars can legitimately
    overlap in a camera image, so a high overlap never deletes or changes a
    label automatically.
    """
    center_x_a, center_y_a, width_a, height_a = box_a
    center_x_b, center_y_b, width_b, height_b = box_b
    left_a = center_x_a - width_a / 2
    top_a = center_y_a - height_a / 2
    right_a = center_x_a + width_a / 2
    bottom_a = center_y_a + height_a / 2
    left_b = center_x_b - width_b / 2
    top_b = center_y_b - height_b / 2
    right_b = center_x_b + width_b / 2
    bottom_b = center_y_b + height_b / 2

    intersection_left = max(left_a, left_b)
    intersection_top = max(top_a, top_b)
    intersection_right = min(right_a, right_b)
    intersection_bottom = min(bottom_a, bottom_b)
    intersection_width = max(0.0, intersection_right - intersection_left)
    intersection_height = max(0.0, intersection_bottom - intersection_top)
    intersection_area = intersection_width * intersection_height
    union_area = width_a * height_a + width_b * height_b - intersection_area
    return intersection_area / union_area if union_area else 0.0


def find_near_duplicate_boxes(boxes, iou_threshold=0.98):
    """Return index pairs whose boxes are almost identical.

    This flags likely accidental duplicate annotations for a human to inspect.
    It deliberately returns warnings instead of removing boxes because only a
    reviewer can determine whether two overlapping labels represent one car or
    two cars in an occlusion.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be greater than zero and at most one.")
    duplicate_pairs = []
    indexed_boxes = sorted(enumerate(boxes), key=lambda item: item[1][0])
    largest_width = max((box[2] for box in boxes), default=0.0)
    maximum_center_gap = (1 - iou_threshold) * largest_width
    for ordered_index, (first_index, first_box) in enumerate(indexed_boxes):
        for second_index, second_box in indexed_boxes[ordered_index + 1 :]:
            first_center_x, first_center_y, first_width, first_height = first_box
            second_center_x, second_center_y, second_width, second_height = second_box
            if second_center_x - first_center_x > maximum_center_gap:
                break
            # A pair cannot reach the requested IoU if its overlap on either
            # axis is already too small.  These inexpensive necessary checks
            # keep audits fast on dense frames without excluding any pair that
            # could satisfy the final IoU calculation.
            horizontal_overlap = (
                (first_width + second_width) / 2
                - abs(first_center_x - second_center_x)
            )
            vertical_overlap = (
                (first_height + second_height) / 2
                - abs(first_center_y - second_center_y)
            )
            if (
                horizontal_overlap < iou_threshold * max(first_width, second_width)
                or vertical_overlap < iou_threshold * max(first_height, second_height)
            ):
                continue
            iou = calculate_box_iou(first_box, second_box)
            if iou >= iou_threshold:
                duplicate_pairs.append(
                    {
                        "first_box_index": first_index,
                        "second_box_index": second_index,
                        "iou": iou,
                    }
                )
    return duplicate_pairs


def validate_yolo_box_values(values, label_description, required_class_id=None):
    """Validate one normalized YOLO box and return its parsed values.

    ``required_class_id`` keeps TrackSight's final dataset strictly one-class,
    while source import can validate an existing multi-class dataset before
    converting each source class to TrackSight's ``vehicle`` class.
    """
    if len(values) != 5:
        raise ValueError(f"Invalid YOLO label at {label_description}")
    try:
        class_id, center_x, center_y, box_width, box_height = map(float, values)
    except ValueError as error:
        raise ValueError(f"Non-numeric YOLO label at {label_description}") from error
    if not class_id.is_integer() or class_id < 0:
        raise ValueError(f"Invalid YOLO class ID at {label_description}")
    class_id = int(class_id)
    if required_class_id is not None and class_id != required_class_id:
        raise ValueError(
            f"Only class {required_class_id} ('vehicle') is allowed at "
            f"{label_description}"
        )
    if not all(0 <= value <= 1 for value in (center_x, center_y, box_width, box_height)):
        raise ValueError(f"Out-of-range YOLO box at {label_description}")
    if box_width == 0 or box_height == 0:
        raise ValueError(f"Zero-size YOLO box at {label_description}")
    if (
        center_x - box_width / 2 < 0
        or center_y - box_height / 2 < 0
        or center_x + box_width / 2 > 1
        or center_y + box_height / 2 > 1
    ):
        raise ValueError(f"YOLO box extends outside its image at {label_description}")
    return class_id, center_x, center_y, box_width, box_height


def validate_yolo_label_values(values, label_description):
    """Validate one normalized TrackSight ``vehicle`` box."""
    validate_yolo_box_values(values, label_description, required_class_id=0)


def find_label_path(image_path):
    """Map ``.../images/.../frame.jpg`` to the matching YOLO label file."""
    path_parts = list(image_path.parts)
    try:
        images_index = [part.lower() for part in path_parts].index("images")
    except ValueError as error:
        raise ValueError(
            f"Image path must be inside an 'images' directory: {image_path}"
        ) from error
    return Path(
        *path_parts[:images_index], "labels", *path_parts[images_index + 1 :]
    ).with_suffix(".txt")


def validate_yolo_label_file(label_path):
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label file: {label_path}")
    label_lines = label_path.read_text(encoding="utf-8").splitlines()
    for line_number, label_line in enumerate(label_lines, start=1):
        validate_yolo_label_values(
            label_line.split(),
            f"{label_path}:{line_number}",
        )
    return len(label_lines)


def get_dataset_root(dataset_config, dataset_config_path):
    configured_root = Path(dataset_config.get("path", "."))
    if configured_root.is_absolute():
        return configured_root
    return (dataset_config_path.parent / configured_root).resolve()


def validate_yolo_training_dataset(dataset_config_path):
    """Validate a folder-based, one-class TrackSight YOLO dataset.

    The returned counts are recorded only for reporting; no dataset files are
    altered by this check.
    """
    dataset_config_path = Path(dataset_config_path)
    if not dataset_config_path.is_file():
        raise FileNotFoundError(
            f"Could not find dataset configuration: {dataset_config_path}"
        )
    dataset_config = YAML.load(dataset_config_path)
    class_names = dataset_config.get("names")
    if isinstance(class_names, dict):
        class_names = list(class_names.values())
    if class_names != ["vehicle"]:
        raise ValueError(
            "TrackSight training data must define exactly one class named "
            "'vehicle'. Import sources through TrackSight so their labels are "
            "converted consistently."
        )

    dataset_root = get_dataset_root(dataset_config, dataset_config_path)
    split_counts = {}
    for split_name in ("train", "val", "test"):
        split_path = dataset_config.get(split_name)
        if split_path is None:
            if split_name == "test":
                continue
            raise ValueError(f"Dataset configuration is missing '{split_name}'.")
        if not isinstance(split_path, str):
            raise ValueError(
                f"TrackSight expects a folder path for '{split_name}', not a list."
            )
        images_directory = Path(split_path)
        if not images_directory.is_absolute():
            images_directory = dataset_root / images_directory
        if not images_directory.is_dir():
            raise FileNotFoundError(
                f"Missing {split_name} images directory: {images_directory}"
            )
        image_paths = sorted(
            image_path
            for image_path in images_directory.iterdir()
            if image_path.is_file()
            and image_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        if not image_paths:
            raise ValueError(f"No images found in {split_name}: {images_directory}")
        box_count = sum(
            validate_yolo_label_file(find_label_path(image_path))
            for image_path in image_paths
        )
        split_counts[split_name] = {
            "images": len(image_paths),
            "vehicle_boxes": box_count,
        }
    return split_counts
