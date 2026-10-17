VEHICLE_CLASS_NAMES = {"car", "racecar", "motorcycle", "bus", "truck", "vehicle"}


def is_vehicle_class_name(class_name):
    """Accept common one- and two-word vehicle class names from YOLO models."""
    normalized_name = str(class_name).strip().lower().replace("_", " ").replace("-", " ")
    name_tokens = set(normalized_name.split())
    return bool(name_tokens & VEHICLE_CLASS_NAMES)


def get_vehicle_class_ids(model):
    model_class_names = model.names
    if isinstance(model_class_names, dict):
        class_items = model_class_names.items()
    else:
        class_items = enumerate(model_class_names)

    vehicle_class_ids = [
        class_id
        for class_id, class_name in class_items
        if is_vehicle_class_name(class_name)
    ]
    if not vehicle_class_ids:
        raise ValueError(
            "The model does not contain any supported vehicle classes: "
            f"{sorted(VEHICLE_CLASS_NAMES)}"
        )
    return vehicle_class_ids


def detect_vehicles(
    model,
    source_image,
    roi_left,
    roi_top,
    roi_right,
    roi_bottom,
    confidence_threshold=0.25,
):
    track_region = source_image[roi_top:roi_bottom, roi_left:roi_right]
    vehicle_class_ids = get_vehicle_class_ids(model)
    results = model(
        track_region,
        classes=vehicle_class_ids,
        imgsz=1280,
        conf=confidence_threshold,
        agnostic_nms=True,
        verbose=False,
    )
    first_result = results[0]
    annotated_image = first_result.plot()
    return first_result, annotated_image


def detect_full_frame_vehicles(model, frame, confidence_threshold):
    """Run one consistent vehicle detector pass over a complete video frame.

    Tracking must receive every detection together.  Splitting a crowded frame
    into independent crops can give the same physical car two unrelated boxes,
    which then produces duplicate IDs. A whole-frame pass lets YOLO perform
    its normal non-maximum suppression before BoT-SORT associates objects.
    """
    if not 0 < confidence_threshold <= 1:
        raise ValueError("Detection confidence must be above zero and at most one.")
    vehicle_class_ids = get_vehicle_class_ids(model)
    return model(
        frame,
        classes=vehicle_class_ids,
        imgsz=1280,
        conf=confidence_threshold,
        agnostic_nms=True,
        verbose=False,
    )[0]


def filter_boxes_to_regions(boxes, regions):
    """Keep boxes whose centre belongs to at least one configured race area."""
    kept_indexes = []
    for box_index, box in enumerate(boxes.xyxy.cpu().tolist()):
        x1, y1, x2, y2 = box
        box_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        if any(
            is_point_inside_box(box_center, region)
            for region in regions.values()
        ):
            kept_indexes.append(box_index)
    return boxes[kept_indexes]

def convert_boxes_to_full_frame(boxes, roi_left, roi_top):
    full_frame_detections = []
    for x1, y1, x2, y2, confidence, class_id in boxes.data.cpu().tolist():
        full_frame_detections.append(
            {
                "xyxy": (
                    x1 + roi_left,
                    y1 + roi_top,
                    x2 + roi_left,
                    y2 + roi_top,
                ),
                "confidence": confidence,
                "class_id": int(class_id),
            }
        )
    return full_frame_detections

def is_point_inside_box(point, box):
    point_x, point_y = point
    x1, y1, x2, y2 = box
    return x1 <= point_x <= x2 and y1 <= point_y <= y2 

def calculate_overlap_fraction(box_a, box_b):
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b
    intersection_left = max(x1_a, x1_b)
    intersection_right = min(x2_a, x2_b)
    intersection_top = max(y1_a, y1_b)
    intersection_bottom = min(y2_a, y2_b)
    intersection_width = max(0, intersection_right - intersection_left)
    intersection_height = max(0, intersection_bottom - intersection_top)
    intersection_area = intersection_width * intersection_height
    box_a_area = (x2_a - x1_a) * (y2_a - y1_a)
    if box_a_area == 0:
        return 0
    return intersection_area / box_a_area

def is_box_touching_shared_crop_edge(box, current_region, all_regions, edge_margin):
    x1, y1, x2, y2 = box
    region_left, region_top, region_right, region_bottom = current_region
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2
    edges = [
        (x1 <= region_left + edge_margin, "vertical", region_left),
        (x2 >= region_right - edge_margin, "vertical", region_right),
        (y1 <= region_top + edge_margin, "horizontal", region_top),
        (y2 >= region_bottom - edge_margin, "horizontal", region_bottom),
    ]
    for touches_edge, edge_direction, edge_position in edges:
        if not touches_edge:
            continue
        for other_region in all_regions.values():
            if other_region == current_region:
                continue
            other_left, other_top, other_right, other_bottom = other_region
            if edge_direction == "vertical":
                covers_edge = other_left < edge_position < other_right
                covers_box_center = other_top <= box_center_y <= other_bottom
            else:
                covers_edge = other_top < edge_position < other_bottom
                covers_box_center = other_left <= box_center_x <= other_right
            if covers_edge and covers_box_center:
                return True
    return False

def remove_shared_crop_edge_detections(detections, all_regions, edge_margin, overlap_fraction_threshold):
    kept_detections = []
    for detection in detections:
        box = detection["xyxy"]
        region_name = detection["region_name"]
        current_region = all_regions[region_name]
        box_center = (
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2
        )
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        box_area = box_width * box_height
        touches_shared_edge = is_box_touching_shared_crop_edge(box, current_region, all_regions, edge_margin)
        has_larger_matching_detection = False
        if touches_shared_edge:
            for other_detection in detections:
                if other_detection is detection:
                    continue
                if other_detection["region_name"] == region_name:
                    continue
                other_box = other_detection["xyxy"]
                other_width = other_box[2] - other_box[0]
                other_height = other_box[3] - other_box[1]
                other_area = other_width * other_height
                overlap_fraction = calculate_overlap_fraction(box, other_box)
                matches_other_box = (
                    is_point_inside_box(box_center, other_box) or overlap_fraction >= overlap_fraction_threshold
                )
                if matches_other_box and other_area > box_area:
                    has_larger_matching_detection = True
                    break
        if not touches_shared_edge or not has_larger_matching_detection:
            kept_detections.append(detection)
    return kept_detections

def calculate_iou(box_a, box_b):
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b
    intersection_left = max(x1_a, x1_b)
    intersection_right = min(x2_a, x2_b)
    intersection_top = max(y1_a, y1_b)
    intersection_bottom = min(y2_a, y2_b)
    intersection_width = max(0, intersection_right - intersection_left)
    intersection_height = max(0, intersection_bottom - intersection_top)
    intersection_area = intersection_width * intersection_height
    box_a_area = (x2_a - x1_a) * (y2_a - y1_a)
    box_b_area = (x2_b - x1_b) * (y2_b - y1_b)
    union_area = box_a_area + box_b_area - intersection_area
    if union_area == 0:
        return 0
    iou = intersection_area / union_area
    return iou

def remove_duplicate_detections(
    detections,
    iou_threshold,
    same_region_iou_threshold=0.8,
):
    detections_by_confidence = sorted(
        detections,
        key=lambda detection: detection["confidence"],
        reverse=True,
    )
    unique_detections = []
    for candidate_detection in detections_by_confidence:
        candidate_box = candidate_detection["xyxy"]
        candidate_region_name = candidate_detection["region_name"]
        is_duplicate = False
        for kept_detection in unique_detections:
            kept_box = kept_detection["xyxy"]
            iou = calculate_iou(candidate_box, kept_box)
            same_region = kept_detection["region_name"] == candidate_region_name
            duplicate_iou_threshold = (
                same_region_iou_threshold if same_region else iou_threshold
            )
            if iou >= duplicate_iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_detections.append(candidate_detection)
    return unique_detections
