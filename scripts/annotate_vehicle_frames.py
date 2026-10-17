import argparse
from pathlib import Path

import cv2


WINDOW_NAME = "TrackSight vehicle annotation"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Draw one box around every visible vehicle in collected frames."
    )
    parser.add_argument(
        "images_directory",
        type=Path,
        help="A source images folder created by collect_training_frames.",
    )
    return parser.parse_args()


def load_yolo_boxes(label_path, image_width, image_height):
    if not label_path.is_file():
        return []

    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Invalid YOLO label line in: {label_path}")
        class_id, center_x, center_y, box_width, box_height = map(float, values)
        if int(class_id) != 0:
            raise ValueError(f"Only class 0 ('vehicle') is allowed in: {label_path}")
        left = round((center_x - box_width / 2) * image_width)
        top = round((center_y - box_height / 2) * image_height)
        right = round((center_x + box_width / 2) * image_width)
        bottom = round((center_y + box_height / 2) * image_height)
        boxes.append((left, top, right, bottom))
    return boxes


def save_yolo_boxes(label_path, boxes, image_width, image_height):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for left, top, right, bottom in boxes:
        center_x = ((left + right) / 2) / image_width
        center_y = ((top + bottom) / 2) / image_height
        box_width = (right - left) / image_width
        box_height = (bottom - top) / image_height
        lines.append(f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


class VehicleBoxAnnotator:
    def __init__(self, image, boxes):
        self.image = image
        self.boxes = boxes
        self.start_point = None
        self.current_point = None

    def handle_mouse_event(self, event, x, y, _flags, _parameters):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.current_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.start_point is not None:
            self.current_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start_point is not None:
            left = min(self.start_point[0], x)
            top = min(self.start_point[1], y)
            right = max(self.start_point[0], x)
            bottom = max(self.start_point[1], y)
            if right - left >= 4 and bottom - top >= 4:
                self.boxes.append((left, top, right, bottom))
            self.start_point = None
            self.current_point = None

    def render(self):
        rendered_image = self.image.copy()
        for left, top, right, bottom in self.boxes:
            cv2.rectangle(rendered_image, (left, top), (right, bottom), (0, 255, 0), 2)
        if self.start_point is not None and self.current_point is not None:
            cv2.rectangle(rendered_image, self.start_point, self.current_point, (255, 255, 0), 2)
        instructions = "drag: box | u: undo | s: save + next | n: next | q: quit"
        cv2.putText(
            rendered_image,
            instructions,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            rendered_image,
            instructions,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
        )
        return rendered_image


def annotate_images(images_directory):
    image_paths = sorted(
        path for path in images_directory.iterdir() if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {images_directory}")

    labels_directory = images_directory.parent / "labels"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        image_height, image_width = image.shape[:2]
        label_path = labels_directory / f"{image_path.stem}.txt"
        annotator = VehicleBoxAnnotator(
            image,
            load_yolo_boxes(label_path, image_width, image_height),
        )
        cv2.setMouseCallback(WINDOW_NAME, annotator.handle_mouse_event)

        while True:
            cv2.imshow(WINDOW_NAME, annotator.render())
            key = cv2.waitKey(20) & 0xFF
            if key == ord("u") and annotator.boxes:
                annotator.boxes.pop()
            elif key == ord("s"):
                save_yolo_boxes(label_path, annotator.boxes, image_width, image_height)
                break
            elif key == ord("n"):
                break
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


def main():
    arguments = parse_arguments()
    annotate_images(arguments.images_directory)


if __name__ == "__main__":
    main()
