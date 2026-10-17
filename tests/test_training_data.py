import unittest

from src.training_data import (
    calculate_box_iou,
    find_near_duplicate_boxes,
    validate_yolo_box_values,
    validate_yolo_label_values,
)
from src.detect_vehicles import get_vehicle_class_ids


class TrainingDataTests(unittest.TestCase):
    def test_accepts_a_box_inside_the_image(self):
        validate_yolo_label_values(
            ["0", "0.5", "0.5", "0.4", "0.2"],
            "example:1",
        )

    def test_rejects_a_box_that_extends_past_an_image_edge(self):
        with self.assertRaisesRegex(ValueError, "outside its image"):
            validate_yolo_label_values(
                ["0", "0.9", "0.5", "0.4", "0.2"],
                "example:1",
            )

    def test_rejects_a_non_vehicle_class(self):
        with self.assertRaisesRegex(ValueError, "Only class 0"):
            validate_yolo_label_values(
                ["1", "0.5", "0.5", "0.4", "0.2"],
                "example:1",
            )

    def test_source_validation_accepts_a_nonnegative_source_class(self):
        class_id, *_ = validate_yolo_box_values(
            ["4", "0.5", "0.5", "0.4", "0.2"],
            "example:1",
        )

        self.assertEqual(class_id, 4)

    def test_rejects_a_fractional_class_id(self):
        with self.assertRaisesRegex(ValueError, "class ID"):
            validate_yolo_box_values(
                ["1.5", "0.5", "0.5", "0.4", "0.2"],
                "example:1",
            )

    def test_finds_vehicle_classes_without_case_sensitivity(self):
        class Model:
            names = {0: "Race Car", 1: "VEHICLE", 2: "person"}

        self.assertEqual(get_vehicle_class_ids(Model()), [0, 1])

    def test_calculates_perfect_box_overlap(self):
        self.assertAlmostEqual(
            calculate_box_iou((0.5, 0.5, 0.4, 0.2), (0.5, 0.5, 0.4, 0.2)),
            1.0,
        )

    def test_warns_about_nearly_identical_boxes_without_removing_them(self):
        warnings = find_near_duplicate_boxes(
            [(0.5, 0.5, 0.4, 0.2), (0.501, 0.5, 0.4, 0.2)],
            iou_threshold=0.98,
        )

        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["first_box_index"], 0)
        self.assertEqual(warnings[0]["second_box_index"], 1)


if __name__ == "__main__":
    unittest.main()
