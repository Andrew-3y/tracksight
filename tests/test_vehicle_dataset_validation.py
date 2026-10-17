import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.build_vehicle_dataset import validate_label_file, write_dataset_yaml
from scripts.import_yolo_vehicle_source import convert_label_lines


class InMemoryLabelPath:
    def __init__(self, content):
        self.content = content

    def is_file(self):
        return True

    def read_text(self, encoding):
        return self.content

    def __str__(self):
        return "in-memory-label.txt"


class VehicleDatasetValidationTests(unittest.TestCase):
    def test_dataset_config_uses_its_own_directory_as_the_root(self):
        with TemporaryDirectory() as temporary_directory:
            dataset_yaml_path = write_dataset_yaml(Path(temporary_directory))

            self.assertIn(
                "path: .",
                dataset_yaml_path.read_text(encoding="utf-8"),
            )

    def test_rejects_a_box_that_extends_past_the_right_edge(self):
        label_path = InMemoryLabelPath("0 0.9 0.5 0.4 0.2\n")

        with self.assertRaisesRegex(ValueError, "outside its image"):
            validate_label_file(label_path)

    def test_import_rejects_a_box_that_extends_past_the_bottom_edge(self):
        label_path = InMemoryLabelPath("4 0.5 0.9 0.2 0.4\n")

        with self.assertRaisesRegex(ValueError, "outside its image"):
            convert_label_lines(label_path, vehicle_class_ids=[4])

    def test_import_converts_only_approved_source_classes_to_vehicle(self):
        label_path = InMemoryLabelPath(
            "4 0.5 0.5 0.2 0.4\n"
            "7 0.2 0.3 0.1 0.1\n"
        )

        self.assertEqual(
            convert_label_lines(label_path, vehicle_class_ids=[4]),
            ["0 0.5 0.5 0.2 0.4"],
        )

    def test_import_requires_at_least_one_vehicle_class(self):
        label_path = InMemoryLabelPath("4 0.5 0.5 0.2 0.4\n")

        with self.assertRaisesRegex(ValueError, "At least one"):
            convert_label_lines(label_path, vehicle_class_ids=[])


if __name__ == "__main__":
    unittest.main()
