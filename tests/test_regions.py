import unittest

from src.calibration import validate_calibration_points
from src.regions import scale_normalized_regions


class RegionAndCalibrationValidationTests(unittest.TestCase):
    def test_scales_a_valid_normalized_region(self):
        self.assertEqual(
            scale_normalized_regions({"track": [0.1, 0.2, 0.9, 0.8]}, 1000, 500),
            {"track": (100, 100, 900, 400)},
        )

    def test_rejects_non_numeric_region_coordinates(self):
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            scale_normalized_regions({"track": [0.1, "top", 0.9, 0.8]}, 1000, 500)

    def test_rejects_non_finite_calibration_points(self):
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            validate_calibration_points(
                [[0, 0], [1, 0], [1, 1], [float("nan"), 1]],
                [[0, 0], [1, 0], [1, 1], [0, 1]],
            )


if __name__ == "__main__":
    unittest.main()
