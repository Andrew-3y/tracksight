import unittest

from scripts.review_vehicle_labels import normalized_box_to_pixels


class ReviewVehicleLabelsTests(unittest.TestCase):
    def test_converts_a_normalized_box_to_image_pixels(self):
        pixels = normalized_box_to_pixels(
            (0.5, 0.5, 0.4, 0.2),
            image_width=1000,
            image_height=500,
        )

        self.assertEqual(pixels, (300, 200, 700, 300))


if __name__ == "__main__":
    unittest.main()
