import unittest

import cv2
import numpy as np

from src.camera_motion import CameraReferenceTransformer, build_background_mask


class CameraReferenceTransformerTests(unittest.TestCase):
    def test_background_mask_excludes_vehicle_box(self):
        mask = build_background_mask((100, 200), [(50, 20, 100, 60)])

        self.assertEqual(mask[0, 0], 255)
        self.assertEqual(mask[40, 75], 0)

    def test_maps_a_translated_current_frame_to_the_reference_frame(self):
        reference = np.zeros((180, 240, 3), dtype=np.uint8)
        for x_coordinate in range(20, 230, 30):
            for y_coordinate in range(20, 170, 30):
                cv2.circle(reference, (x_coordinate, y_coordinate), 3, (255, 255, 255), -1)
        current = cv2.warpAffine(
            reference,
            np.float32([[1, 0, 8], [0, 1, 5]]),
            (240, 180),
        )
        transformer = CameraReferenceTransformer(minimum_inliers=4)

        transformer.update(reference, [])
        current_to_reference = transformer.update(current, [])
        projected = cv2.perspectiveTransform(
            np.float32([[[108, 105]]]),
            current_to_reference,
        )[0][0]

        self.assertAlmostEqual(projected[0], 100, delta=1.0)
        self.assertAlmostEqual(projected[1], 100, delta=1.0)
        diagnostics = transformer.get_diagnostics()
        self.assertEqual(diagnostics["frame_outcomes"]["reference_frame"], 1)
        self.assertEqual(diagnostics["frame_outcomes"]["transform_updated"], 1)


if __name__ == "__main__":
    unittest.main()
