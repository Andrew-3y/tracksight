"""Map frames from a moving camera back to one calibration reference frame."""

import cv2
import numpy as np


def build_background_mask(frame_shape, vehicle_boxes, margin_pixels=8):
    """Return a feature mask that excludes detected, moving vehicles.

    Camera movement must be estimated from stationary scenery, not from cars.
    The white pixels in the result are safe places to find background features;
    every supplied vehicle box is expanded slightly and marked black.
    """
    if margin_pixels < 0:
        raise ValueError("Feature-mask margin cannot be negative.")
    frame_height, frame_width = frame_shape[:2]
    mask = np.full((frame_height, frame_width), 255, dtype=np.uint8)
    for left, top, right, bottom in vehicle_boxes:
        cv2.rectangle(
            mask,
            (max(0, round(left) - margin_pixels), max(0, round(top) - margin_pixels)),
            (
                min(frame_width - 1, round(right) + margin_pixels),
                min(frame_height - 1, round(bottom) + margin_pixels),
            ),
            0,
            thickness=-1,
        )
    return mask


class CameraReferenceTransformer:
    """Maintain a robust current-frame-to-reference homography.

    The first supplied frame defines the calibration reference.  Every later
    transform maps a point in the current camera image back into that first
    image, allowing one calibration profile to remain useful while a camera
    pans, tilts, or makes a modest zoom change.
    """

    def __init__(self, minimum_inliers=12):
        if minimum_inliers < 4:
            raise ValueError("At least four inliers are required for a homography.")
        self.minimum_inliers = minimum_inliers
        self.previous_gray_frame = None
        self.previous_background_mask = None
        self.current_to_reference = np.eye(3, dtype=np.float64)
        self.diagnostic_counts = {
            "reference_frame": 0,
            "transform_updated": 0,
            "retained_insufficient_features": 0,
            "retained_insufficient_inliers": 0,
            "retained_invalid_transform": 0,
        }
        self.last_status = None

    def get_diagnostics(self):
        """Return aggregate, non-sensitive stabilization outcomes for one run."""
        return {
            "method": "sparse_optical_flow_background_features",
            "minimum_inliers": self.minimum_inliers,
            "frame_outcomes": dict(self.diagnostic_counts),
        }

    def update(self, frame, vehicle_boxes):
        """Return the current image-to-reference transform for this frame.

        If there are not enough reliable background matches, the last reliable
        transform is retained.  This deliberately avoids turning a poor camera
        estimate into a fabricated change in vehicle trajectory.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background_mask = build_background_mask(gray_frame.shape, vehicle_boxes)
        if self.previous_gray_frame is None:
            self.previous_gray_frame = gray_frame
            self.previous_background_mask = background_mask
            self.last_status = "reference_frame"
            self.diagnostic_counts[self.last_status] += 1
            return self.current_to_reference.copy()

        previous_points = cv2.goodFeaturesToTrack(
            self.previous_gray_frame,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=8,
            mask=self.previous_background_mask,
        )
        if previous_points is None:
            self.last_status = "retained_insufficient_features"
        else:
            current_points, statuses, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_gray_frame,
                gray_frame,
                previous_points,
                None,
            )
            if current_points is None or statuses is None:
                self.last_status = "retained_insufficient_features"
            else:
                valid = statuses.reshape(-1).astype(bool)
                previous_points = previous_points[valid]
                current_points = current_points[valid]
                if len(previous_points) < self.minimum_inliers:
                    self.last_status = "retained_insufficient_features"
                else:
                    current_to_previous, inlier_mask = cv2.findHomography(
                        current_points,
                        previous_points,
                        cv2.RANSAC,
                        3.0,
                    )
                    inlier_count = (
                        0
                        if inlier_mask is None
                        else int(inlier_mask.reshape(-1).sum())
                    )
                    if (
                        current_to_previous is not None
                        and inlier_count >= self.minimum_inliers
                        and np.isfinite(current_to_previous).all()
                    ):
                        self.current_to_reference = (
                            self.current_to_reference @ current_to_previous
                        )
                        self.last_status = "transform_updated"
                    elif current_to_previous is None or not np.isfinite(current_to_previous).all():
                        self.last_status = "retained_invalid_transform"
                    else:
                        self.last_status = "retained_insufficient_inliers"

        self.diagnostic_counts[self.last_status] += 1

        self.previous_gray_frame = gray_frame
        self.previous_background_mask = background_mask
        return self.current_to_reference.copy()
