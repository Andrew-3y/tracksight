"""Production vehicle tracking built on Ultralytics' BoT-SORT implementation.

The application previously used a small custom nearest-neighbour tracker.  That
is useful for learning, but it is not robust enough for a dense racing pack:
several cars can be nearby, briefly overlap, or be partially hidden. BoT-SORT
uses a Kalman motion model plus global assignment. Its global-motion
compensation also estimates camera pan and zoom, so shared camera movement is
not mistaken for every car suddenly changing direction.
"""

from types import SimpleNamespace
from math import ceil

from ultralytics.trackers.bot_sort import BOTSORT


def sort_track_rows_for_display_ids(tracked_rows):
    """Order tracker rows by screen position before assigning new visible IDs."""
    return sorted(
        tracked_rows,
        key=lambda row: (float(row[1]), float(row[0])),
    )


class MotionCompensatedVehicleTracker:
    """Convert YOLO vehicle detections into stable, visible track IDs."""

    def __init__(
        self,
        fps,
        max_missing_seconds,
        high_confidence_threshold,
        low_confidence_threshold,
        new_track_confidence_threshold,
        match_threshold,
        minimum_confirmed_seconds=0.2,
        camera_motion_method="sparseOptFlow",
    ):
        if fps <= 0:
            raise ValueError("Video FPS must be positive.")
        if max_missing_seconds < 0:
            raise ValueError("Maximum missing seconds cannot be negative.")
        if not 0 < low_confidence_threshold <= high_confidence_threshold <= 1:
            raise ValueError(
                "Tracker confidence thresholds must satisfy "
                "0 < low confidence <= high confidence <= 1."
            )
        if not high_confidence_threshold <= new_track_confidence_threshold <= 1:
            raise ValueError(
                "New-track confidence must be at least the high-confidence "
                "threshold and at most 1."
            )
        if not 0 < match_threshold <= 1:
            raise ValueError("Tracker match threshold must be above zero and at most 1.")
        if minimum_confirmed_seconds <= 0:
            raise ValueError("Minimum confirmed seconds must be positive.")
        if camera_motion_method not in {"sparseOptFlow", "none"}:
            raise ValueError(
                "Camera motion method must be sparseOptFlow or none."
            )
        self.tracker = BOTSORT(
            SimpleNamespace(
                track_high_thresh=high_confidence_threshold,
                track_low_thresh=low_confidence_threshold,
                new_track_thresh=new_track_confidence_threshold,
                track_buffer=max(1, round(fps * max_missing_seconds)),
                match_thresh=match_threshold,
                fuse_score=True,
                gmc_method=camera_motion_method,
                proximity_thresh=0.5,
                appearance_thresh=0.8,
                with_reid=False,
                model="auto",
            )
        )
        # BoT-SORT assigns an internal ID as soon as it sees a plausible
        # detection.  A one-frame detection is not reliable enough to expose
        # to a person, save as a trajectory, or use as evidence. Keep the
        # internal ID private until it has been observed in consecutive frames.
        self.consecutive_observation_counts = {}
        self.last_seen_update_indexes = {}
        self.update_index = 0
        self.display_ids = {}
        self.next_display_id = 1
        self.minimum_confirmed_frames = max(
            1,
            ceil(fps * minimum_confirmed_seconds),
        )

    def update(self, boxes, frame):
        """Return real, currently observed vehicle detections.

        ``boxes`` is the filtered set of YOLO boxes for this frame. BoT-SORT
        returns only detections it can associate with an active track; it does
        not invent an estimated box when the detector has no observation.
        """
        self.update_index += 1
        tracked_rows = sort_track_rows_for_display_ids(
            self.tracker.update(boxes.cpu().numpy(), frame)
        )
        tracked_detections = []
        for x1, y1, x2, y2, native_track_id, confidence, class_id, _ in tracked_rows:
            native_track_id = int(native_track_id)
            previous_update_index = self.last_seen_update_indexes.get(native_track_id)
            if previous_update_index == self.update_index - 1:
                self.consecutive_observation_counts[native_track_id] = (
                    self.consecutive_observation_counts.get(native_track_id, 0) + 1
                )
            else:
                self.consecutive_observation_counts[native_track_id] = 1
            self.last_seen_update_indexes[native_track_id] = self.update_index
            is_confirmed = native_track_id in self.display_ids
            if (
                not is_confirmed
                and self.consecutive_observation_counts[native_track_id]
                >= self.minimum_confirmed_frames
            ):
                self.display_ids[native_track_id] = self.next_display_id
                self.next_display_id += 1
                is_confirmed = True
            tracked_detections.append(
                {
                    "xyxy": (float(x1), float(y1), float(x2), float(y2)),
                    "track_id": self.display_ids.get(native_track_id),
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                    "is_estimated": False,
                    "is_confirmed": is_confirmed,
                }
            )
        return tracked_detections
