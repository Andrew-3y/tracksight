import unittest

from src.native_tracking import (
    MotionCompensatedVehicleTracker,
    sort_track_rows_for_display_ids,
)


class MotionCompensatedVehicleTrackerTests(unittest.TestCase):
    def test_rejects_reversed_confidence_thresholds(self):
        with self.assertRaisesRegex(ValueError, "low confidence <= high confidence"):
            MotionCompensatedVehicleTracker(
                fps=60,
                max_missing_seconds=1.5,
                high_confidence_threshold=0.2,
                low_confidence_threshold=0.4,
                new_track_confidence_threshold=0.3,
                match_threshold=0.8,
            )

    def test_rejects_new_track_threshold_below_high_threshold(self):
        with self.assertRaisesRegex(ValueError, "New-track confidence"):
            MotionCompensatedVehicleTracker(
                fps=60,
                max_missing_seconds=1.5,
                high_confidence_threshold=0.4,
                low_confidence_threshold=0.2,
                new_track_confidence_threshold=0.3,
                match_threshold=0.8,
            )

    def test_accepts_the_default_recovery_thresholds(self):
        tracker = MotionCompensatedVehicleTracker(
            fps=60,
            max_missing_seconds=1.5,
            high_confidence_threshold=0.3,
            low_confidence_threshold=0.2,
            new_track_confidence_threshold=0.3,
            match_threshold=0.8,
        )

        self.assertEqual(tracker.tracker.max_frames_lost, 90)
        self.assertEqual(tracker.minimum_confirmed_frames, 12)

    def test_rejects_zero_confirmation_seconds(self):
        with self.assertRaisesRegex(ValueError, "Minimum confirmed seconds"):
            MotionCompensatedVehicleTracker(
                fps=60,
                max_missing_seconds=1.5,
                high_confidence_threshold=0.3,
                low_confidence_threshold=0.2,
                new_track_confidence_threshold=0.45,
                match_threshold=0.8,
                minimum_confirmed_seconds=0,
            )

    def test_only_publishes_an_id_after_confirmation(self):
        class FakeBoTSORT:
            def update(self, _boxes, _frame):
                return [[10, 20, 30, 40, 99, 0.9, 2, 0]]

        class FakeBoxes:
            def cpu(self):
                return self

            def numpy(self):
                return []

        tracker = object.__new__(MotionCompensatedVehicleTracker)
        tracker.tracker = FakeBoTSORT()
        tracker.consecutive_observation_counts = {}
        tracker.last_seen_update_indexes = {}
        tracker.update_index = 0
        tracker.display_ids = {}
        tracker.next_display_id = 1
        tracker.minimum_confirmed_frames = 3

        boxes = FakeBoxes()
        first = tracker.update(boxes=boxes, frame=None)[0]
        second = tracker.update(boxes=boxes, frame=None)[0]
        third = tracker.update(boxes=boxes, frame=None)[0]

        self.assertFalse(first["is_confirmed"])
        self.assertIsNone(first["track_id"])
        self.assertFalse(second["is_confirmed"])
        self.assertIsNone(second["track_id"])
        self.assertTrue(third["is_confirmed"])
        self.assertEqual(third["track_id"], 1)

    def test_confirmation_requires_consecutive_observations(self):
        class FakeBoTSORT:
            def __init__(self):
                self.rows = [
                    [[10, 20, 30, 40, 99, 0.9, 2, 0]],
                    [[10, 20, 30, 40, 99, 0.9, 2, 0]],
                    [],
                    [[10, 20, 30, 40, 99, 0.9, 2, 0]],
                    [[10, 20, 30, 40, 99, 0.9, 2, 0]],
                    [[10, 20, 30, 40, 99, 0.9, 2, 0]],
                ]

            def update(self, _boxes, _frame):
                return self.rows.pop(0)

        class FakeBoxes:
            def cpu(self):
                return self

            def numpy(self):
                return []

        tracker = object.__new__(MotionCompensatedVehicleTracker)
        tracker.tracker = FakeBoTSORT()
        tracker.consecutive_observation_counts = {}
        tracker.last_seen_update_indexes = {}
        tracker.update_index = 0
        tracker.display_ids = {}
        tracker.next_display_id = 1
        tracker.minimum_confirmed_frames = 3
        boxes = FakeBoxes()

        tracker.update(boxes, None)
        tracker.update(boxes, None)
        tracker.update(boxes, None)
        after_gap = tracker.update(boxes, None)[0]
        before_confirmation = tracker.update(boxes, None)[0]
        confirmed = tracker.update(boxes, None)[0]

        self.assertFalse(after_gap["is_confirmed"])
        self.assertFalse(before_confirmation["is_confirmed"])
        self.assertTrue(confirmed["is_confirmed"])


    def test_sorts_simultaneous_new_rows_in_screen_order(self):
        rows = [
            [500, 400, 550, 450, 30, 0.90, 2, 0],
            [200, 100, 250, 150, 20, 0.90, 2, 0],
            [100, 100, 150, 150, 10, 0.90, 2, 0],
        ]

        sorted_rows = sort_track_rows_for_display_ids(rows)

        self.assertEqual(
            [int(row[4]) for row in sorted_rows],
            [10, 20, 30],
        )

if __name__ == "__main__":
    unittest.main()
