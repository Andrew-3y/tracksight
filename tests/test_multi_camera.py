import unittest

from src.multi_camera import build_multi_camera_playback, validate_multi_camera_manifest


class MultiCameraTests(unittest.TestCase):
    def test_uses_a_reviewer_supplied_offset_without_claiming_auto_sync(self):
        manifest = {"cameras": [{"camera_id": "b", "label": "Camera B", "video_path": "missing.mp4", "video_time_offset_seconds": -2.0}]}
        playback = build_multi_camera_playback(10.0, manifest)

        self.assertEqual(playback[0]["playback_time_seconds"], 8.0)
        self.assertFalse(playback[0]["video_available"])

    def test_rejects_duplicate_camera_ids(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_multi_camera_manifest({"cameras": [
                {"camera_id": "a", "label": "A", "video_path": "a.mp4"},
                {"camera_id": "a", "label": "B", "video_path": "b.mp4"},
            ]})

    def test_rejects_empty_or_non_finite_camera_configuration(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            validate_multi_camera_manifest({"cameras": []})
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_multi_camera_manifest({"cameras": [{
                "camera_id": "a",
                "label": "Camera A",
                "video_path": "a.mp4",
                "video_time_offset_seconds": "nan",
            }]})
