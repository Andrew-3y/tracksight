import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from src.input_sources import classify_input_source


class InputSourceTests(unittest.TestCase):
    def test_classifies_network_stream_without_opening_it(self):
        result = classify_input_source("rtsp://camera.example/race")
        self.assertEqual(result["source_type"], "network_stream")
        self.assertTrue(result["requires_recording_output"])

    def test_classifies_existing_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "demo.mp4"
            path.touch()
            self.assertEqual(classify_input_source(path)["source_type"], "video_file")
