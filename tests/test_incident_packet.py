import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from src.incident_packet import build_incident_packet


class IncidentPacketTests(unittest.TestCase):
    def test_packet_is_portable_and_hashes_the_selected_supporting_clip(self):
        with TemporaryDirectory() as temporary_directory:
            clip_path = Path(temporary_directory) / "support.mp4"
            clip_path.write_bytes(b"clip evidence")
            thread = {
                "thread_id": "thread-predicted-0",
                "review_event_ids": ["predicted_risk:25:1-2"],
                "events": [
                    {
                        "event_key": "predicted-0",
                        "review_event_id": "predicted_risk:25:1-2",
                        "clip": str(clip_path),
                    }
                ],
                "preferred_clip": str(clip_path),
            }
            packet = build_incident_packet(
                thread,
                {"run_provenance": {"source_video": r"C:\\private\\race.mp4"}},
                {"decisions": {}},
            )

            with ZipFile(__import__("io").BytesIO(packet)) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                archive_names = set(archive.namelist())
                source_evidence = json.loads(archive.read("source_evidence.json"))

            self.assertIn("supporting_clips/support.mp4", archive_names)
            self.assertNotIn("clip", source_evidence[0])
            clip_entry = next(
                item for item in manifest["files"] if item["path"] == "supporting_clips/support.mp4"
            )
            self.assertEqual(clip_entry["sha256"], hashlib.sha256(b"clip evidence").hexdigest())


if __name__ == "__main__":
    unittest.main()
