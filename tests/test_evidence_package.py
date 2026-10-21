import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from src.evidence_package import create_marshal_evidence_package


class MarshalEvidencePackageTests(unittest.TestCase):
    def test_creates_a_portable_complete_reviewer_handoff(self):
        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            clip_path = temporary_path / "observed_proximity_001.mp4"
            clip_path.write_bytes(b"annotated clip")
            package_path = temporary_path / "marshal_evidence_package.zip"
            briefing = {
                "run": {"source_video": r"C:\\private\\race_video.mp4"},
                "predicted_risk_highlights": [],
                "observed_proximity_highlights": [
                    {"supporting_clip": str(clip_path)}
                ],
            }
            run_summary = {
                "run_provenance": {
                    "source_video": r"C:\\private\\race_video.mp4",
                    "detector_model": r"C:\\private\\model.pt",
                    "regions_profile": r"C:\\private\\regions.json",
                    "calibration_profile": r"C:\\private\\calibration.json",
                }
            }
            review_evidence = [
                {
                    "track_ids": [1, 2],
                    "review_clip_path": r"C:\\private\\review.mp4",
                    "marshal_briefing_clip_path": str(clip_path),
                }
            ]

            create_marshal_evidence_package(
                package_path,
                briefing,
                run_summary,
                [],
                review_evidence,
                {"1": [{"frame_index": 0}]},
            )

            with ZipFile(package_path) as archive:
                archive_names = set(archive.namelist())
                self.assertIn("README.txt", archive_names)
                self.assertIn("marshal_briefing.json", archive_names)
                self.assertIn(
                    "briefing_clips/observed_proximity_001.mp4", archive_names
                )
                packaged_briefing = json.loads(
                    archive.read("marshal_briefing.json")
                )
                packaged_summary = json.loads(archive.read("run_summary.json"))
                packaged_evidence = json.loads(
                    archive.read("observed_proximity_evidence.json")
                )

            self.assertEqual(
                packaged_briefing["run"]["source_video"], "race_video.mp4"
            )
            self.assertEqual(
                packaged_briefing["observed_proximity_highlights"][0]["supporting_clip"],
                "briefing_clips/observed_proximity_001.mp4",
            )
            self.assertEqual(
                packaged_summary["run_provenance"]["detector_model"], "model.pt"
            )
            self.assertNotIn("review_clip_path", packaged_evidence[0])
            self.assertNotIn("marshal_briefing_clip_path", packaged_evidence[0])

    def test_rejects_a_briefing_without_a_saved_clip(self):
        with TemporaryDirectory() as temporary_directory:
            missing_clip = Path(temporary_directory) / "missing.mp4"

            with self.assertRaisesRegex(ValueError, "supporting clip"):
                create_marshal_evidence_package(
                    Path(temporary_directory) / "package.zip",
                    {
                        "predicted_risk_highlights": [
                            {"supporting_clip": str(missing_clip)}
                        ],
                        "observed_proximity_highlights": [],
                    },
                    {},
                    [],
                    [],
                    {},
                )

    def test_includes_optional_safety_zone_evidence(self):
        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            clip_path = temporary_path / "support.mp4"
            clip_path.write_bytes(b"clip")
            package_path = temporary_path / "package.zip"

            create_marshal_evidence_package(
                package_path,
                {
                    "predicted_risk_highlights": [
                        {"supporting_clip": str(clip_path)}
                    ],
                    "observed_proximity_highlights": [],
                },
                {},
                [],
                [],
                {},
                safety_zone_evidence={"zones": [], "observations": []},
            )

            with ZipFile(package_path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertIn("safety_zone_evidence.json", archive.namelist())
                self.assertIn("safety_zone_evidence.json", manifest["contents"])

    def test_includes_portable_safety_candidate_evidence(self):
        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            clip_path = temporary_path / "stationary_vehicle.mp4"
            clip_path.write_bytes(b"clip")
            package_path = temporary_path / "package.zip"
            candidate_evidence = {
                "schema_version": 1,
                "stationary_vehicle_candidates": [
                    {
                        "track_ids": [7],
                        "marshal_briefing_clip_path": str(clip_path),
                    }
                ],
                "restricted_area_entry_candidates": [],
            }

            create_marshal_evidence_package(
                package_path,
                {
                    "predicted_risk_highlights": [],
                    "observed_proximity_highlights": [],
                    "safety_candidate_highlights": [
                        {"supporting_clip": str(clip_path)}
                    ],
                },
                {},
                [],
                [],
                {},
                safety_candidate_evidence=candidate_evidence,
            )

            with ZipFile(package_path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                packaged_candidates = json.loads(
                    archive.read("safety_candidate_evidence.json")
                )

            self.assertIn("safety_candidate_evidence.json", manifest["contents"])
            self.assertNotIn(
                "marshal_briefing_clip_path",
                packaged_candidates["stationary_vehicle_candidates"][0],
            )


if __name__ == "__main__":
    unittest.main()
