import unittest

from src.evidence_chain import build_evidence_chain


class EvidenceChainTests(unittest.TestCase):
    def test_marks_an_untrusted_id_as_withheld(self):
        stages = build_evidence_chain(
            {"track_ids": [1, 2], "preferred_clip": None},
            [{"event_key": "observed-0"}],
            {
                "spatial_analysis_enabled": True,
                "track_quality": {
                    "1": {"trusted_for_safety_analysis": True},
                    "2": {"trusted_for_safety_analysis": False},
                },
            },
            annotated_video_exists=True,
        )
        continuity = next(stage for stage in stages if stage["stage"] == "Vehicle-ID continuity")
        self.assertEqual(continuity["status"], "Withheld")

    def test_uses_full_video_as_honest_clip_fallback(self):
        stages = build_evidence_chain(
            {"track_ids": [1, 2], "preferred_clip": None},
            [],
            {"spatial_analysis_enabled": False, "track_quality": {}},
            annotated_video_exists=True,
        )
        replay = next(stage for stage in stages if stage["stage"] == "Supporting replay")
        self.assertEqual(replay["status"], "Full video fallback")


if __name__ == "__main__":
    unittest.main()
