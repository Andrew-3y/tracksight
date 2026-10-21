import unittest

from src.approach_reconstruction import (
    build_approach_reconstruction,
    build_observed_approach_trace,
)


def point(frame_index, time_seconds, x_position, width=10.0):
    return {
        "frame_index": frame_index,
        "time_seconds": time_seconds,
        "track_position": (x_position, 20.0),
        "track_width_units": width,
    }


class ApproachReconstructionTests(unittest.TestCase):
    def test_reconstructs_paired_real_observations_and_net_closing(self):
        trajectories = {
            "1": [point(10, 0.4, 0), point(11, 0.44, 1), point(12, 0.48, 2)],
            "2": [point(10, 0.4, 30), point(11, 0.44, 20), point(12, 0.48, 12)],
        }
        event = {
            "track_ids": [1, 2],
            "start_frame": 10,
            "end_frame": 12,
            "start_time_seconds": 0.4,
            "end_time_seconds": 0.48,
            "highest_risk_frame": 12,
            "highest_risk_time_seconds": 0.48,
        }

        reconstruction = build_approach_reconstruction(
            trajectories,
            event,
            "Predicted risk",
        )

        self.assertEqual(len(reconstruction["trace"]), 3)
        self.assertEqual(reconstruction["summary"]["net_trend"], "net_closing")
        self.assertEqual(reconstruction["summary"]["closest_frame"], 12)
        self.assertEqual(reconstruction["summary"]["missing_pair_frames"], 0)
        self.assertEqual(
            [item["moment"] for item in reconstruction["timeline"]],
            [
                "Evidence window began",
                "Saved evidence anchor",
                "Evidence window ended",
            ],
        )

    def test_omits_detection_gaps_instead_of_inventing_distance(self):
        trajectories = {
            "1": [point(10, 0.4, 0), point(12, 0.48, 2)],
            "2": [point(10, 0.4, 20), point(11, 0.44, 15), point(12, 0.48, 12)],
        }

        trace = build_observed_approach_trace(trajectories, [1, 2], 10, 12)

        self.assertEqual([item["frame"] for item in trace], [10, 12])

    def test_withholds_a_trend_when_no_paired_calibrated_observations_exist(self):
        trajectories = {
            "1": [point(10, 0.4, 0)],
            "2": [{"frame_index": 10, "time_seconds": 0.4}],
        }
        event = {
            "track_ids": [1, 2],
            "start_frame": 10,
            "end_frame": 10,
            "start_seconds": 0.4,
            "end_seconds": 0.4,
            "closest_frame": 10,
            "closest_time_seconds": 0.4,
        }

        reconstruction = build_approach_reconstruction(
            trajectories,
            event,
            "Observed proximity",
        )

        self.assertEqual(
            reconstruction["summary"]["status"],
            "withheld_without_paired_real_observations",
        )
        self.assertEqual(reconstruction["summary"]["missing_pair_frames"], 1)


if __name__ == "__main__":
    unittest.main()
