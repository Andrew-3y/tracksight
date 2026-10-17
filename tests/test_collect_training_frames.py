import unittest

from scripts.collect_training_frames import should_sample_frame


class CollectTrainingFramesTests(unittest.TestCase):
    def test_start_time_becomes_the_first_sample(self):
        self.assertTrue(should_sample_frame(60, start_frame_index=60, frame_step=120))

    def test_following_sample_is_relative_to_start_time(self):
        self.assertTrue(should_sample_frame(180, start_frame_index=60, frame_step=120))

    def test_old_absolute_grid_is_not_mistaken_for_a_sample(self):
        self.assertFalse(should_sample_frame(120, start_frame_index=60, frame_step=120))


if __name__ == "__main__":
    unittest.main()
