import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.dashboard_analysis import (
    build_tracking_command,
    get_matching_profile_path,
    make_dashboard_output_path,
    persist_uploaded_video,
    run_tracking_command,
)


class DashboardAnalysisTests(unittest.TestCase):
    def test_selects_only_a_profile_named_for_the_video(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "demo.mp4"
            config_directory = root / "configs"
            video_path.touch()
            config_directory.mkdir()
            matching_profile = config_directory / "demo_calibration.json"
            matching_profile.touch()

            self.assertEqual(
                get_matching_profile_path(video_path, config_directory, "_calibration"),
                matching_profile,
            )

    def test_output_path_does_not_overwrite_existing_evidence(self):
        with TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            output_path = make_dashboard_output_path(output_directory, "Demo run 1")

            self.assertEqual(output_path.name, "annotated_demo_run_1.mp4")
            output_path.touch()
            with self.assertRaisesRegex(FileExistsError, "existing evidence"):
                make_dashboard_output_path(output_directory, "Demo run 1")

    def test_uploaded_video_uses_a_safe_content_addressed_filename(self):
        with TemporaryDirectory() as temporary_directory:
            saved_path = persist_uploaded_video(
                "..\\My demo video.MP4",
                b"video bytes",
                Path(temporary_directory),
            )

            self.assertTrue(saved_path.is_file())
            self.assertTrue(saved_path.name.startswith("my_demo_video_"))
            self.assertEqual(saved_path.read_bytes(), b"video bytes")

    def test_tracking_command_uses_argument_list_not_a_shell(self):
        with TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            video_path = project_root / "data" / "raw" / "demo.mp4"
            model_path = project_root / "models" / "model.pt"
            output_path = project_root / "outputs" / "tracking" / "annotated_demo.mp4"
            video_path.parent.mkdir(parents=True)
            model_path.parent.mkdir(parents=True)
            output_path.parent.mkdir(parents=True)
            video_path.touch()
            model_path.touch()

            command = build_tracking_command(
                project_root,
                video_path,
                model_path,
                output_path,
                python_executable="python",
            )

            self.assertEqual(command[:3], ["python", "-m", "src.run_video_tracking"])
            self.assertIn(str(output_path.resolve()), command)

    def test_tracking_command_includes_an_explicit_safety_zone_profile(self):
        with TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            video_path = project_root / "data" / "raw" / "demo.mp4"
            model_path = project_root / "models" / "model.pt"
            output_path = project_root / "outputs" / "tracking" / "annotated_demo.mp4"
            zone_path = project_root / "configs" / "demo_safety_zones.json"
            video_path.parent.mkdir(parents=True)
            model_path.parent.mkdir(parents=True)
            output_path.parent.mkdir(parents=True)
            zone_path.parent.mkdir(parents=True)
            video_path.touch()
            model_path.touch()
            zone_path.write_text("{}", encoding="utf-8")

            command = build_tracking_command(
                project_root,
                video_path,
                model_path,
                output_path,
                safety_zone_config_path=zone_path,
                python_executable="python",
            )

            self.assertIn("--safety-zone-config", command)
            self.assertIn(str(zone_path.resolve()), command)

    def test_runner_reports_the_pipeline_error_text(self):
        def failing_runner(*args, **kwargs):
            return subprocess.CompletedProcess(args[0], 1, stderr="Detector failed")

        with self.assertRaisesRegex(RuntimeError, "Detector failed"):
            run_tracking_command(
                ["python", "-m", "src.run_video_tracking"],
                Path.cwd(),
                command_runner=failing_runner,
            )

    def test_runner_reports_when_python_cannot_be_started(self):
        def unavailable_runner(*args, **kwargs):
            raise OSError("Python executable is unavailable")

        with self.assertRaisesRegex(RuntimeError, "Could not start local"):
            run_tracking_command(
                ["python", "-m", "src.run_video_tracking"],
                Path.cwd(),
                command_runner=unavailable_runner,
            )


if __name__ == "__main__":
    unittest.main()
