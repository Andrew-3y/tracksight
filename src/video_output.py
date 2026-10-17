"""Create MP4 files that browsers can play reliably."""

from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg


def create_working_video_writer(output_path, fps, frame_size):
    """Open an OpenCV writer for an intermediate video file.

    OpenCV is used while frames are being produced because it accepts one
    NumPy image at a time.  Its widely available ``mp4v`` encoder is only an
    intermediate format: it is re-encoded to H.264 before TrackSight exposes
    the result in the dashboard.
    """
    output_path = Path(output_path)
    working_path = output_path.with_name(f"{output_path.stem}.working.mp4")
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(working_path), codec, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create working video: {working_path}")
    return writer, working_path


def finalize_browser_video(working_path, output_path):
    """Encode a finished working video as a web-compatible H.264 MP4.

    H.264 with ``yuv420p`` is supported by the browsers used by Streamlit.
    ``+faststart`` moves MP4 metadata to the beginning of the file, allowing
    the dashboard player to start loading it without first downloading the
    entire video.
    """
    working_path = Path(working_path)
    output_path = Path(output_path)
    staged_path = output_path.with_name(f"{output_path.stem}.browser.mp4")
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    conversion = subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(working_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(staged_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if conversion.returncode != 0:
        raise RuntimeError(
            "Could not encode a browser-compatible MP4: "
            f"{conversion.stderr.strip()}"
        )
    staged_path.replace(output_path)
    working_path.unlink(missing_ok=True)
    return output_path
