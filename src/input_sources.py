"""Classify TrackSight inputs so file and live-source use are never conflated."""

from pathlib import Path
from urllib.parse import urlparse


SUPPORTED_LIVE_PROTOCOLS = {"rtsp", "rtsps", "http", "https"}


def classify_input_source(source):
    """Return a truthful source classification without opening a network stream."""
    source_text = str(source).strip()
    parsed = urlparse(source_text)
    if parsed.scheme.lower() in SUPPORTED_LIVE_PROTOCOLS and parsed.netloc:
        return {"source_type": "network_stream", "source": source_text, "requires_recording_output": True}
    path = Path(source_text)
    if path.is_file():
        return {"source_type": "video_file", "source": str(path), "requires_recording_output": False}
    raise ValueError("Input must be an existing video file or an rtsp/http(s) stream URL.")
