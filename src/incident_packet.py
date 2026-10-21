"""Create a small, portable evidence packet for one reviewer-selected case."""

import hashlib
import json
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from src.evidence_package import make_portable_summary


INCIDENT_PACKET_TYPE = "tracksight_incident_packet"
INCIDENT_PACKET_SCHEMA_VERSION = 1
INCIDENT_PACKET_README = """TrackSight incident packet

This packet preserves the evidence selected by a human reviewer for one
time-bounded incident thread. It does not decide fault, contact, a rule
violation, or an incident outcome. Review the supporting clip and the full
available race footage before making a decision.

The manifest contains SHA-256 hashes for every included file. Paths from the
computer that created the packet have been removed.
"""


def _portable_event(event):
    """Keep only event evidence, stripping source-machine paths."""
    return {
        key: deepcopy(value)
        for key, value in event.items()
        if key != "clip"
    }


def _current_decisions_for_thread(thread, decisions):
    """Return current human notes for the evidence IDs in one incident thread."""
    saved_decisions = decisions.get("decisions", {}) if isinstance(decisions, dict) else {}
    return [
        {"review_event_id": review_event_id, **deepcopy(saved_decisions[review_event_id])}
        for review_event_id in thread["review_event_ids"]
        if review_event_id in saved_decisions
    ]


def _build_packet_files(thread, run_summary, decisions, safety_zone_evidence):
    """Return the JSON payloads before they are hashed and zipped."""
    incident_thread = {
        key: deepcopy(value)
        for key, value in thread.items()
        if key not in {"events", "preferred_clip"}
    }
    incident_thread["supporting_clips"] = []
    packet_files = {
        "README.txt": INCIDENT_PACKET_README.encode("utf-8"),
        "incident_thread.json": json.dumps(incident_thread, indent=2).encode("utf-8"),
        "source_evidence.json": json.dumps(
            [_portable_event(event) for event in thread["events"]], indent=2
        ).encode("utf-8"),
        "run_summary.json": json.dumps(
            make_portable_summary(run_summary), indent=2
        ).encode("utf-8"),
        "human_review_decisions.json": json.dumps(
            _current_decisions_for_thread(thread, decisions), indent=2
        ).encode("utf-8"),
    }
    if safety_zone_evidence is not None:
        packet_files["safety_zone_context.json"] = json.dumps(
            safety_zone_evidence, indent=2
        ).encode("utf-8")
    return incident_thread, packet_files


def build_incident_packet(thread, run_summary, decisions, safety_zone_evidence=None):
    """Return a shareable ZIP containing one thread, evidence, and real clips.

    Only clips that already exist on disk are included. The packet records their
    hashes, so it does not claim that a missing supporting clip was available.
    """
    incident_thread, packet_files = _build_packet_files(
        thread,
        run_summary,
        decisions,
        safety_zone_evidence,
    )
    included_clips = {}
    for event in thread["events"]:
        clip_path = event.get("clip")
        if not isinstance(clip_path, str):
            continue
        clip_path = Path(clip_path)
        if not clip_path.is_file() or clip_path.name in included_clips:
            continue
        archive_name = f"supporting_clips/{clip_path.name}"
        included_clips[clip_path.name] = (archive_name, clip_path.read_bytes())
        incident_thread["supporting_clips"].append(archive_name)

    packet_files["incident_thread.json"] = json.dumps(
        incident_thread, indent=2
    ).encode("utf-8")
    for archive_name, content in included_clips.values():
        packet_files[archive_name] = content
    manifest = {
        "package_type": INCIDENT_PACKET_TYPE,
        "schema_version": INCIDENT_PACKET_SCHEMA_VERSION,
        "thread_id": thread["thread_id"],
        "files": [
            {
                "path": archive_name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
            for archive_name, content in sorted(packet_files.items())
        ],
    }
    packet_files["manifest.json"] = json.dumps(manifest, indent=2).encode("utf-8")
    archive_buffer = BytesIO()
    with ZipFile(archive_buffer, "w", compression=ZIP_DEFLATED) as archive:
        for archive_name, content in sorted(packet_files.items()):
            archive.writestr(archive_name, content)
    return archive_buffer.getvalue()
