"""Verify the presence and optional hashes of generated TrackSight evidence."""

import hashlib
from pathlib import Path


EVIDENCE_INTEGRITY_DISCLAIMER = (
    "Hashes are a point-in-time integrity check of saved output files. They do not "
    "prove the underlying video is authentic, the model is accurate, or an event occurred."
)

_ARTIFACTS = (
    ("Annotated camera video", "annotated_video", True),
    ("Trajectory observations", "trajectory_json", True),
    ("Run summary", "run_summary_json", True),
    ("Observed-proximity evidence", "review_evidence_json", False),
    ("Predicted-risk evidence", "risk_evidence_json", False),
    ("Human-review decisions", "human_review_decisions_json", False),
    ("Race-control actions", "race_control_action_log_json", False),
    ("Reviewer corrections", "review_corrections_json", False),
    ("Independent second opinions", "second_opinions_json", False),
)


def file_sha256(path, block_size=1024 * 1024):
    """Return the SHA-256 digest without loading an evidence file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as evidence_file:
        for block in iter(lambda: evidence_file.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def build_evidence_integrity(artifacts, include_hashes=False):
    """Return path-free availability and optional checksum rows for one run."""
    rows = []
    for label, artifact_key, required in _ARTIFACTS:
        path = Path(artifacts[artifact_key])
        if path.is_file():
            rows.append(
                {
                    "artifact": label,
                    "status": "Present",
                    "file_name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path) if include_hashes else "Not calculated",
                    "required_for_saved_run": required,
                }
            )
        else:
            rows.append(
                {
                    "artifact": label,
                    "status": "Missing" if required else "Not created",
                    "file_name": path.name,
                    "size_bytes": None,
                    "sha256": "Unavailable",
                    "required_for_saved_run": required,
                }
            )
    return rows


def build_integrity_manifest(run_name, rows):
    """Return a portable, shareable output-integrity manifest without local paths."""
    return {
        "manifest_type": "tracksight_output_integrity",
        "run_name": run_name,
        "disclaimer": EVIDENCE_INTEGRITY_DISCLAIMER,
        "hashes_calculated": any(row["sha256"] not in {"Not calculated", "Unavailable"} for row in rows),
        "artifacts": rows,
    }
