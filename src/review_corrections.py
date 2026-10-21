"""Keep reviewer correction records separate from immutable detection evidence."""

import json
from copy import deepcopy
from datetime import datetime, timezone


REVIEW_CORRECTION_SCHEMA_VERSION = 1
CORRECTION_TYPES = (
    "Correct detection",
    "Missed vehicle",
    "Merged vehicles",
    "Incorrect ID",
    "False positive",
    "Other",
)
REVIEW_CORRECTIONS_DISCLAIMER = (
    "Reviewer corrections are human feedback, not machine-generated labels. "
    "They require independent frame-level annotation and dataset validation before model training or evaluation."
)


def empty_review_corrections():
    """Return the append-only record format for one completed run."""
    return {"schema_version": REVIEW_CORRECTION_SCHEMA_VERSION, "corrections": []}


def load_review_corrections(path):
    """Load an existing correction record or create an empty in-memory record."""
    if not path.is_file():
        return empty_review_corrections()
    with path.open(encoding="utf-8") as correction_file:
        corrections = json.load(correction_file)
    if (
        not isinstance(corrections, dict)
        or corrections.get("schema_version") != REVIEW_CORRECTION_SCHEMA_VERSION
        or not isinstance(corrections.get("corrections"), list)
    ):
        raise ValueError("Reviewer corrections use an unsupported file format.")
    return corrections


def append_review_correction(corrections, event, correction_type, reviewer_name, note, recorded_at=None):
    """Append one reviewer observation without changing prior evidence or corrections."""
    if correction_type not in CORRECTION_TYPES:
        raise ValueError("Choose a supported correction type.")
    if not reviewer_name or not reviewer_name.strip():
        raise ValueError("Reviewer name is required.")
    if not note or not note.strip():
        raise ValueError("Describe what should be independently annotated.")
    if len(reviewer_name.strip()) > 100 or len(note.strip()) > 2000:
        raise ValueError("Reviewer name or correction note is too long.")
    updated = deepcopy(corrections)
    if updated.get("schema_version") != REVIEW_CORRECTION_SCHEMA_VERSION:
        raise ValueError("Reviewer corrections use an unsupported file format.")
    updated.setdefault("corrections", [])
    updated["corrections"].append({
        "correction_id": f"correction-{len(updated['corrections']) + 1}",
        "recorded_at": recorded_at or datetime.now(timezone.utc).isoformat(),
        "reviewer_name": reviewer_name.strip(),
        "correction_type": correction_type,
        "annotation_note": note.strip(),
        "evidence_event": {
            "event_key": str(event["event_key"]),
            "evidence_type": str(event["evidence_type"]),
            "track_ids": list(event["track_id_values"]),
            "frame": int(event["frame"]),
            "time_seconds": float(event["time_seconds"]),
        },
    })
    return updated


def write_review_corrections(corrections, path):
    """Persist the append-only correction record beside the run artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as correction_file:
        json.dump(corrections, correction_file, indent=2)
    return path


def build_correction_learning_export(run_name, corrections):
    """Create a portable training-handoff manifest without claiming labels exist."""
    return {
        "export_type": "tracksight_reviewer_corrections",
        "schema_version": REVIEW_CORRECTION_SCHEMA_VERSION,
        "disclaimer": REVIEW_CORRECTIONS_DISCLAIMER,
        "run_name": run_name,
        "correction_count": len(corrections.get("corrections", [])),
        "corrections": deepcopy(corrections.get("corrections", [])),
    }
