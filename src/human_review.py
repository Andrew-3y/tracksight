"""Store and export auditable human decisions about TrackSight evidence."""

import json
from copy import deepcopy
from datetime import datetime, timezone


HUMAN_REVIEW_OUTCOMES = ("Needs review", "Cleared", "Escalate")
HUMAN_REVIEW_DISCLAIMER = (
    "Human-review notes are recorded separately from TrackSight evidence. "
    "They do not establish fault, contact, a rule violation, or an incident outcome."
)
HUMAN_REVIEW_SCHEMA_VERSION = 1


def make_review_event_id(evidence_type, track_ids, frame):
    """Create a deterministic per-run identifier for one evidence event."""
    normalized_track_ids = "-".join(str(track_id) for track_id in sorted(track_ids))
    normalized_type = evidence_type.lower().replace(" ", "_")
    return f"{normalized_type}:{frame}:{normalized_track_ids}"


def empty_human_review_decisions():
    """Return the initial file structure for a completed run's review notes."""
    return {
        "schema_version": HUMAN_REVIEW_SCHEMA_VERSION,
        "decisions": {},
    }


def load_human_review_decisions(decision_path):
    """Load saved human decisions, or return an empty record for a new run."""
    if not decision_path.is_file():
        return empty_human_review_decisions()
    with decision_path.open(encoding="utf-8") as decision_file:
        decisions = json.load(decision_file)
    if (
        not isinstance(decisions, dict)
        or decisions.get("schema_version") != HUMAN_REVIEW_SCHEMA_VERSION
        or not isinstance(decisions.get("decisions"), dict)
    ):
        raise ValueError("Human-review decisions use an unsupported file format.")
    return decisions


def validate_human_review_decision(outcome, reviewer_name, review_note):
    """Reject incomplete or unsupported user-entered review decisions."""
    if outcome not in HUMAN_REVIEW_OUTCOMES:
        raise ValueError("Choose a valid human-review outcome.")
    if not reviewer_name or not reviewer_name.strip():
        raise ValueError("Reviewer name is required.")
    if not review_note or not review_note.strip():
        raise ValueError("A review note is required.")
    if len(reviewer_name.strip()) > 100:
        raise ValueError("Reviewer name must be 100 characters or fewer.")
    if len(review_note.strip()) > 2000:
        raise ValueError("Review note must be 2,000 characters or fewer.")


def is_current_human_review_decision(decision, event_summary):
    """Return whether a saved decision still describes the current evidence."""
    return isinstance(decision, dict) and decision.get("event") == event_summary


def filter_current_human_review_decisions(decisions, event_summaries):
    """Keep decisions that still match the current run's source evidence."""
    current_decisions = empty_human_review_decisions()
    saved_decisions = decisions.get("decisions", {})
    for event_id, event_summary in event_summaries.items():
        saved_decision = saved_decisions.get(event_id)
        if is_current_human_review_decision(saved_decision, event_summary):
            current_decisions["decisions"][event_id] = deepcopy(saved_decision)
    return current_decisions


def save_human_review_decision(
    decisions,
    event_id,
    event_summary,
    outcome,
    reviewer_name,
    review_note,
    reviewed_at=None,
):
    """Add or update a human decision without changing the source evidence."""
    validate_human_review_decision(outcome, reviewer_name, review_note)
    updated_decisions = deepcopy(decisions)
    updated_decisions.setdefault("schema_version", HUMAN_REVIEW_SCHEMA_VERSION)
    updated_decisions.setdefault("decisions", {})
    if updated_decisions["schema_version"] != HUMAN_REVIEW_SCHEMA_VERSION:
        raise ValueError("Human-review decisions use an unsupported file format.")
    timestamp = reviewed_at or datetime.now(timezone.utc).isoformat()
    previous_decision = updated_decisions["decisions"].get(event_id, {})
    history = deepcopy(previous_decision.get("history", []))
    if previous_decision and not history:
        history.append(
            {
                "outcome": previous_decision.get("outcome"),
                "reviewer_name": previous_decision.get("reviewer_name"),
                "review_note": previous_decision.get("review_note"),
                "recorded_at": previous_decision.get("updated_at") or previous_decision.get("created_at"),
                "legacy_import": True,
            }
        )
    history.append(
        {
            "outcome": outcome,
            "reviewer_name": reviewer_name.strip(),
            "review_note": review_note.strip(),
            "recorded_at": timestamp,
        }
    )
    updated_decisions["decisions"][event_id] = {
        "event": deepcopy(event_summary),
        "outcome": outcome,
        "reviewer_name": reviewer_name.strip(),
        "review_note": review_note.strip(),
        "created_at": previous_decision.get("created_at", timestamp),
        "updated_at": timestamp,
        "history": history,
    }
    return updated_decisions


def write_human_review_decisions(decisions, decision_path):
    """Persist a per-run review file beside the immutable source artifacts."""
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    with decision_path.open("w", encoding="utf-8") as decision_file:
        json.dump(decisions, decision_file, indent=2)
    return decision_path


def build_human_review_report(run_name, decisions):
    """Create a portable, path-free export of one run's saved human decisions."""
    return {
        "report_type": "tracksight_human_review_decisions",
        "schema_version": HUMAN_REVIEW_SCHEMA_VERSION,
        "disclaimer": HUMAN_REVIEW_DISCLAIMER,
        "run_name": run_name,
        "decisions": list(decisions["decisions"].values()),
    }
