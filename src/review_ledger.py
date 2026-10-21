"""Preserve independent human review context without changing source evidence."""

import json
from copy import deepcopy
from datetime import datetime, timezone


SECOND_OPINION_SCHEMA_VERSION = 1
SECOND_OPINION_TYPES = (
    "Agree with recorded decision",
    "Disagree with recorded decision",
    "Request more evidence",
)
SECOND_OPINION_DISCLAIMER = (
    "A second opinion records an independent human view. It does not automatically "
    "settle a disagreement, change source evidence, or establish fault, contact, "
    "a rule breach, or an incident outcome."
)
DECISION_LEDGER_DISCLAIMER = (
    "The ledger joins immutable source evidence with saved human review records. "
    "It is an audit aid, not an automated incident decision."
)


def empty_second_opinions():
    """Return the append-only format for independent reviewer opinions."""
    return {"schema_version": SECOND_OPINION_SCHEMA_VERSION, "opinions": []}


def load_second_opinions(path):
    """Load independent opinions, or return an empty record for a new run."""
    if not path.is_file():
        return empty_second_opinions()
    with path.open(encoding="utf-8") as opinion_file:
        opinions = json.load(opinion_file)
    if (
        not isinstance(opinions, dict)
        or opinions.get("schema_version") != SECOND_OPINION_SCHEMA_VERSION
        or not isinstance(opinions.get("opinions"), list)
    ):
        raise ValueError("Second-review opinions use an unsupported file format.")
    return opinions


def append_second_opinion(
    opinions,
    event,
    recorded_decision,
    opinion_type,
    reviewer_name,
    note,
    recorded_at=None,
):
    """Append a separate reviewer opinion without replacing the primary decision."""
    if opinion_type not in SECOND_OPINION_TYPES:
        raise ValueError("Choose a supported second-review opinion.")
    if not reviewer_name or not reviewer_name.strip():
        raise ValueError("Second reviewer name is required.")
    if not note or not note.strip():
        raise ValueError("Explain the independent review observation.")
    if len(reviewer_name.strip()) > 100 or len(note.strip()) > 2000:
        raise ValueError("Second reviewer name or note is too long.")
    primary_reviewer = str(recorded_decision.get("reviewer_name", "")).strip()
    if primary_reviewer.casefold() == reviewer_name.strip().casefold():
        raise ValueError("A second opinion must be recorded by a different reviewer.")
    updated = deepcopy(opinions)
    if updated.get("schema_version") != SECOND_OPINION_SCHEMA_VERSION:
        raise ValueError("Second-review opinions use an unsupported file format.")
    entries = updated.setdefault("opinions", [])
    entries.append(
        {
            "opinion_id": f"second-opinion-{len(entries) + 1}",
            "recorded_at": recorded_at or datetime.now(timezone.utc).isoformat(),
            "reviewer_name": reviewer_name.strip(),
            "opinion_type": opinion_type,
            "note": note.strip(),
            "recorded_decision_snapshot": {
                "outcome": recorded_decision["outcome"],
                "reviewer_name": primary_reviewer,
                "updated_at": recorded_decision["updated_at"],
            },
            "evidence_event": {
                "review_event_id": str(event["review_event_id"]),
                "evidence_type": str(event["evidence_type"]),
                "track_ids": list(event["track_id_values"]),
                "frame": int(event["frame"]),
                "time_seconds": float(event["time_seconds"]),
            },
        }
    )
    return updated


def write_second_opinions(opinions, path):
    """Persist append-only independent opinions beside the completed run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as opinion_file:
        json.dump(opinions, opinion_file, indent=2)
    return path


def _human_review_history(decisions):
    """Return current and historic human-review records in a common ledger shape."""
    records = []
    for event_id, decision in decisions.get("decisions", {}).items():
        history = decision.get("history")
        if not history:
            history = [{
                "outcome": decision.get("outcome"),
                "reviewer_name": decision.get("reviewer_name"),
                "review_note": decision.get("review_note"),
                "recorded_at": decision.get("updated_at") or decision.get("created_at"),
                "legacy_import": True,
            }]
        for item in history:
            records.append(
                {
                    "recorded_at": item.get("recorded_at"),
                    "record_type": "Human review decision",
                    "event_reference": event_id,
                    "reviewer": item.get("reviewer_name", "—"),
                    "decision_or_action": item.get("outcome", "—"),
                    "note": item.get("review_note", "—"),
                    "recording_mode": "Imported current decision" if item.get("legacy_import") else "Versioned decision record",
                }
            )
    return records


def build_decision_ledger(decisions, action_log, corrections, second_opinions):
    """Build portable review-history rows from saved human-controlled records."""
    records = _human_review_history(decisions)
    for item in action_log.get("actions", []):
        records.append(
            {
                "recorded_at": item.get("recorded_at"),
                "record_type": "Race-control action",
                "event_reference": item.get("case_id", "—"),
                "reviewer": item.get("operator_name", "—"),
                "decision_or_action": item.get("action", "—"),
                "note": item.get("action_note", "—"),
                "recording_mode": "Append-only action",
            }
        )
    for item in corrections.get("corrections", []):
        event = item.get("evidence_event", {})
        records.append(
            {
                "recorded_at": item.get("recorded_at"),
                "record_type": "Reviewer correction",
                "event_reference": event.get("event_key", "—"),
                "reviewer": item.get("reviewer_name", "—"),
                "decision_or_action": item.get("correction_type", "—"),
                "note": item.get("annotation_note", "—"),
                "recording_mode": "Append-only correction",
            }
        )
    for item in second_opinions.get("opinions", []):
        event = item.get("evidence_event", {})
        records.append(
            {
                "recorded_at": item.get("recorded_at"),
                "record_type": "Independent second opinion",
                "event_reference": event.get("review_event_id", "—"),
                "reviewer": item.get("reviewer_name", "—"),
                "decision_or_action": item.get("opinion_type", "—"),
                "note": item.get("note", "—"),
                "recording_mode": "Append-only second opinion",
            }
        )
    return sorted(records, key=lambda record: (str(record["recorded_at"]), record["record_type"]))
