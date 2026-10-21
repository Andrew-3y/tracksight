"""Append-only human race-control action records for TrackSight review cases."""

import json
from copy import deepcopy
from datetime import datetime, timezone


RACE_CONTROL_ACTIONS = (
    "Monitor",
    "Request additional review",
    "Escalate to race control",
    "Close as reviewed",
)
RACE_CONTROL_LOG_SCHEMA_VERSION = 1
REQUIRED_REVIEW_CHECKS = (
    "supporting_video_reviewed",
    "identity_continuity_considered",
    "camera_trust_reviewed",
)
RACE_CONTROL_LOG_DISCLAIMER = (
    "This log records human actions around TrackSight review material. It does not "
    "alter source evidence or establish fault, contact, a rule violation, or an incident outcome."
)


def empty_race_control_action_log():
    return {"schema_version": RACE_CONTROL_LOG_SCHEMA_VERSION, "actions": []}


def load_race_control_action_log(action_log_path):
    if not action_log_path.is_file():
        return empty_race_control_action_log()
    with action_log_path.open(encoding="utf-8") as action_file:
        action_log = json.load(action_file)
    if (
        not isinstance(action_log, dict)
        or action_log.get("schema_version") != RACE_CONTROL_LOG_SCHEMA_VERSION
        or not isinstance(action_log.get("actions"), list)
    ):
        raise ValueError("Race-control actions use an unsupported file format.")
    return action_log


def validate_race_control_action(action, operator_name, action_note, review_checklist):
    if action not in RACE_CONTROL_ACTIONS:
        raise ValueError("Choose a valid race-control action.")
    if not operator_name or not operator_name.strip():
        raise ValueError("Operator name is required.")
    if not action_note or not action_note.strip():
        raise ValueError("An action note is required.")
    if len(operator_name.strip()) > 100:
        raise ValueError("Operator name must be 100 characters or fewer.")
    if len(action_note.strip()) > 2000:
        raise ValueError("Action notes must be 2,000 characters or fewer.")
    if not isinstance(review_checklist, dict) or not all(
        review_checklist.get(check) is True for check in REQUIRED_REVIEW_CHECKS
    ):
        raise ValueError(
            "Confirm the video, ID-continuity, and camera-trust checks before recording an action."
        )


def append_race_control_action(
    action_log,
    case,
    action,
    operator_name,
    action_note,
    review_checklist,
    recorded_at=None,
):
    """Append a timestamped case snapshot without changing prior history."""
    validate_race_control_action(action, operator_name, action_note, review_checklist)
    updated_log = deepcopy(action_log)
    if updated_log.get("schema_version") != RACE_CONTROL_LOG_SCHEMA_VERSION:
        raise ValueError("Race-control actions use an unsupported file format.")
    actions = updated_log.setdefault("actions", [])
    timestamp = recorded_at or datetime.now(timezone.utc).isoformat()
    actions.append(
        {
            "action_id": f"{case['case_id']}:action-{len(actions) + 1}",
            "case_id": case["case_id"],
            "track_ids": list(case["track_ids"]),
            "track_ids_label": case["track_ids_label"],
            "start_time_seconds": case["start_time_seconds"],
            "end_time_seconds": case["end_time_seconds"],
            "source_event_keys": list(case["source_event_keys"]),
            "priority_snapshot": case["priority"],
            "action": action,
            "operator_name": operator_name.strip(),
            "action_note": action_note.strip(),
            "review_checklist": {
                check: True for check in REQUIRED_REVIEW_CHECKS
            },
            "recorded_at": timestamp,
        }
    )
    return updated_log


def write_race_control_action_log(action_log, action_log_path):
    action_log_path.parent.mkdir(parents=True, exist_ok=True)
    with action_log_path.open("w", encoding="utf-8") as action_file:
        json.dump(action_log, action_file, indent=2)
    return action_log_path


def build_race_control_action_report(run_name, action_log):
    return {
        "report_type": "tracksight_race_control_actions",
        "schema_version": RACE_CONTROL_LOG_SCHEMA_VERSION,
        "disclaimer": RACE_CONTROL_LOG_DISCLAIMER,
        "run_name": run_name,
        "actions": list(action_log["actions"]),
    }
