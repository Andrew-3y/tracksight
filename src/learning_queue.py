"""Export difficult, reviewable evidence for a human labelling workflow.

This creates annotation *tasks*, never ground truth. Only a human labeller may
turn the exported frame/clip references into approved training labels.
"""

from pathlib import Path


LEARNING_QUEUE_DISCLAIMER = (
    "This export is a human-labelling task queue, not ground truth. Model boxes, "
    "IDs, and review outcomes must be independently checked before training or evaluation."
)


def _portable_clip_name(value):
    """Return a filename only for a real saved clip path, never a dataframe NaN."""
    return Path(value).name if isinstance(value, str) and value.strip() else None


def build_learning_queue(timeline_rows, safety_candidates=None):
    """Build stable annotation tasks from saved evidence and policy prompts."""
    tasks = []
    for row in timeline_rows:
        if row.get("review_status") == "Cleared":
            continue
        tasks.append({
            "task_id": f"evidence:{row['event_key']}",
            "task_type": "evidence_event",
            "labeling_status": "needs_human_annotation",
            "evidence_type": row["evidence_type"],
            "track_ids": list(row["track_id_values"]),
            "frame": int(row["frame"]),
            "time_seconds": float(row["time_seconds"]),
            "why_selected": (
                f"Saved {row['evidence_type'].lower()} evidence with "
                f"review status {row.get('review_status', 'Unreviewed')}."
            ),
            "supporting_clip": _portable_clip_name(row.get("clip")),
        })
    for candidate in safety_candidates or []:
        tasks.append({
            "task_id": f"policy:{candidate['candidate_type']}:{candidate['anchor_frame']}:{candidate['track_id']}",
            "task_type": "policy_candidate",
            "labeling_status": "needs_human_annotation",
            "evidence_type": candidate["candidate_type"],
            "track_ids": list(candidate["track_ids"]),
            "frame": int(candidate["anchor_frame"]),
            "time_seconds": float(candidate["anchor_time_seconds"]),
            "why_selected": candidate["review_reason"],
            "supporting_clip": (
                _portable_clip_name(candidate.get("marshal_briefing_clip_path"))
            ),
        })
    return sorted(tasks, key=lambda task: (task["time_seconds"], task["task_id"]))


def build_learning_export(run_name, tasks):
    """Create a portable manifest suitable for a labelling-team handoff."""
    return {
        "export_type": "tracksight_human_labelling_queue",
        "disclaimer": LEARNING_QUEUE_DISCLAIMER,
        "run_name": run_name,
        "task_count": len(tasks),
        "tasks": tasks,
    }
