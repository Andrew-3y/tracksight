"""Create a portable, privacy-conscious evidence handoff for a human reviewer."""

import json
from copy import deepcopy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PACKAGE_TYPE = "tracksight_marshal_evidence_package"
PACKAGE_SCHEMA_VERSION = 1
PACKAGE_README = """TrackSight Marshal evidence package

This package contains a human-review briefing, source evidence records, saved
trajectory observations, and the annotated clips supporting every briefing
highlight. It does not determine fault, contact, a rule violation, or an
incident outcome. Review the clips and full available race footage before any
decision.

Clip references in marshal_briefing.json are relative to this ZIP file under
briefing_clips/. Local computer paths have been removed so this package can be
shared without exposing the producer's directory structure.
"""


def make_portable_path(path_value):
    """Replace an absolute local path with the file name safe to share."""
    return None if path_value is None else Path(path_value).name


def make_portable_briefing(briefing):
    """Copy a briefing and rewrite its clip/source references for the ZIP."""
    portable_briefing = deepcopy(briefing)
    run = portable_briefing.get("run", {})
    run["source_video"] = make_portable_path(run.get("source_video"))
    for highlight_type in (
        "predicted_risk_highlights",
        "observed_proximity_highlights",
        "safety_candidate_highlights",
    ):
        for highlight in portable_briefing.get(highlight_type, []):
            clip_path = highlight.get("supporting_clip")
            if clip_path:
                highlight["supporting_clip"] = (
                    f"briefing_clips/{Path(clip_path).name}"
                )
    return portable_briefing


def make_portable_summary(run_summary):
    """Copy run metadata while removing producer-machine directory paths."""
    portable_summary = deepcopy(run_summary)
    provenance = portable_summary.get("run_provenance")
    if not isinstance(provenance, dict):
        return portable_summary
    for path_key in (
        "source_video",
        "detector_model",
        "regions_profile",
        "calibration_profile",
        "safety_zone_profile",
    ):
        provenance[path_key] = make_portable_path(provenance.get(path_key))
    return portable_summary


def make_portable_evidence(evidence):
    """Remove local-only clip references from raw evidence records."""
    portable_evidence = deepcopy(evidence)
    for event in portable_evidence:
        event.pop("review_clip_path", None)
        event.pop("marshal_briefing_clip_path", None)
    return portable_evidence


def make_portable_safety_candidate_evidence(safety_candidate_evidence):
    """Keep candidate settings while removing machine-local clip paths."""
    portable_evidence = deepcopy(safety_candidate_evidence)
    for candidate_type in (
        "stationary_vehicle_candidates",
        "restricted_area_entry_candidates",
    ):
        for candidate in portable_evidence.get(candidate_type, []):
            candidate.pop("review_clip_path", None)
            candidate.pop("marshal_briefing_clip_path", None)
    return portable_evidence


def get_briefing_clip_paths(briefing):
    """Return all mandatory supporting clips and reject an incomplete briefing."""
    clip_paths = []
    for highlight_type in (
        "predicted_risk_highlights",
        "observed_proximity_highlights",
        "safety_candidate_highlights",
    ):
        for highlight in briefing.get(highlight_type, []):
            clip_path = highlight.get("supporting_clip")
            if not isinstance(clip_path, str) or not Path(clip_path).is_file():
                raise ValueError(
                    "Each Marshal briefing highlight must have a saved supporting clip."
                )
            clip_paths.append(Path(clip_path))
    return clip_paths


def create_marshal_evidence_package(
    output_path,
    briefing,
    run_summary,
    risk_evidence,
    review_evidence,
    trajectories,
    safety_zone_evidence=None,
    safety_candidate_evidence=None,
):
    """Write a self-contained ZIP handoff from already-saved TrackSight output."""
    clip_paths = get_briefing_clip_paths(briefing)
    if len({clip_path.name for clip_path in clip_paths}) != len(clip_paths):
        raise ValueError("Marshal briefing clips must have distinct file names.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    portable_briefing = make_portable_briefing(briefing)
    manifest = {
        "package_type": PACKAGE_TYPE,
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "briefing_highlight_count": len(clip_paths),
        "contents": [
            "README.txt",
            "manifest.json",
            "marshal_briefing.json",
            "run_summary.json",
            "predicted_risk_evidence.json",
            "observed_proximity_evidence.json",
            "trajectories.json",
            "briefing_clips/",
        ],
    }
    if safety_zone_evidence is not None:
        manifest["contents"].append("safety_zone_evidence.json")
    if safety_candidate_evidence is not None:
        manifest["contents"].append("safety_candidate_evidence.json")
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("README.txt", PACKAGE_README)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        archive.writestr(
            "marshal_briefing.json", json.dumps(portable_briefing, indent=2)
        )
        archive.writestr(
            "run_summary.json",
            json.dumps(make_portable_summary(run_summary), indent=2),
        )
        archive.writestr(
            "predicted_risk_evidence.json",
            json.dumps(make_portable_evidence(risk_evidence), indent=2),
        )
        archive.writestr(
            "observed_proximity_evidence.json",
            json.dumps(make_portable_evidence(review_evidence), indent=2),
        )
        if safety_candidate_evidence is not None:
            archive.writestr(
                "safety_candidate_evidence.json",
                json.dumps(
                    make_portable_safety_candidate_evidence(
                        safety_candidate_evidence
                    ),
                    indent=2,
                ),
            )
        archive.writestr("trajectories.json", json.dumps(trajectories, indent=2))
        if safety_zone_evidence is not None:
            archive.writestr(
                "safety_zone_evidence.json",
                json.dumps(safety_zone_evidence, indent=2),
            )
        for clip_path in clip_paths:
            archive.write(clip_path, f"briefing_clips/{clip_path.name}")
    return output_path
