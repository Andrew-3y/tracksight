# TrackSight Marshal — hackathon submission guide

## One-sentence pitch

TrackSight Marshal turns race video into an auditable safety-review workspace: it detects and tracks vehicles, maps confirmed positions onto a calibrated local track view, and presents candidate close-approach evidence for a human to review.

## The problem

Race review often starts with long, dense video footage. A reviewer must find the relevant vehicles, understand their movement, and decide whether a close interaction deserves attention. A conventional detector only says “there is a car in this frame”; it does not preserve an evidence trail across time.

## What TrackSight contributes

```text
race video
  → vehicle detections
  → confirmed tracker-ID segments
  → calibrated local-track trajectories
  → measured motion
  → observed / predicted proximity candidates
  → evidence clips + dashboard for human review
```

The output is intentionally **review assistance**, not an automatic ruling. TrackSight does not claim that a candidate is a crash, a rules breach, fault, measured vehicle telemetry, metres, or km/h.

## Two-minute live demo

### 0–20 seconds — establish the problem

“Race footage can show many vehicles at once. A human reviewer should not have to search every frame manually before they can investigate a close interaction.”

### 20–45 seconds — show the camera video

Open **Camera video**. Explain that a green box and ID appear only after the tracker has enough continuous real detections to confirm the vehicle. This avoids immediately treating one uncertain detection as a trustworthy ID.

### 45–70 seconds — show the spatial digital twin

Open **Spatial map**. Explain that the map uses the bottom-centre “footpoint” of a vehicle box. A four-point calibration maps that image point into a local track coordinate system. For moving broadcast footage, TrackSight aligns the current frame to the calibration reference using stable background features before projecting the footpoint.

The map uses **local track units**, not metres. It is useful for comparing movement in one calibrated view, but it is not surveyed telemetry.

### 70–100 seconds — show the evidence queue

Open **Evidence queue**. Start with **Predicted risk**: TrackSight projects two vehicles’ measured motion over a short horizon and flags persistent, small predicted separations. Switch to **Observed proximity**: this is a separate list of pairs that were actually close for consecutive frames.

The lists are separate on purpose. Predicted separation is normalized by projected vehicle width; observed separation is normalized by detected-box diagonal. They are both useful, but they are not the same unit and should not be ranked as though they were.

### 100–110 seconds — show the Marshal briefing

Open **Marshal briefing**. It converts the same saved evidence into a concise handoff for a human reviewer, retaining the evidence type, track IDs, time, measurement basis, supporting clip, and recommended human action. It does not add a new model judgement or decide an incident.

### 110–120 seconds — show auditability

Open **Provenance**. Point out the exact source video, detector model, race-area profile, camera mode, calibration profile, and thresholds. This makes a result reproducible and lets a reviewer understand how it was generated.

## Key implementation choices

| Choice | Why it exists |
| --- | --- |
| Confirmed detections only | Estimated tracker positions are not saved as observations or shown on the map, preventing guesses from becoming evidence. |
| Region profile | Excludes pits, spectators, parked vehicles, and other non-racing areas when a camera includes them. |
| Calibration | Corrects perspective so that movements can be compared in a local track view rather than raw image pixels. |
| Camera stabilization | For a moving camera, background features estimate camera motion so each frame can be aligned to the calibration reference. |
| Track-quality gate | Only sufficiently continuous, real-observation segments may create safety evidence. |
| Separate evidence categories | Observed proximity and predicted proximity have different meanings and normalization, so the dashboard keeps them separate. |
| Evidence clips | A score alone is not a decision; each shortlisted event is paired with a short annotated clip for review. |
| Marshal briefing | Converts saved evidence into a concise, reproducible handoff without turning a candidate into an automated ruling. |

## Likely judge questions

### Why is this better than ordinary object detection?

Object detection is frame-by-frame. TrackSight adds confirmation, identity association, trajectories, motion, calibration, evidence thresholds, saved clips, and provenance. That turns detections into a review workflow rather than a box-drawing demo.

### Does “candidate risk” mean a crash or a rule violation happened?

No. It means the model found a persistent observed or predicted close approach under the chosen settings. A human must inspect the evidence clip and make the decision.

### Why can a top-down marker disappear?

The map displays only real, confirmed detector observations. If a vehicle is briefly occluded or not detected, the tracker may retain an internal estimated location, but TrackSight does not draw that guess as spatial evidence. This makes the map more honest.

### Does it work only with this TCR clip?

No. The runner accepts any video path. It can use the whole image by default or an optional normalized race-area profile. A matching calibration profile enables local spatial analysis for that camera view. Detection quality and ID continuity still depend on footage quality, occlusion, camera angle, and the detector model.

### Does the moving-camera support create real GPS coordinates?

No. It aligns frames to the camera view used for calibration. The resulting values remain local track units until the user provides a real-world scale reference.

### How did you verify the project?

The repository includes automated tests for camera-reference alignment, regions, tracking quality gates, data validation, dashboard compatibility, and completed-run verification. The final demonstration output is also preflight-checked to ensure the video, trajectories, evidence, and run summary agree.

## Honest limitations

- A video-only system cannot promise permanent identity through a long occlusion, edit, or camera cut.
- Detection and tracking depend on the model and footage conditions.
- Calibration is camera-view-specific; local coordinates are not metres without a scale reference.
- Moving-camera stabilization relies on visible stable background detail and should be treated cautiously through sharp cuts or major zoom changes.
- TrackSight prioritizes traceable evidence for review; it does not replace race officials.

## Demonstration checklist

- [ ] Activate the virtual environment.
- [ ] Generate or select the final completed run.
- [ ] Run `python -m scripts.verify_completed_run` on the annotated final MP4.
- [ ] Start the Streamlit dashboard.
- [ ] Select the final run under **Completed run**.
- [ ] Demonstrate Camera video → Spatial map → Evidence queue → Marshal briefing → Provenance.
- [ ] Keep the media-attribution document with the submission.
- [ ] Choose a source-code licence before publishing the repository publicly.
