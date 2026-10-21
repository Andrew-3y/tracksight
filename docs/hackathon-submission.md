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
  → observed / predicted proximity + low-displacement + reviewer-defined restricted-area candidates
  → evidence clips + dashboard for human review
  → incident threads + optional safety-zone observations + portable case packet
  → confidence-gated race-control queue + append-only action log
```

The output is intentionally **review assistance**, not an automatic ruling. TrackSight does not claim that a candidate is a crash, a rules breach, fault, measured vehicle telemetry, metres, or km/h.

## Two-minute live demo

### 0–15 seconds — establish the problem

“Race footage can show many vehicles at once. A human reviewer should not have to search every frame manually before they can investigate a close interaction.”

### 15–30 seconds — launch an auditable run

Open **Run analysis**. Choose the prepared TCR source video. Explain that the matching race-area and calibration profiles are selected automatically only because their filenames match this exact camera source. For a time-bounded demo, use the already completed run next; explain that this page starts the same local pipeline as the terminal command, verifies the saved result, then opens it for review.

### 30–50 seconds — show the camera video

Open **Camera video**. Explain that a green box and ID appear only after the tracker has enough continuous real detections to confirm the vehicle. This avoids immediately treating one uncertain detection as a trustworthy ID.

### 50–70 seconds — show the spatial digital twin

Open **Spatial map**. Explain that the map uses the bottom-centre “footpoint” of a vehicle box. A four-point calibration maps that image point into a local track coordinate system. For moving broadcast footage, TrackSight aligns the current frame to the calibration reference using stable background features before projecting the footpoint.

The map uses **local track units**, not metres. It is useful for comparing movement in one calibrated view, but it is not surveyed telemetry.

### 70–95 seconds — show the evidence queue

Open **Evidence queue**. Start with **Predicted risk**: TrackSight projects two vehicles’ measured motion over a short horizon and flags persistent, small predicted separations. Switch to **Observed proximity**: this is a separate list of pairs that were actually close for consecutive frames.

The lists are separate on purpose. Predicted separation is normalized by projected vehicle width; observed separation is normalized by detected-box diagonal. They are both useful, but they are not the same unit and should not be ranked as though they were.

### 95–107 seconds — replay the evidence in context

Open **Incident replay**. Select a candidate and first show the approach reconstruction: its saved start, anchor, and end frames, the number of paired real calibrated observations, any omitted pair frames, and the measured net separation trend. Then show the two selected vehicles' actual observed paths and the exact-frame traffic snapshot: gold is the selected pair and gray is surrounding real projected traffic. Finally show the annotated camera and calibrated local-track videos, which begin at the same source-video time. Explain that a missing detector observation becomes a visible gap—not an invented path—and that the replay is evidence for a human decision, not an automatic ruling.

Open **Evidence quality**. Point out that it reports only frames and ID segments eligible for safety evidence, identifies excluded short or discontinuous segments, and never presents coverage as an accuracy score. This shows the system knows when it lacks enough evidence to make a safety claim.

Open **Candidate hotspots**. Explain that each small point is a saved candidate event placed at the midpoint of the two vehicles' calibrated positions. Larger numbered rings group repeated same-type events at a local location, using a neighbourhood measured in the current run's median projected vehicle widths. This prioritises review; it does not label a corner dangerous or create a new risk score.

Open **Incident threads**. Explain that TrackSight groups nearby saved records only if they concern the same two vehicle IDs; it caps a case at six seconds so repeated evidence cannot become one misleading “incident.” Select a case to show the source records and supporting clip, then use **Download incident packet**. The packet carries the selected evidence, any current matching human decisions, available clips, optional zone context, and SHA-256 hashes—without local machine paths.

Open **Safety zones** if the run used a zone profile. Explain that zones are named polygons defined by the reviewer in this camera's calibrated local-track coordinates, such as a pit-exit merge or marshal post. The system records only real, trusted vehicle observations inside the polygon; it does not determine an infringement or unsafe driving. If there was no per-camera profile, show that TrackSight correctly withholds the view rather than inventing circuit geometry.

Open **Safety candidates**. Explain that a low-displacement candidate is measured in projected vehicle widths rather than an arbitrary pixel or circuit-specific speed threshold, while a restricted-area candidate requires an explicitly reviewer-defined zone policy. Each card shows the saved evidence basis, calibration and ID-continuity safeguards, a supporting replay when saved, and the statement that human review is still required.

Open **Marshal handover**. Explain that it does not create a new score or automated decision: it gives the next human reviewer the session duration, unresolved and cleared evidence counts, three next review items, calibration/continuity limits, and optional track-policy prompts. Download the handover if the judges want to see its portable, path-free JSON format.

Open **Evaluation and limits**. Explain that this is intentionally not a made-up accuracy scoreboard. It shows only the completed run's known context, outputs, and evidence gates, then makes the missing ground-truth benchmark explicit. This demonstrates that TrackSight is designed to be useful and auditable without overstating what a video-only model can prove.

Open **Learning queue**. Explain that TrackSight converts unresolved evidence into portable human-labelling tasks, rather than pretending that its own boxes or IDs are ground truth. This is the project’s improvement loop: review creates high-value annotation work, independently labelled data enables a future benchmark and retraining, and the system’s limitations remain visible throughout.

Open **Safety command center**. Choose a case and show the decision ladder. Explain that the status is not an AI verdict: it is a compact explanation of whether calibration, ID continuity, saved time-local evidence, replay availability, and existing human review support monitoring, immediate review, or withholding. Open the replay and **Evidence correlation** table, then download the after-action report. This shows the full operational loop from candidate evidence to an auditable human handoff.

Open **Reviewer corrections**. Record a correction such as `Merged vehicles` or `Incorrect ID` and explain that TrackSight saves it separately from the immutable evidence. The exported correction handoff is not a label file: a human must still independently annotate the referenced frames before training or measuring a model.

Open **Decision ledger**. Explain that it assembles the versioned human-review record, append-only race-control actions, reviewer corrections, and independent second opinions into one chronological audit trail. It does not change source evidence or convert a human record into an automated verdict. Then open **Independent second review**: a different named reviewer can agree, disagree, or request more evidence, but cannot overwrite the primary human decision.

Open **Evidence integrity**. Show that the dashboard first checks whether each saved output exists. Press **Calculate saved-output hashes** only if the judges ask for an integrity manifest; explain that it produces point-in-time SHA-256 checksums of outputs, not a claim that the source footage is authentic or the detector is accurate. Open **Operational coverage** to show real detector positions, continuity-qualified positions, and frames where spatial review is withheld—without calling this an accuracy score.

Open **Grounded case assistant**. Select one of the fixed evidence questions. Explain that it has no free-form generation or hidden model inference: it answers strictly from the selected case, decision ladder, evidence-source table, and applicable human records.

Open **Telemetry correlation** only when an aligned CSV is available. Explain that the CSV must share the video clock, that rows outside the visible half-second alignment window are withheld, and that aligned brake/throttle/speed values are supplementary evidence rather than proof of a driver’s action. Open **Multi-camera review** only when the supplied manifest and video files have manually verified time offsets; it is a reviewer-controlled context view, not automatic camera fusion.

Use **Race-control export** to download the full chronological evidence timeline and human-review statuses without paths from the producing computer.

### 107–115 seconds — record a human decision

Open **Human review decisions**. Select an evidence event, inspect its supporting clip, then explain that a race reviewer—not TrackSight—records `Needs review`, `Cleared`, or `Escalate` with their name and an audit note. The system stores this separately from the raw model evidence and does not reuse a decision if the evidence changes after reprocessing.

### 115–120 seconds — show the Marshal briefing and auditability

Open **Marshal briefing**. It converts the same saved evidence into a concise handoff for a human reviewer, retaining the evidence type, track IDs, time, measurement basis, supporting clip, and recommended human action. It does not add a new model judgement or decide an incident.

Use **Download Marshal evidence package** to demonstrate that the same review package can be handed to another reviewer without exposing paths from the producing computer.

Open **Race-control timeline** to show the completed video as one reviewable sequence. Filter the saved evidence by type, involved vehicle, or time range; choose an exact labelled event to open its supporting clip. Predicted and observed evidence deliberately remain in separate lanes because their measurements have different valid normalizations.

Open **Race-control desk**. Explain that TrackSight never converts a raw measurement into an automated severity score. It first requires a calibrated view and continuity-qualified IDs; repeated nearby evidence for the exact same pair is prioritised for review. A human can append an accountable action without modifying the underlying evidence, then download a one-page review report.

Open **Camera trust monitor**. Show the four safeguards: camera calibration, camera mode, ID continuity, and moving-camera stabilization. Explain that new runs retain aggregate stabilization outcomes; if the camera lacks enough background detail, TrackSight keeps the last reliable transform instead of inventing a correction. Older runs visibly state that this new diagnostic was not saved.

Open **Live race-control simulation**. Move the replay clock to demonstrate that a candidate does not appear until its saved source-video time. The view is explicitly a replay of completed data, not a claim that TrackSight processed the broadcast live.

Open **Session comparison** only for two runs from the same recorded calibration. Explain that TrackSight refuses incompatible cameras rather than overlaying unrelated map coordinates. The remaining map shows recurring candidate-evidence locations as human-review prompts, not claims about a dangerous corner or driver behaviour.

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
| Decision-ready incident replay | Brings the exact saved reason, a traceable start-anchor-end approach reconstruction, two involved vehicles' observed movement, exact-frame traffic context, time-aligned camera/local-track replay, and current human status into one review screen without creating a new model conclusion. |
| Evidence quality and coverage | Shows which real observations were eligible for safety analysis and what was withheld, without mistaking a coverage ratio for detector accuracy. |
| Candidate-risk hotspots | Groups repeated same-type, calibrated candidate events by location using an adaptive vehicle-width neighbourhood, while withholding the view when calibration or trusted evidence is absent. |
| Human review decisions | Records accountable reviewer decisions and notes separately from model evidence; changed evidence cannot silently inherit an old decision. |
| Incident threads | Groups nearby, vehicle-linked records into a time-bounded review case while preserving distinct observed and predicted measurements. |
| Safety zones | Uses only reviewer-defined, camera-specific local-track polygons and real trusted observations; it never infers a rule breach. |
| Safety candidates | Adds calibrated low-displacement prompts and explicit restricted-area-entry prompts while preserving a human-review requirement and never inferring a stopped car or rule breach. |
| Track policy engine | Supports optional, reviewer-configured local-track prompts for restricted areas, low displacement on an active racing area, and motion opposed to a configured corridor direction—without inferring rules or direction from the video. |
| Incident packet | Exports a portable selected-case handoff with supporting evidence, available clips, optional zone context, and SHA-256 integrity hashes. |
| Marshal briefing | Converts saved evidence into a concise, reproducible handoff without turning a candidate into an automated ruling. |
| Marshal handover | Presents session status, unresolved next-review items, cleared/escalated counts, coverage, limitations, and optional policy prompts for the next human reviewer. |
| Evaluation and limits | Shows the real scope, recorded outputs, safeguards, and limits of a completed run without inventing accuracy metrics. |
| Learning queue | Converts unresolved evidence into portable human-labelling tasks without mislabelling model output as ground truth. |
| Safety command center | Combines one selected review case, a transparent decision ladder, replay, evidence-source availability, and a portable after-action report without making an automated incident ruling. |
| Reviewer corrections | Stores append-only human detection/ID feedback for independent annotation, rather than treating feedback as immediate ground-truth labels. |
| Decision ledger | Brings versioned human decisions, append-only actions, corrections, and independent opinions into one chronological audit record without changing source evidence. |
| Independent second review | Requires a different named reviewer to append an agree, disagree, or request-more-evidence opinion; it never overwrites the primary human decision. |
| Evidence integrity | Lists generated output availability and creates SHA-256 checksums only on explicit reviewer request, without implying source-footage authenticity. |
| Operational coverage | Shows recorded detector positions and continuity eligibility across video frames while avoiding a misleading accuracy or safety-confidence score. |
| Grounded case assistant | Provides fixed evidence questions whose answers are composed only from the selected saved case and applicable human records, not a generative incident verdict. |
| Telemetry correlation | Aligns optional user-supplied, video-clock telemetry within a visible time window and withholds unmatched samples. |
| Multi-camera review | Shows supplementary camera context only from reviewer-supplied video files and verified time offsets; no automatic fusion is claimed. |
| Race-control export | Exports a path-free chronological evidence and human-review timeline for handover. |
| Network-source input | Accepts an RTSP/HTTP(S) source only when an explicit output MP4 is supplied; it records the same reviewable evidence pipeline. |
| Race-control desk | Uses quality gates and repeated exact-pair evidence to prioritise human review, records append-only human actions, and exports a path-free report. |
| Session comparison | Refuses mismatched calibrations and compares only compatible local-track evidence across completed runs. |
| Recurring review map | Groups repeated candidate-evidence locations across completed runs only when their local-track calibrations match. |
| Camera trust monitor | Makes calibration, ID-continuity, and saved moving-camera stabilization safeguards visible; missing diagnostics stay visible rather than inferred. |
| Live race-control simulation | Replays completed evidence in source-video time to demonstrate a live-review workflow without claiming live inference or telemetry. |

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
- [ ] Start the dashboard and open **Run analysis**.
- [ ] Generate or select the final completed run.
- [ ] Run `python -m scripts.verify_completed_run` on the annotated final MP4.
- [ ] Select the final run under **Completed run**.
- [ ] Demonstrate Camera video → Spatial map → Evidence queue → Incident replay → Evidence quality → Camera trust monitor → Candidate hotspots → Incident threads → Live race-control simulation → Race-control desk → Safety command center → Decision ledger → Independent second review → Evidence integrity → Operational coverage → Grounded case assistant → Session comparison → Safety zones → Safety candidates → Human review decisions → Marshal briefing → Marshal handover → Evaluation and limits → Provenance.
- [ ] Keep the media-attribution document with the submission.
- [ ] Choose a source-code licence before publishing the repository publicly.
