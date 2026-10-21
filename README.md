# TrackSight Marshal

TrackSight Marshal is a local race-footage review tool. Give it a video, and it produces an annotated replay, tracked vehicle trajectories, and a dashboard that helps a human reviewer find moments worth watching again.

The point is not to replace race control. The point is to make review less manual: instead of scrubbing through an entire session, a reviewer can start with the moments where the system saw two tracked vehicles come unusually close, then inspect the supporting video and decide what happened.

## What it does

- Detects vehicles in race footage and assigns temporary track IDs while the association is reliable.
- Keeps a record of real detector observations separately from estimated tracker positions.
- Supports camera-specific race-area profiles, so cars outside the relevant track area can be excluded.
- Projects trusted vehicle positions into a local top-down track view when a camera has been calibrated.
- Saves observed close-approach evidence and short-term predicted-close-approach evidence as separate records.
- Produces short review clips, an annotated camera video, trajectories, provenance, and an optional Marshal evidence package.
- Provides a Streamlit dashboard for replay, human review, a decision ledger, evidence integrity checks, and handover.

The dashboard is deliberately run-aware. A camera-only run does not show spatial-review or incident tools that need calibration and saved candidate evidence. An evidence-rich calibrated run exposes those tools when they are actually useful.

## What it does not claim

TrackSight does not decide who was at fault, confirm contact, declare a rule breach, or replace an official. It also does not claim physical metres, km/h, telemetry, or permanent vehicle identity from video alone.

Every candidate is evidence for a human to inspect, not a verdict.

## Quick start

Create and activate a virtual environment, then install the project dependencies:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the dashboard:

```powershell
streamlit run streamlit_app.py
```

Open the address Streamlit prints in the terminal, normally `http://localhost:8501`.

From there, choose **Run analysis** to process a video, or **Review completed run** to inspect a saved result. The dashboard reads saved outputs; it does not rerun the detector when you open a review screen.

## Run from the command line

You can also process a local video directly:

```powershell
python -m src.run_video_tracking ".\data\raw\your_race_video.mp4" --model-path ".\models\your_model.pt"
```

For broad footage that includes pits, spectator areas, or access roads, add a race-area profile:

```powershell
python -m src.run_video_tracking ".\data\raw\your_race_video.mp4" `
  --model-path ".\models\your_model.pt" `
  --regions-config ".\configs\your_race_video.json"
```

Profiles are tied to a specific camera view. TrackSight does not apply one camera's profile to another video by default.

## Calibrate a camera view

Calibration is optional. Without it, TrackSight can still produce an annotated camera video and trajectories. With it, it can create a local top-down view and use spatial evidence safeguards.

```powershell
python -m scripts.calibrate_track ".\data\raw\your_race_video.mp4"
```

Click four points around a flat section of track in clockwise order. The tool saves a camera-specific calibration file under `configs/`. The resulting map uses local track units, not metres, unless a real-world scale reference is supplied.

For moving broadcast footage, TrackSight uses stable background features to align frames to the calibration reference. If that alignment is not reliable, the system withholds spatial safety analysis rather than pretending that an uncalibrated measurement is precise.

## What you get after a run

For a video called `your_race_video.mp4`, saved outputs appear under `outputs/tracking/`:

- `annotated_your_race_video.mp4` — the detected vehicles and their current track IDs.
- `annotated_your_race_video_trajectories.json` — real observed positions and associated tracking information.
- `annotated_your_race_video_run_summary.json` — the run's settings, coverage, and continuity information.
- `annotated_your_race_video_review_evidence.json` — observed close-approach records, when spatial review is available.
- `annotated_your_race_video_risk_evidence.json` — predicted close-approach records, when enough measured motion is available.
- Review clips, a top-down video, a Marshal briefing, and an evidence package when that run creates them.

The exact files depend on the video and whether it has a usable calibration and saved evidence. Missing spatial outputs do not mean the video failed; they can simply mean that the run was intentionally camera-only.

## Dashboard walkthrough

For a typical calibrated evidence run, the best demo path is:

1. **Overview** — see what was processed and what was withheld.
2. **Camera video** — inspect the annotated source footage.
3. **Evidence queue** — choose an observed or predicted close-approach record.
4. **Incident replay** or **Safety command center** — inspect the saved evidence, supporting replay, and review ladder for one case.
5. **Human review decisions** — record a human outcome without changing the underlying evidence.
6. **Decision ledger** and **Evidence integrity** — show the audit trail and the generated files behind it.

Other views are available when their inputs exist. For example, telemetry correlation appears for a run with evidence because it accepts an optional reviewer-supplied CSV; safety zones appear only when that run has zone evidence.

## Verify a completed run

Before a demo or handover, verify that the recorded video and its JSON outputs agree:

```powershell
python -m scripts.verify_completed_run ".\outputs\tracking\annotated_your_race_video.mp4"
```

This is a read-only preflight check. It does not change the video or any evidence file.

## Development checks

Run the test suite from the project root:

```powershell
python -m unittest discover -s tests
```

The tests cover the detection/tracking pipeline, calibration safeguards, saved evidence formats, dashboard workflows, and audit records. They prove the project logic behaves as expected for its test cases; they do not turn a video-only system into a guarantee for every camera, weather condition, occlusion, or race series.

## Limitations worth knowing

- Vehicle detection and ID continuity can degrade with heavy occlusion, camera cuts, blur, poor lighting, or dense packs.
- A long occlusion or edit can create a new ID segment. TrackSight records segments rather than claiming a permanent identity it cannot prove.
- Calibration is camera-specific. A profile from one viewpoint is not valid for a different viewpoint.
- Spatial evidence is withheld when the saved camera conditions do not pass the project's safeguards.
- Candidate events are prompts for human review. They are not automatic incident decisions.

## Project notes

The dashboard and all processing run locally. Optional evidence files store the records created for one run; reviewer decisions, corrections, and second opinions are kept separate from the original detector evidence so the source record remains auditable.

For hackathon narration and suggested judge questions, see [docs/hackathon-submission.md](docs/hackathon-submission.md). Demonstration-media attribution is in [docs/media-attribution.md](docs/media-attribution.md).

## License

No public source-code licence has been selected yet. Demo footage and datasets have their own attribution and licence terms.
