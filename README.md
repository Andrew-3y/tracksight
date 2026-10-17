# TrackSight Marshal

TrackSight Marshal is a computer-vision system for reviewing race footage. It turns vehicle observations into auditable spatial evidence for a human reviewer. It supports a calibrated fixed camera directly, or a calibrated moving camera by first aligning each frame to the calibration reference from stable background features.

```text
video → detections → tracking → calibrated trajectories → motion → candidate risk → evidence clips + dashboard
```

It is designed to reduce the amount of footage a human reviewer needs to inspect. It does **not** declare that a close approach is a crash, a rule violation, or proof of fault.

## Hackathon demo in one minute

TrackSight Marshal takes race footage and produces a reviewable chain of evidence:

1. **Annotated camera video** shows the vehicles that were detected and assigned a temporary on-screen ID.
2. **Spatial digital twin** maps confirmed vehicle footpoints into a calibrated local track view, including a moving-camera stabilization step when needed.
3. **Evidence queue** separates observed proximity from predicted close approaches and opens short supporting clips for a human reviewer.
4. **Provenance** records the input video, model, calibration, and settings that produced the result.

For the included TCR demonstration clip, process the video and then open the dashboard:

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.run_video_tracking ".\data\raw\tcr_world_tour_el_pinar_2023_clean_pack.mp4" --output-path ".\outputs\tracking\tracksight_marshal_tcr_clean_pack_final.mp4"
streamlit run streamlit_app.py
```

In the dashboard, choose `tracksight_marshal_tcr_clean_pack_final` under **Completed run**. For a suggested narration and likely judge questions, see [the hackathon guide](docs/hackathon-submission.md).

## What it produces

- An annotated MP4 showing detected vehicles and motion-compensated BoT-SORT IDs while they remain reliably associated on screen.
- A perspective-corrected top-down MP4 when the camera has a usable calibration profile.
- A trajectory JSON file containing each confirmed vehicle's observed positions, confidence, calibrated track position, and motion status.
- A review-evidence JSON file listing sustained close approaches between two tracked vehicles.
- A predicted-risk JSON file listing pairs that are projected to reach a small separation during a short future horizon.
- A run-summary JSON file that distinguishes all saved ID segments from the segments trusted for safety analysis.
- Up to ten short annotated review clips around the closest, non-overlapping evidence moments.
- Up to ten short annotated clips around the strongest candidate-risk moments.
- A local Streamlit dashboard for reviewing all saved evidence from one completed run.

## Install

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run on a video

Place a video anywhere on your computer and provide its path. You also need a compatible YOLO model file.

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.run_video_tracking ".\data\raw\your_race_video.mp4" --model-path ".\models\your_model.pt"
```

If no region profile exists, TrackSight uses the full frame automatically. For footage that includes pits, spectators, or other irrelevant areas, provide an optional profile with `--regions-config` to restrict detection to race areas.

## Calibrate a camera view

For spatial analysis, create one calibration profile for each camera view:

```powershell
.\.venv\Scripts\Activate.ps1
python -m scripts.calibrate_track ".\data\raw\your_race_video.mp4"
```

The tool asks you to click four corners of a flat road corridor in clockwise order. It saves `configs/<video-name>_calibration.json` and a top-down preview. When its name matches the video name, the runner loads it automatically.

Calibration removes much of the camera's perspective distortion. For a moving view, TrackSight additionally estimates the view's movement from background features and aligns every frame to the calibration reference before projecting vehicle positions. The resulting coordinates are **local track units**, not metres, until a real-world scale reference is supplied.

## Outputs

For an input video named `your_race_video.mp4`, TrackSight writes to `outputs/tracking/`:

- `annotated_your_race_video.mp4`
- `annotated_your_race_video_top_down.mp4` when calibration is available
- `annotated_your_race_video_trajectories.json`
- `annotated_your_race_video_review_evidence.json`
- `annotated_your_race_video_review_clips/`
- `annotated_your_race_video_risk_evidence.json`
- `annotated_your_race_video_risk_review_clips/`
- `annotated_your_race_video_run_summary.json`

## Review dashboard

After processing a video, open the local dashboard:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

The dashboard reads saved output files only. It never reruns detection when you open a video, map, trajectory chart, or evidence clip.

## Submission preflight

Before demonstrating or submitting a completed run, verify that the annotated
MP4 can be decoded and that its trajectory, evidence, and run-summary files
agree with one another. This check is read-only: it never changes the video or
its evidence.

```powershell
.\.venv\Scripts\Activate.ps1
python -m scripts.verify_completed_run ".\outputs\tracking\annotated_your_race_video.mp4"
```

For a calibrated moving broadcast camera, a successful preflight should report
that spatial analysis is enabled with camera stabilization. The annotated camera
video, local map, and candidate evidence remain review artifacts: cuts, heavy
zoom changes, weak background detail, or inaccurate calibration can still make
the spatial result unreliable.

## Training-data quality gates

TrackSight uses one final class, `vehicle`. A source dataset may use many
classes, however, so only source class IDs that actually mean road or race
vehicles may be imported. The importer requires those IDs explicitly rather
than converting every labelled object into a vehicle.

For example, if a downloaded source defines race cars as source class `4`, its
import command must include `--vehicle-class-ids 4`. If it defines road cars
and trucks as `2` and `7`, use `--vehicle-class-ids 2 7`. The numbers come from
the source dataset's class list; they are not TrackSight vehicle IDs.

Before training, TrackSight validates that every YOLO box is numeric, belongs
to the correct class, has a non-zero size, and stays completely within the
image. It also has two read-only review tools:

```powershell
.\.venv\Scripts\Activate.ps1
python -m scripts.review_vehicle_labels "<images-folder>" "<labels-folder>" "<empty-review-output-folder>"
python -m scripts.audit_vehicle_labels "<images-folder>" "<labels-folder>" "<new-audit-report.json>"
```

The first command renders boxes on copied images so a person can inspect them.
The second reports near-identical boxes, which are possible accidental duplicate
labels. It does not delete or change any source label: overlap alone cannot tell
whether the labels are two cars in an occlusion or an error.

## How close-approach evidence works

For every frame, TrackSight compares the centres of vehicles from trusted track segments. A pair is considered close only when its centre distance is within a configurable number of average vehicle-box diagonals, and it remains close for multiple consecutive frames. This normalizes the measurement for different video resolutions and apparent vehicle sizes.

The defaults are intentionally review-oriented:

- `--close-approach-ratio 1.0`
- `--minimum-close-frames 3`
- `--review-pre-seconds 2.0`
- `--review-post-seconds 2.0`
- `--max-review-clips 10`

These values can be changed at the command line for a specific review policy.

## How predicted-risk evidence works

For pairs with calibrated positions and measured motion, TrackSight compares their relative position and relative velocity. It projects both vehicles forward for a short horizon, finds the moment of minimum predicted separation, and normalizes that separation by the pair's projected vehicle widths.

The defaults are deliberately conservative and review-oriented:

- `--prediction-horizon-seconds 1.5`
- `--maximum-predicted-separation-vehicle-widths 1.5`
- `--minimum-risk-frames 18`
- `--maximum-motion-gap-seconds 0.25`
- `--minimum-trusted-track-frames 30`

TrackSight does not calculate a risk prediction if either vehicle has no calibration, does not have a continuous history of real observations, or has a detection gap longer than the configured maximum. This prevents the system from presenting an invented velocity as measured evidence.

## Important limitations

Vehicle detection and identity continuity depend on the model, camera angle, occlusion, lighting, and the quality of the footage. No video-only tracker can honestly guarantee a permanent identity after a long occlusion, an edit, or a camera cut. Camera stabilization relies on stable visible background features and should be treated cautiously through abrupt cuts or major zoom changes. Calibration is camera-specific and local-map positions are not physical metres without a scale reference. TrackSight flags evidence for human review; it does not replace race officials or make automated incident decisions.

## License

No public source-code licence has been selected yet. The demonstration footage has separate attribution and licence terms in [media attribution](docs/media-attribution.md).
