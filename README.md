# TrackSight

A computer-vision toolkit for reviewing race footage from a single trackside camera.

## Pipeline

```text
video → frames → vehicle detection → tracking → trajectories → review evidence
```

TrackSight is designed to keep each finding tied to the footage and motion data that produced it.

## Install

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

Place a video in `data/raw`, set `VIDEO_PATH` in `scripts/video_probe.py`, then run:

```powershell
python .\scripts\video_probe.py
```

The video probe reads the footage, reports its metadata, and saves the first frame to:

```text
outputs/frames/first_frame.png
```

## Outputs

Generated files are written to `outputs/`:

- frames and annotated footage
- vehicle detections and tracking records
- trajectories and candidate incident evidence
- review clips

Raw video, model weights, local environments, and generated outputs are kept out of Git.

## License

License to be added before public release.
