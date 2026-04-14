# VigilEye

VigilEye is a local-only student attentive alertness system for online classes, self-study sessions, and smart classroom experiments. It uses the modern MediaPipe Face Landmarker task API, webcam-based facial analysis, and a lightweight Streamlit dashboard to estimate attention in real time without sending any data to the cloud.

## What It Does

VigilEye combines multiple signals into a single real-time attention score:

- Eye closure and drowsiness using Eye Aspect Ratio (EAR)
- Yawning and fatigue using Mouth Aspect Ratio (MAR)
- Head posture using OpenCV solvePnP with MediaPipe 3D landmarks
- Looking-away detection using refined iris landmarks and gaze deviation
- Lightweight engagement heuristics using Face Landmarker blendshape scores
- Session logging and a live instructor dashboard

The output is a continuously updated 0-100 attention score, a status label, CSV session logs, and a live dashboard view.

## Core Stack

- MediaPipe Face Landmarker task API
- OpenCV for webcam capture, visualization, and head pose estimation
- NumPy and pandas for metrics and logging
- Streamlit for the instructor dashboard
- Pygame for optional local audio alerts

## Project Layout

```text
VigilEye/
├── launch_vigileye.py        # Unified launcher for detector + dashboard
├── student_alertness.py      # Real-time detector and scoring pipeline
├── dashboard.py              # Streamlit instructor dashboard
├── requirements.txt          # Verified dependency set
├── models/                   # Face Landmarker task model lives here
├── runtime/                  # Live JSON state for the dashboard
└── logs/                     # Session CSV logs
```

## Features

- Mirror-view webcam preview with color-coded overlays
- Real-time display of EAR, MAR, head angles, gaze offsets, FPS, status, and attention score
- Attention state categories such as attentive, moderate, low attention, drowsy, looking away, and no face detected
- Optional audio alerts with graceful fallback to terminal bell if audio initialization fails
- Built-in self-test mode for model and webcam verification
- Headless mode for automated smoke tests or remote sessions
- Local dashboard with live session metrics, recent events, and trend charts
- CSV logging for post-session analysis

## How It Works

### Detector pipeline

1. Capture a frame from the webcam.
2. Run MediaPipe Face Landmarker in live-stream mode.
3. Extract facial landmarks, iris landmarks, and blendshape scores.
4. Compute EAR, MAR, head pose, gaze deviation, and engagement heuristics.
5. Combine the subsignals into a weighted attention score.
6. Render overlays, write runtime JSON, and append to the session CSV log.

### Attention score inputs

The score uses a weighted combination of:

- Eyes: 32%
- Mouth: 10%
- Head pose: 20%
- Gaze: 22%
- Engagement blendshapes: 16%

These thresholds and weights are configurable near the top of `student_alertness.py`.

## Requirements

- Python 3.11 or 3.12 recommended
- A webcam
- A desktop session that can open OpenCV windows for the live preview mode
- Linux, macOS, or Windows with a compatible Python scientific stack

## Important Python Note

If your machine defaults to Python 3.13 and scientific imports crash, use Python 3.12 instead. On this machine, the project was validated successfully with a project-local Conda environment at `.py312` because the original Python 3.13 local environment segfaulted when importing NumPy.

## Installation

### Recommended setup on this machine

```bash
conda create -p "$PWD/.py312" python=3.12 pip -y
"$PWD/.py312/bin/python" -m pip install -r requirements.txt
```

### Standard virtual environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Verified dependency versions

```bash
python -m pip install \
	opencv-python==4.13.0.92 \
	mediapipe==0.10.33 \
	numpy==2.4.4 \
	pandas==3.0.2 \
	streamlit==1.56.0 \
	streamlit-autorefresh==1.0.1 \
	pygame==2.6.1
```

## Download the Face Landmarker Model

Download the current float16 Face Landmarker task bundle into `models/`:

```bash
mkdir -p models
curl -L https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task -o models/face_landmarker.task
```

The implementation intentionally uses the task API model bundle rather than the deprecated legacy FaceMesh path.

## Quick Start

### One-command launcher

The simplest way to run the system is:

```bash
./.py312/bin/python launch_vigileye.py --camera-check
```

This launcher:

- picks the best available local Python interpreter, preferring `.py312`
- runs a detector preflight check before the session
- starts the Streamlit dashboard
- starts the real-time detector
- shuts the dashboard down automatically when the detector exits

### Dashboard URL

When the launcher starts successfully, the dashboard is available at:

```text
http://localhost:8501
```

## Common Commands

### Preflight only

```bash
./.py312/bin/python student_alertness.py --self-test --camera-check
```

### Launch detector and dashboard together

```bash
./.py312/bin/python launch_vigileye.py --camera-check
```

### Detector only

```bash
./.py312/bin/python launch_vigileye.py --detector-only
```

### Dashboard only

```bash
./.py312/bin/python launch_vigileye.py --dashboard-only --dashboard-headless
```

### Headless automated smoke run

```bash
./.py312/bin/python launch_vigileye.py \
	--camera-check \
	--headless-detector \
	--detector-max-frames 30 \
	--dashboard-headless \
	--session-name smoke_test
```

### Run detector directly

```bash
./.py312/bin/python student_alertness.py --model models/face_landmarker.task --session-name class_demo
```

### Run dashboard directly

```bash
./.py312/bin/streamlit run dashboard.py
```

## Launcher Options

Most useful `launch_vigileye.py` flags:

| Flag | Purpose |
| --- | --- |
| `--camera-check` | Include webcam probing in preflight |
| `--skip-self-test` | Skip model and webcam preflight |
| `--detector-only` | Run only the detector |
| `--dashboard-only` | Run only the Streamlit dashboard |
| `--dashboard-headless` | Start Streamlit without opening a browser |
| `--headless-detector` | Run detector without the OpenCV preview window |
| `--detector-max-frames N` | Stop automatically after `N` frames |
| `--session-name NAME` | Label the current run and its log files |
| `--camera-index N` | Choose a different webcam |

## Detector Options

Most useful `student_alertness.py` flags:

| Flag | Purpose |
| --- | --- |
| `--self-test` | Validate model loading and MediaPipe initialization |
| `--camera-check` | Validate webcam access during self-test |
| `--no-display` | Disable the OpenCV preview window |
| `--max-frames N` | Stop after `N` frames |
| `--no-audio-alerts` | Disable local audio alerts |
| `--alert-sound PATH` | Use a custom alert sound |
| `--session-name NAME` | Label the session for logs and dashboard |

## Outputs

### Runtime state

The detector writes live state to:

```text
runtime/latest_state.json
```

This file is read by the Streamlit dashboard.

### Session logs

Each run writes a CSV log to:

```text
logs/<session_name>_<timestamp>.csv
```

Each row includes:

- timestamp
- session name
- attention score
- status
- EAR and MAR
- pitch, yaw, and roll
- gaze offsets and deviation
- engagement label
- fatigue and look-away durations
- alert state

## Dashboard

The dashboard provides:

- live attention score and status
- EAR, MAR, FPS, gaze, and head-pose summaries
- recent session events and attention trend charts
- status distribution across the current log
- simple instructor guidance based on the current detected state

If the dashboard says that no live runtime state is available, make sure:

- the detector is running
- `models/face_landmarker.task` exists
- the webcam opened successfully
- `runtime/latest_state.json` is being updated

## Tuning

The main thresholds live at the top of `student_alertness.py`.

Important values to tune:

- `EAR_CLOSED_THRESHOLD`
- `DROWSY_EYES_CLOSED_SECONDS`
- `MAR_YAWN_THRESHOLD`
- `YAWN_SECONDS`
- `HEAD_YAW_LIMIT_DEG`
- `HEAD_PITCH_LIMIT_DEG`
- `GAZE_HORIZONTAL_LIMIT`
- `GAZE_VERTICAL_LIMIT`
- `ATTENTION_WEIGHTS`

For a new webcam or seating position, tune conservatively first to reduce false positives.

## Troubleshooting

### NumPy or MediaPipe crashes on startup

Use Python 3.12 instead of Python 3.13.

### `Model not found`

Download the model to `models/face_landmarker.task` using the command above.

### Dashboard starts but shows no live state

Run the detector first or use the unified launcher.

### Detector opens but stays at `NO FACE DETECTED`

Check lighting, camera framing, webcam permissions, and whether your face is mostly frontal and inside the frame.

### Qt font warnings from OpenCV on Linux

Those warnings are non-fatal. The detector can still run.

### Audio alert does not play

If `pygame` cannot initialize the mixer on your system, the detector falls back to a terminal bell character.

## Privacy

- all inference is local
- no cloud calls are made
- no external emotion API is used
- the dashboard reads local files only

If you use this in a classroom setting, make sure learners know what is being measured and why.

## Current Scope

This implementation is designed for a single visible learner on one webcam. It is suitable for local experimentation, research prototypes, and instructor tooling, but it is not a substitute for broader classroom context or human judgment.

## Validation Status

This project was verified locally on April 14, 2026 with the project-local Python 3.12 environment:

- detector self-test passed
- Face Landmarker initialized successfully
- webcam index 0 produced frames
- dashboard booted successfully
- headless 60-frame detector smoke test passed
- unified launcher preflight plus coordinated startup and shutdown passed

## Roadmap

Potential next improvements:

1. Per-user calibration for EAR, gaze, and posture baselines.
2. Multi-student support with separate face tracking IDs.
3. ONNX Runtime or TensorRT acceleration.
4. Better boredom versus confusion disambiguation.
5. Teacher-side analytics and intervention summaries.