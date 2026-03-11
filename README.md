# Browser Eye Navigation

Collect training data for browser tab selection from webcam gaze/head features plus confirmed tab clicks.

## Setup

```bash
UV_CACHE_DIR=.uv-cache uv sync
mkdir -p models
curl -L -o models/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
```

Requirements:

- Linux with X11
- `xdotool`
- webcam
- Brave, Chrome, or Chromium

## Start Data Collection

1. Load `tab_click_logger_extension` as an unpacked extension in Brave/Chrome.
   Reload it after code changes so the richer candidate-tab metadata is collected.
2. Start the collector:

```bash
UV_CACHE_DIR=.uv-cache uv run python src/data_collection.py
```

3. Keep the browser focused and click tabs with the mouse.
4. Look at the tab before clicking it.
5. Stop with `Ctrl+C`.

Output: `./data/train.jsonl`

Each collected row now includes approximate tab-strip geometry plus candidate tab metadata from the browser extension, which the trainer uses to build ranking examples.

## Train The Ranker

Install dependencies, collect data, then run:

```bash
UV_CACHE_DIR=.uv-cache uv run python src/train.py
```

By default this writes:

- `models/tab_ranker.pkl`
- `models/tab_ranker.metadata.json`
- `models/tab_ranker.metrics.json`

If you train on older rows that do not yet contain candidate-tab metadata, `src/train.py` falls back to a legacy equal-width tab layout estimate so the pipeline still runs.
