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
   This extension is collection-only.
   Reload it after code changes so the collector keeps receiving current tab ordering and titles.
2. Start the collector:

```bash
UV_CACHE_DIR=.uv-cache uv run python src/data_collection.py
```

3. Keep the browser focused and click tabs with the mouse.
4. Look at the tab before clicking it.
5. Stop with `Ctrl+C`.

Output: `./data/train.jsonl`

Each collected row now includes collector-generated tab geometry, per-tab candidate metadata, and row-level geometry quality flags. The extension still supplies tab identity and ordering, but the collector now rebuilds candidate bounds locally.

## Train The Ranker

Install dependencies, collect data, then run:

```bash
UV_CACHE_DIR=.uv-cache uv run python src/train.py
```

By default this writes:

- `models/tab_ranker.pkl`
- `models/tab_ranker.metadata.json`
- `models/tab_ranker.metrics.json`

By default, training keeps only rows whose geometry was measured by the collector and whose click lands inside the reported clicked tab bounds. Use `--allow-unmeasured-geometry` or `--allow-suspect-geometry` if you need to include older or lower-confidence rows.

If you train on older rows that do not yet contain candidate-tab metadata, `src/train.py` still has a legacy equal-width fallback so the pipeline can run when explicitly allowed.

## Run The Model

Load `tab_runtime_control_extension` as a separate unpacked extension in Brave/Chrome for runtime control. This runtime extension reports focused-window tab state to the local runtime and activates tabs in the exact browser window returned by the runtime.

```bash
UV_CACHE_DIR=.uv-cache uv run python src/run_model.py --model-path models/tab_ranker_v2.pkl
```

Runtime defaults:

- localhost port `8768`
- runtime extension currently uses a page-level `Ctrl+B` listener as a console-log smoke test
- Python runtime only returns predictions
- tab switching happens inside the runtime extension while autoplay is enabled
- only the currently focused Chromium window is eligible for switching
