# Browser Eye Navigation

Chromium tab navigation via webcam gaze tracking.
The tab model now predicts normalized tab position, so it can be remapped to
different runtime tab counts (for example calibrate with 10, run with 3 or 20).

This repository contains the tab-focused workflow that was split from `eye-tracker`:

- `calibration.py`: calibrate and fit tab model
- `test.py`: objective tab accuracy test
- `tracker.py`: live tab prediction + switch-state publisher
- `chromium_tab_overlay_extension/`: calibration/test visual tab markers
- `chromium_tab_switch_extension/`: toggle gaze-driven tab switching
- `train` + `run_codex_tab_log_tuning.sh`: tab-only tuning flow

## Install

```bash
uv sync
```

## Workflow

1. Calibrate model:
```bash
uv run python calibration.py --rounds 2
```

2. Test accuracy:
```bash
uv run python test.py --rounds 1
# Optional runtime remap count:
uv run python test.py --rounds 1 --tabs 20
```

3. Live tracking:
```bash
uv run python tracker.py
# Optional runtime remap count for local status output:
uv run python tracker.py --tabs 20
# Game mode: bottom-right camera+points + red-dot target tab icon (overlay extension required):
uv run python tracker.py --game-mode
# In game mode, target tab icon is replaced with a red dot and restored when target advances.
# Game mode locks to the currently active Chromium window and pauses if focus switches away.
```

4. Train/tune from latest calibration log:
```bash
./train
```
Each run writes reasoning to `calibration_logs/train_runs/` and appends an index
entry in `calibration_logs/train_history.md`.

## Extensions

Load unpacked extensions from:

- `chromium_tab_overlay_extension/`
- `chromium_tab_switch_extension/`

In `tracker` mode, toggle auto-switch with `Ctrl+Shift+Y` (or the extension icon).
