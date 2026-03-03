# Browser Eye Navigation

Chromium tab navigation via webcam gaze tracking.

This repository contains the tab-focused workflow that was split from `eye-tracker`:

- `tab_calibration.py`: calibrate and fit tab model
- `tab_test.py`: objective tab accuracy test
- `tab_tracker.py`: live tab prediction + switch-state publisher
- `chromium_tab_overlay_extension/`: calibration/test visual tab markers
- `chromium_tab_switch_extension/`: toggle gaze-driven tab switching
- `train_tab` + `run_codex_tab_log_tuning.sh`: tab-only tuning flow

## Install

```bash
uv sync
```

## Workflow

1. Calibrate model:
```bash
uv run python tab_calibration.py --tabs 8 --rounds 2
```

2. Test accuracy:
```bash
uv run python tab_test.py --rounds 1
```

3. Live tracking:
```bash
uv run python tab_tracker.py
```

## Extensions

Load unpacked extensions from:

- `chromium_tab_overlay_extension/`
- `chromium_tab_switch_extension/`

In `tab_tracker` mode, toggle auto-switch with `Ctrl+Shift+Y` (or the extension icon).
