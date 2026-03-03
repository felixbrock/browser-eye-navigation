#!/usr/bin/env python3
"""Tab-focused calibration/training data collection and model fitting."""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from collections import defaultdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from screeninfo import get_monitors

from gaze_core import (
    EyeTracker,
    L_BOTTOM,
    L_OUTER,
    L_TOP,
    R_BOTTOM,
    R_OUTER,
    R_TOP,
    WEBCAM_H,
    WEBCAM_INDEX,
    WEBCAM_W,
)
from tab_model import (
    TAB_SETTINGS_FILE,
    chromium_tab_rectangles,
    focus_chromium_family_window,
    fit_tab_model,
    is_active_chromium_family_window,
    predict_tab,
    save_tab_model,
    select_chromium_tab,
    tab_rectangles,
)


OUT_DIR = "calibration_logs"
TARGET_TOP_H = 72
DEFAULT_TABS = 8
DEFAULT_ROUNDS = 2
TARGET_CAPTURE_SECONDS = 1.1
TARGET_DWELL_SECONDS = 0.25
TARGET_TIMEOUT_SECONDS = 6.0
MIN_EYE_OPEN_RATIO = 0.075
MAX_IRIS_STEP = 0.090
MAX_HEAD_MOTION = 0.25
BROWSER_SETTLE_SECONDS = 0.30
DEFAULT_OVERLAY_PORT = 8765


class BrowserOverlayPublisher:
    """Small localhost JSON state publisher for Chromium extension overlays."""

    def __init__(self, port):
        self.port = int(port)
        self._server = None
        self._thread = None
        self._lock = threading.Lock()
        self._state = {
            "enabled": False,
            "target_tab": 0,
            "trial_index": 0,
            "trial_total": 0,
            "phase": "idle",
            "updated_at": time.time(),
        }

    def start(self):
        publisher = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/state":
                    self.send_response(404)
                    self.end_headers()
                    return
                with publisher._lock:
                    body = json.dumps(publisher._state).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt, *args):
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def update(self, **kwargs):
        with self._lock:
            self._state.update(kwargs)
            self._state["updated_at"] = time.time()

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._thread = None


def _eye_open_ratio(lm):
    left_open = float(np.hypot(lm[L_BOTTOM].x - lm[L_TOP].x, lm[L_BOTTOM].y - lm[L_TOP].y))
    right_open = float(np.hypot(lm[R_BOTTOM].x - lm[R_TOP].x, lm[R_BOTTOM].y - lm[R_TOP].y))
    inter_eye = float(np.hypot(lm[L_OUTER].x - lm[R_OUTER].x, lm[L_OUTER].y - lm[R_OUTER].y))
    return ((left_open + right_open) * 0.5) / (inter_eye + 1e-7)


def _head_motion_norm(head_state, anchor_state):
    if anchor_state is None:
        return 0.0
    hx, hy, hs = head_state
    ax, ay, a_scale = anchor_state
    dx_norm = float((hx - ax) / (a_scale + 1e-7))
    dy_norm = float((hy - ay) / (a_scale + 1e-7))
    ds = float(np.log((hs + 1e-7) / (a_scale + 1e-7)))
    return max(abs(dx_norm), abs(dy_norm), abs(ds))


def _draw_target_hint(canvas, tab_count, target_idx, trial_idx, total_trials, samples, rejects, phase):
    canvas[:] = (18, 18, 18)
    cv2.putText(
        canvas,
        f"Target browser tab: {target_idx + 1}",
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (245, 245, 245),
        2,
    )
    cv2.putText(
        canvas,
        f"Trial {trial_idx}/{total_trials}  phase={phase}  samples={samples} rejects={rejects}",
        (18, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (190, 190, 190),
        1,
    )

    left = 18
    top = 88
    width = canvas.shape[1] - 36
    gap = 6
    tab_w = max(24, int((width - gap * (tab_count - 1)) / max(tab_count, 1)))
    tab_h = 44
    x = left
    for idx in range(tab_count):
        x1 = x
        x2 = min(left + width, x1 + tab_w)
        is_target = idx == target_idx
        fill = (40, 40, 40)
        border = (120, 120, 120)
        text = (220, 220, 220)
        if is_target:
            fill = (30, 30, 140)
            border = (30, 30, 255)
            text = (235, 235, 255)
        cv2.rectangle(canvas, (x1, top), (x2, top + tab_h), fill, -1)
        cv2.rectangle(canvas, (x1, top), (x2, top + tab_h), border, 3 if is_target else 1)
        cv2.putText(
            canvas,
            str(idx + 1),
            (x1 + 9, top + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            text,
            2,
        )
        x = x2 + gap


def _fit_from_log(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    samples_by_tab = defaultdict(list)
    for trial in payload.get("trials", []):
        tab_idx = int(trial["tab_index"])
        for s in trial.get("samples", []):
            samples_by_tab[tab_idx].append([float(s["iris_h"]), float(s["iris_v"])])
    model = fit_tab_model(samples_by_tab)
    if model is None:
        raise RuntimeError("not enough valid samples in log to fit tab model")
    save_tab_model(model)
    print(f"Saved tab model to: {TAB_SETTINGS_FILE}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabs", type=int, default=DEFAULT_TABS)
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--log-file", type=str, default="")
    ap.add_argument("--allow-any-window", action="store_true")
    ap.add_argument("--show-status-window", action="store_true")
    ap.add_argument("--disable-browser-overlay", action="store_true")
    ap.add_argument("--browser-overlay-port", type=int, default=DEFAULT_OVERLAY_PORT)
    args = ap.parse_args()

    if args.fit_only:
        if not args.log_file:
            raise SystemExit("--fit-only requires --log-file")
        _fit_from_log(args.log_file)
        return

    tab_count = max(2, min(20, int(args.tabs)))
    rounds = max(1, min(8, int(args.rounds)))
    if not args.allow_any_window and tab_count > 9:
        raise SystemExit("Chromium auto tab selection supports up to 9 tabs. Use --tabs <= 9.")

    monitor = get_monitors()[0]
    sw, sh = monitor.width, monitor.height
    if args.allow_any_window:
        _ = tab_rectangles(sw, TARGET_TOP_H, tab_count)
    else:
        ok, focus_err = focus_chromium_family_window()
        if not ok:
            raise SystemExit(focus_err)
        _, err = chromium_tab_rectangles(sw, TARGET_TOP_H, tab_count)
        if err is not None:
            raise SystemExit(
                f"{err}\nCould not auto-focus Chromium-family window. Pass --allow-any-window if needed."
            )
    trial_order = []
    for _ in range(rounds):
        arr = list(range(tab_count))
        random.shuffle(arr)
        trial_order.extend(arr)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_H)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    tracker = EyeTracker()
    win = "Tab Calibration Status"
    trials = []
    samples_by_tab = defaultdict(list)
    prev_h = None
    prev_v = None

    show_status = bool(args.show_status_window)
    use_browser_overlay = (not args.allow_any_window) and (not args.disable_browser_overlay)
    overlay_publisher = None
    if use_browser_overlay:
        try:
            overlay_publisher = BrowserOverlayPublisher(args.browser_overlay_port)
            overlay_publisher.start()
            print(
                "Browser overlay publisher running at "
                f"http://127.0.0.1:{args.browser_overlay_port}/state"
            )
        except OSError as exc:
            print(
                "WARNING: could not start browser overlay publisher; "
                f"continuing without extension overlay ({exc})"
            )
            overlay_publisher = None

    try:
        tracker.reset_head_anchor()
        for trial_idx, tab_idx in enumerate(trial_order, start=1):
            if overlay_publisher is not None:
                overlay_publisher.update(
                    enabled=True,
                    target_tab=int(tab_idx) + 1,
                    trial_index=int(trial_idx),
                    trial_total=int(len(trial_order)),
                    phase="settle",
                )
            samples = []
            quality_rejects = 0
            settled_since = None
            capture_started = None
            t0 = time.time()
            trial_target_set = False
            phase = "paused"

            while True:
                elapsed = time.time() - t0
                if elapsed >= TARGET_TIMEOUT_SECONDS:
                    break
                if show_status:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        print("Cancelled.")
                        return
                browser_active = True
                if not args.allow_any_window:
                    browser_active = is_active_chromium_family_window()
                if not browser_active:
                    phase = "paused"
                    trial_target_set = False
                    settled_since = None
                    capture_started = None
                    t0 = time.time()
                    if overlay_publisher is not None:
                        overlay_publisher.update(
                            enabled=False,
                            target_tab=int(tab_idx) + 1,
                            trial_index=int(trial_idx),
                            trial_total=int(len(trial_order)),
                            phase="paused",
                        )
                    if show_status:
                        disp = np.zeros((154, 920, 3), dtype=np.uint8)
                        _draw_target_hint(
                            disp,
                            tab_count=tab_count,
                            target_idx=tab_idx,
                            trial_idx=trial_idx,
                            total_trials=len(trial_order),
                            samples=len(samples),
                            rejects=quality_rejects,
                            phase=phase,
                        )
                        cv2.imshow(win, disp)
                        cv2.moveWindow(win, 16, 16)
                    continue
                if not args.allow_any_window and not trial_target_set:
                    _, err = chromium_tab_rectangles(sw, TARGET_TOP_H, tab_count)
                    if err is not None:
                        phase = "paused"
                        t0 = time.time()
                        continue
                    select_chromium_tab(tab_idx)
                    time.sleep(BROWSER_SETTLE_SECONDS)
                    trial_target_set = True
                    t0 = time.time()
                    phase = "settle"
                    if overlay_publisher is not None:
                        overlay_publisher.update(
                            enabled=True,
                            target_tab=int(tab_idx) + 1,
                            trial_index=int(trial_idx),
                            trial_total=int(len(trial_order)),
                            phase=phase,
                        )
                ret, frame = cap.read()
                if not ret:
                    continue
                result = tracker.process(frame, apply_head_comp=True, return_meta=True)

                disp = np.zeros((154, 920, 3), dtype=np.uint8)

                if result is not None:
                    (h, v), lm, head_state = result
                    eye_open = _eye_open_ratio(lm)
                    head_motion = _head_motion_norm(head_state, tracker.head_anchor)
                    iris_step = 0.0
                    if prev_h is not None and prev_v is not None:
                        iris_step = float(np.hypot(h - prev_h, v - prev_v))
                    prev_h = h
                    prev_v = v
                    quality_ok = (
                        eye_open >= MIN_EYE_OPEN_RATIO
                        and head_motion <= MAX_HEAD_MOTION
                        and iris_step <= MAX_IRIS_STEP
                    )
                    if quality_ok:
                        if settled_since is None:
                            settled_since = elapsed
                    else:
                        settled_since = None
                        quality_rejects += 1

                    if (
                        capture_started is None
                        and settled_since is not None
                        and (elapsed - settled_since) >= TARGET_DWELL_SECONDS
                    ):
                        capture_started = elapsed

                    if capture_started is not None and quality_ok:
                        phase = "capture"
                        if overlay_publisher is not None:
                            overlay_publisher.update(phase="capture")
                        samples.append(
                            {
                                "t": float(elapsed),
                                "iris_h": float(h),
                                "iris_v": float(v),
                                "eye_open_ratio": float(eye_open),
                                "head_motion_norm": float(head_motion),
                            }
                        )

                    if capture_started is not None and (elapsed - capture_started) >= TARGET_CAPTURE_SECONDS:
                        break

                if show_status:
                    _draw_target_hint(
                        disp,
                        tab_count=tab_count,
                        target_idx=tab_idx,
                        trial_idx=trial_idx,
                        total_trials=len(trial_order),
                        samples=len(samples),
                        rejects=quality_rejects,
                        phase=phase,
                    )
                    cv2.imshow(win, disp)
                    cv2.moveWindow(win, 16, 16)

            for s in samples:
                samples_by_tab[tab_idx].append([s["iris_h"], s["iris_v"]])
            trials.append(
                {
                    "trial_index": trial_idx,
                    "tab_index": int(tab_idx),
                    "samples": samples,
                    "summary": {
                        "samples": len(samples),
                        "quality_rejects": int(quality_rejects),
                    },
                }
            )
            print(f"Trial {trial_idx}/{len(trial_order)} tab={tab_idx + 1} samples={len(samples)} rejects={quality_rejects}")

        model = fit_tab_model(samples_by_tab)
        if model is None:
            print("Not enough valid data to fit tab model.")
            return
        save_tab_model(model)

        # quick in-sample score for operator feedback
        correct = 0
        total = 0
        for tab_idx, pts in samples_by_tab.items():
            for h, v in pts:
                pred, _, _ = predict_tab(model, h, v)
                correct += int(pred == tab_idx)
                total += 1
        train_acc = float(correct) / float(total) if total else 0.0

        payload = {
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "screen": {"width": sw, "height": sh},
            "tab_count": tab_count,
            "rounds": rounds,
            "target_capture_seconds": TARGET_CAPTURE_SECONDS,
            "target_dwell_seconds": TARGET_DWELL_SECONDS,
            "target_timeout_seconds": TARGET_TIMEOUT_SECONDS,
            "model": model,
            "train_accuracy_in_sample": train_acc,
            "trials": trials,
        }
        os.makedirs(OUT_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(OUT_DIR, f"tab_calibration_{stamp}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved tab calibration log: {out_path}")
        print(f"Saved tab model: {TAB_SETTINGS_FILE}")
        print(f"In-sample training accuracy: {train_acc * 100.0:.1f}% ({correct}/{total})")
    finally:
        if overlay_publisher is not None:
            overlay_publisher.update(enabled=False, phase="done")
            overlay_publisher.stop()
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
