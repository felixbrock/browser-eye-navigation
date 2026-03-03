#!/usr/bin/env python3
"""Guided accuracy test for tab-focused gaze classification."""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from screeninfo import get_monitors

from gaze_core import EyeTracker, WEBCAM_H, WEBCAM_INDEX, WEBCAM_W
from tab_model import (
    TAB_SETTINGS_FILE,
    chromium_tab_rectangles,
    focus_chromium_family_window,
    is_active_chromium_family_window,
    load_tab_model,
    predict_tab,
    select_chromium_tab,
    tab_rectangles,
)


TOP_H = 72
CAPTURE_SECONDS = 1.0
FOCUS_SETTLE_SECONDS = 0.30
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


def _draw_status(canvas, target_idx, pred_idx, phase, trial_i, total):
    canvas[:] = (18, 18, 18)
    cv2.putText(
        canvas,
        f"Target tab: {target_idx + 1}",
        (18, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.88,
        (245, 245, 245),
        2,
    )
    cv2.putText(
        canvas,
        f"Current pred: {pred_idx + 1 if pred_idx is not None else '-'}",
        (18, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (220, 220, 220),
        2,
    )
    cv2.putText(
        canvas,
        f"Trial {trial_i}/{total}  phase={phase}",
        (18, 126),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (180, 180, 180),
        1,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-file", type=str, default=TAB_SETTINGS_FILE)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--show-status-window", action="store_true")
    ap.add_argument("--allow-any-window", action="store_true")
    ap.add_argument("--disable-browser-overlay", action="store_true")
    ap.add_argument("--browser-overlay-port", type=int, default=DEFAULT_OVERLAY_PORT)
    args = ap.parse_args()

    if not os.path.exists(args.model_file):
        raise SystemExit(f"No model at {args.model_file}. Run tab_calibration.py first.")
    model = load_tab_model(args.model_file)
    tab_count = int(model.get("tab_count") or len(model.get("centers", {})))
    if tab_count < 2:
        raise SystemExit("Invalid model; recalibrate first.")
    if not args.allow_any_window and tab_count > 9:
        raise SystemExit("Chromium test mode supports up to 9 tabs. Recalibrate with --tabs <= 9.")

    monitor = get_monitors()[0]
    sw = monitor.width
    if args.allow_any_window:
        _ = tab_rectangles(sw, TOP_H, tab_count)
    else:
        ok, focus_err = focus_chromium_family_window()
        if not ok:
            raise SystemExit(focus_err)
        _, err = chromium_tab_rectangles(sw, TOP_H, tab_count)
        if err is not None:
            raise SystemExit(
                f"{err}\nCould not auto-focus Chromium-family window. Pass --allow-any-window if needed."
            )

    trial_order = []
    rounds = max(1, min(5, int(args.rounds)))
    for _ in range(rounds):
        arr = list(range(tab_count))
        random.shuffle(arr)
        trial_order.extend(arr)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_H)
    if not cap.isOpened():
        raise SystemExit("ERROR: Cannot open webcam")
    tracker = EyeTracker()
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

    win = "Tab Test Status"
    correct = 0
    total = 0
    try:
        for idx, target_tab in enumerate(trial_order, start=1):
            if not args.allow_any_window:
                while True:
                    if is_active_chromium_family_window():
                        _, err = chromium_tab_rectangles(sw, TOP_H, tab_count)
                        if err is None:
                            break
                    if args.show_status_window:
                        disp = np.zeros((160, 460, 3), dtype=np.uint8)
                        _draw_status(disp, target_tab, None, "paused", idx, len(trial_order))
                        cv2.imshow(win, disp)
                        cv2.moveWindow(win, 16, 16)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            print("Cancelled.")
                            return
                    else:
                        time.sleep(0.05)
                select_chromium_tab(target_tab)
                time.sleep(FOCUS_SETTLE_SECONDS)
            if overlay_publisher is not None:
                overlay_publisher.update(
                    enabled=True,
                    target_tab=int(target_tab) + 1,
                    trial_index=int(idx),
                    trial_total=int(len(trial_order)),
                    phase="capture",
                )

            pred_tab = None
            phase = "capture"
            active_elapsed = 0.0
            last_tick = time.monotonic()
            votes = []
            print(f"Trial {idx}/{len(trial_order)} target={target_tab + 1} capturing...")
            while active_elapsed < CAPTURE_SECONDS:
                now = time.monotonic()
                dt = now - last_tick
                last_tick = now
                browser_active = True
                if not args.allow_any_window:
                    browser_active = is_active_chromium_family_window()
                if not browser_active:
                    phase = "paused"
                    if overlay_publisher is not None:
                        overlay_publisher.update(
                            enabled=False,
                            target_tab=int(target_tab) + 1,
                            trial_index=int(idx),
                            trial_total=int(len(trial_order)),
                            phase=phase,
                        )
                    if args.show_status_window:
                        disp = np.zeros((160, 460, 3), dtype=np.uint8)
                        _draw_status(disp, target_tab, pred_tab, phase, idx, len(trial_order))
                        cv2.imshow(win, disp)
                        cv2.moveWindow(win, 16, 16)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            print("Cancelled.")
                            return
                    else:
                        time.sleep(0.02)
                    continue
                active_elapsed += dt
                phase = "capture"
                ret, frame = cap.read()
                if not ret:
                    continue
                result = tracker.process(frame, apply_head_comp=True)
                if result is not None:
                    (h, v), _ = result
                    pred_tab, _, _ = predict_tab(model, h, v)
                    if pred_tab is not None:
                        votes.append(pred_tab)

                if args.show_status_window:
                    disp = np.zeros((160, 460, 3), dtype=np.uint8)
                    _draw_status(disp, target_tab, pred_tab, phase, idx, len(trial_order))
                    cv2.imshow(win, disp)
                    cv2.moveWindow(win, 16, 16)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("Cancelled.")
                        return

            if votes:
                vals, counts = np.unique(np.asarray(votes, dtype=int), return_counts=True)
                final_pred = int(vals[int(np.argmax(counts))])
            else:
                final_pred = -1
            ok = final_pred == target_tab
            correct += int(ok)
            total += 1
            print(
                f"Trial {idx}/{len(trial_order)} target={target_tab + 1} "
                f"pred={final_pred + 1 if final_pred >= 0 else '-'} ok={ok}"
            )

        acc = float(correct) / float(max(total, 1))
        print(f"Tab test accuracy: {acc * 100.0:.1f}% ({correct}/{total})")
    finally:
        if overlay_publisher is not None:
            overlay_publisher.update(enabled=False, phase="done")
            overlay_publisher.stop()
        cap.release()
        tracker.close()
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
