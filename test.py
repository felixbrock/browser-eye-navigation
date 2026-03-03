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
from model import (
    TAB_SETTINGS_FILE,
    DEFAULT_SESSION_MAX_TABS,
    DEFAULT_SESSION_MIN_TABS,
    chromium_tab_rectangles,
    close_managed_chromium_session,
    focus_managed_chromium_session,
    focus_chromium_family_window,
    is_active_chromium_family_window,
    launch_managed_chromium_session,
    load_tab_model,
    predict_tab_position,
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
    ap.add_argument("--tabs", type=int, default=0, help="optional runtime tab count for test mapping")
    ap.add_argument("--show-status-window", action="store_true")
    ap.add_argument("--allow-any-window", action="store_true")
    ap.add_argument("--disable-browser-overlay", action="store_true")
    ap.add_argument("--browser-overlay-port", type=int, default=DEFAULT_OVERLAY_PORT)
    args = ap.parse_args()

    if not os.path.exists(args.model_file):
        raise SystemExit(f"No model at {args.model_file}. Run calibration.py first.")
    model = load_tab_model(args.model_file)
    model_tab_count = int(model.get("calibration_tab_count") or model.get("tab_count") or len(model.get("centers", {})))
    if model_tab_count < 2:
        raise SystemExit("Invalid model; recalibrate first.")
    tab_count = int(args.tabs) if int(args.tabs) >= 2 else model_tab_count
    managed_session = None
    if not args.allow_any_window:
        use_overlay = not args.disable_browser_overlay
        overlay_dir = os.path.join(os.path.dirname(__file__), "chromium_tab_overlay_extension")
        managed_session, launch_err = launch_managed_chromium_session(
            min_tabs=DEFAULT_SESSION_MIN_TABS,
            max_tabs=DEFAULT_SESSION_MAX_TABS,
            load_overlay_extension=use_overlay,
            overlay_extension_dir=overlay_dir,
        )
        if launch_err is not None:
            raise SystemExit(launch_err)
        tab_count = int(managed_session["tab_count"])
        print(
            "Launched managed browser session: "
            f"tabs={tab_count} profile={managed_session['profile_dir']}"
        )
        time.sleep(1.2)

    monitor = get_monitors()[0]
    sw = monitor.width
    if args.allow_any_window:
        _ = tab_rectangles(sw, TOP_H, tab_count)
    else:
        ok, focus_err = (False, None)
        if managed_session is not None:
            ok, focus_err = focus_managed_chromium_session(managed_session)
        if not ok:
            ok, focus_err = focus_chromium_family_window()
        if not ok:
            if managed_session is not None:
                close_managed_chromium_session(managed_session)
            raise SystemExit(focus_err)
        _, err = chromium_tab_rectangles(sw, TOP_H, tab_count)
        if err is not None:
            if managed_session is not None:
                close_managed_chromium_session(managed_session)
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
        if managed_session is not None:
            close_managed_chromium_session(managed_session)
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
            if overlay_publisher is not None:
                overlay_publisher.update(
                    enabled=True,
                    target_tab=int(target_tab) + 1,
                    trial_index=int(idx),
                    trial_total=int(len(trial_order)),
                    phase="settle",
                )
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
                if overlay_publisher is None:
                    while True:
                        if select_chromium_tab(target_tab, tab_count=tab_count):
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
            pos_votes = []
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
                    pred_tab, conf, _ = predict_tab(model, h, v, tab_count=tab_count)
                    if pred_tab is not None:
                        votes.append(pred_tab)
                        pos = predict_tab_position(model, h, v)
                        if pos is None:
                            pos = float((int(pred_tab) + 0.5) / float(tab_count))
                        if conf >= 0.12:
                            pos_votes.append(float(pos))

                if args.show_status_window:
                    disp = np.zeros((160, 460, 3), dtype=np.uint8)
                    _draw_status(disp, target_tab, pred_tab, phase, idx, len(trial_order))
                    cv2.imshow(win, disp)
                    cv2.moveWindow(win, 16, 16)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("Cancelled.")
                        return

            if pos_votes:
                pos_arr = np.asarray(pos_votes, dtype=float)
                med_pos = float(np.median(pos_arr))
                final_pred = int(np.clip(med_pos * tab_count, 0, tab_count - 1))
            elif votes:
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
        if managed_session is not None:
            close_managed_chromium_session(managed_session)


if __name__ == "__main__":
    main()
