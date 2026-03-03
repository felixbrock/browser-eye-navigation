#!/usr/bin/env python3
"""Live tab-focused gaze classifier (separate from cursor eye tracker)."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import deque
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
    tab_rectangles,
)


TOP_H = 72
DEFAULT_SWITCH_PORT = 8766


class SwitchStatePublisher:
    """Local JSON publisher consumed by the tab-switch extension."""

    def __init__(self, port):
        self.port = int(port)
        self._server = None
        self._thread = None
        self._lock = threading.Lock()
        self._state = {
            "running": False,
            "predicted_tab": 0,
            "confidence": 0.0,
            "tab_count": 0,
            "browser_active": False,
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


def _draw_status(canvas, active_idx, conf):
    canvas[:] = (18, 18, 18)
    cv2.putText(
        canvas,
        f"Predicted tab: {active_idx + 1 if active_idx is not None else '-'}",
        (18, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.90,
        (245, 245, 245),
        2,
    )
    cv2.putText(
        canvas,
        f"Confidence: {conf:.2f}",
        (18, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (210, 210, 210),
        2,
    )
    cv2.putText(
        canvas,
        "Ctrl+C to quit",
        (18, 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (170, 170, 170),
        1,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-file", type=str, default=TAB_SETTINGS_FILE)
    ap.add_argument("--show-camera", action="store_true")
    ap.add_argument("--show-status-window", action="store_true")
    ap.add_argument("--allow-any-window", action="store_true")
    ap.add_argument("--disable-switch-publisher", action="store_true")
    ap.add_argument("--switch-publisher-port", type=int, default=DEFAULT_SWITCH_PORT)
    args = ap.parse_args()

    if not os.path.exists(args.model_file):
        raise SystemExit(
            f"Tab model not found at {args.model_file}. Run: uv run python tab_calibration.py"
        )
    model = load_tab_model(args.model_file)
    tab_count = int(model.get("tab_count") or len(model.get("centers", {})))
    if tab_count < 2:
        raise SystemExit("Invalid tab model; recalibrate with tab_calibration.py")

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

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_H)
    if not cap.isOpened():
        raise SystemExit("ERROR: Cannot open webcam")

    tracker = EyeTracker()
    vote_hist = deque(maxlen=7)
    win = "Tab Tracker Status"
    last_conf = 0.0
    last_reported = None
    switch_publisher = None
    if not args.disable_switch_publisher:
        try:
            switch_publisher = SwitchStatePublisher(args.switch_publisher_port)
            switch_publisher.start()
            switch_publisher.update(running=True, tab_count=int(tab_count), browser_active=True)
            print(
                "Tab switch publisher running at "
                f"http://127.0.0.1:{args.switch_publisher_port}/state"
            )
        except OSError as exc:
            print(
                "WARNING: could not start tab switch publisher; "
                f"continuing without extension integration ({exc})"
            )
            switch_publisher = None

    try:
        while True:
            browser_active = True
            if not args.allow_any_window:
                browser_active = is_active_chromium_family_window()
            ret, frame = cap.read()
            if not ret:
                continue
            if switch_publisher is not None and not browser_active:
                switch_publisher.update(
                    running=True,
                    predicted_tab=0,
                    confidence=0.0,
                    tab_count=int(tab_count),
                    browser_active=False,
                )
                if args.show_status_window:
                    disp = np.zeros((160, 420, 3), dtype=np.uint8)
                    _draw_status(disp, None, 0.0)
                    cv2.imshow(win, disp)
                    cv2.moveWindow(win, 16, 16)
                if args.show_camera:
                    cv2.imshow("Tab Tracker Camera", frame)
                if args.show_status_window or args.show_camera:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                continue
            result = tracker.process(frame, apply_head_comp=True)
            active = None
            if result is not None:
                (h, v), _ = result
                pred, conf, _ = predict_tab(model, h, v)
                if pred is not None:
                    vote_hist.append(pred)
                    vals, counts = np.unique(np.asarray(vote_hist, dtype=int), return_counts=True)
                    active = int(vals[int(np.argmax(counts))])
                    last_conf = conf
                    if switch_publisher is not None:
                        switch_publisher.update(
                            running=True,
                            predicted_tab=int(active) + 1,
                            confidence=float(last_conf),
                            tab_count=int(tab_count),
                            browser_active=True,
                        )
                    if active != last_reported:
                        print(f"Predicted tab={active + 1} confidence={last_conf:.2f}")
                        last_reported = active

            if args.show_status_window:
                disp = np.zeros((160, 420, 3), dtype=np.uint8)
                _draw_status(disp, active, last_conf)
                cv2.imshow(win, disp)
                cv2.moveWindow(win, 16, 16)
            if args.show_camera:
                cv2.imshow("Tab Tracker Camera", frame)

            if args.show_status_window or args.show_camera:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        if switch_publisher is not None:
            switch_publisher.update(running=False)
            switch_publisher.stop()
        cap.release()
        tracker.close()
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass
        try:
            cv2.destroyWindow("Tab Tracker Camera")
        except cv2.error:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
