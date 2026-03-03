#!/usr/bin/env python3
"""Live tab-focused gaze classifier (separate from cursor eye tracker)."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from screeninfo import get_monitors

from gaze_core import (
    EyeTracker,
    L_BOTTOM,
    L_INNER,
    L_IRIS_RING,
    L_OUTER,
    L_TOP,
    R_BOTTOM,
    R_INNER,
    R_IRIS_RING,
    R_OUTER,
    R_TOP,
    WEBCAM_H,
    WEBCAM_INDEX,
    WEBCAM_W,
)
from model import (
    TAB_SETTINGS_FILE,
    active_window_geometry,
    active_window_id,
    chromium_tab_rectangles,
    focus_chromium_family_window,
    is_active_chromium_family_window,
    load_tab_model,
    predict_tab_position,
    predict_tab,
    select_chromium_tab,
    tab_rectangles,
)


TOP_H = 72
DEFAULT_SWITCH_PORT = 8766
DEFAULT_OVERLAY_PORT = 8765
POSITION_EMA_ALPHA = 0.35
MIN_CONFIDENCE_FOR_SWITCH = 0.16
SWITCH_MAJORITY_THRESHOLD = 0.66
DEBUG_CAMERA_WIN = "Tracker Game Camera"
DEBUG_CAMERA_W = 360
DEBUG_CAMERA_H = 270
GAME_PROMPT_WIN = "Tracker Game Prompt"
GAME_HIT_SECONDS = 0.60
OPAQUE_WINDOW_ALPHA_32 = "4294967295"


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
            "predicted_position": 0.0,
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


class BrowserOverlayPublisher:
    """Small localhost JSON publisher for Chromium extension overlays."""

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
            "auto_activate": False,
            "mark_target_only": True,
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


def _draw_measurement_points(frame, landmarks):
    """Draw the key face points used by gaze measurement."""
    if landmarks is None:
        return frame

    h_px, w_px = frame.shape[:2]
    out = frame.copy()

    # Eye contour landmarks used for horizontal/vertical ratio normalization.
    eye_points = (R_OUTER, R_INNER, R_TOP, R_BOTTOM, L_OUTER, L_INNER, L_TOP, L_BOTTOM)
    for idx in eye_points:
        lm = landmarks[idx]
        x = int(np.clip(lm.x * w_px, 0, w_px - 1))
        y = int(np.clip(lm.y * h_px, 0, h_px - 1))
        cv2.circle(out, (x, y), 3, (40, 220, 255), -1)

    # Iris ring points for both eyes.
    for idx in tuple(R_IRIS_RING) + tuple(L_IRIS_RING):
        lm = landmarks[idx]
        x = int(np.clip(lm.x * w_px, 0, w_px - 1))
        y = int(np.clip(lm.y * h_px, 0, h_px - 1))
        cv2.circle(out, (x, y), 2, (0, 255, 120), -1)

    cv2.putText(
        out,
        "Eye measurement points",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
    )
    return out


def _show_debug_camera_window(frame, landmarks, screen_w, screen_h):
    """Show the game camera feed in a bottom-right floating window."""
    annotated = _draw_measurement_points(frame, landmarks)
    dbg = cv2.resize(annotated, (DEBUG_CAMERA_W, DEBUG_CAMERA_H), interpolation=cv2.INTER_AREA)
    x = max(0, int(screen_w) - DEBUG_CAMERA_W - 16)
    y = max(0, int(screen_h) - DEBUG_CAMERA_H - 56)
    cv2.imshow(DEBUG_CAMERA_WIN, dbg)
    cv2.moveWindow(DEBUG_CAMERA_WIN, x, y)


def _force_window_opaque(window_name):
    """Best-effort X11 opacity override to disable compositor translucency."""
    if shutil.which("xdotool") is None or shutil.which("xprop") is None:
        return False
    try:
        proc = subprocess.run(
            ["xdotool", "search", "--name", window_name],
            check=True,
            capture_output=True,
            text=True,
        )
        ids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if not ids:
            return False
        win_id = ids[-1]
        subprocess.run(
            [
                "xprop",
                "-id",
                str(win_id),
                "-f",
                "_NET_WM_WINDOW_OPACITY",
                "32c",
                "-set",
                "_NET_WM_WINDOW_OPACITY",
                OPAQUE_WINDOW_ALPHA_32,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        return True
    except Exception:
        return False


def _is_active_chrome_family_window():
    """True when active window is Google Chrome/Chromium (not Brave)."""
    win = active_window_geometry()
    if win is None:
        return False
    klass = str(win.get("class", "")).lower()
    return ("google-chrome" in klass) or ("chromium" in klass)


def _focus_game_browser_prefer_chrome():
    """Focus Chrome/Chromium first for game mode, then fall back to generic focus."""
    if shutil.which("xdotool") is not None:
        patterns = (
            "google-chrome",
            "Google-chrome",
            "google-chrome-stable",
            "Google-chrome-stable",
            "chromium",
            "Chromium",
        )
        found_chrome_candidate = False
        for pattern in patterns:
            try:
                proc = subprocess.run(
                    ["xdotool", "search", "--class", pattern],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError:
                continue
            win_ids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            if win_ids:
                found_chrome_candidate = True
            for win_id in reversed(win_ids):
                try:
                    subprocess.run(
                        ["xdotool", "windowactivate", "--sync", str(win_id)],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    time.sleep(0.08)
                    if _is_active_chrome_family_window():
                        return True, None
                except subprocess.CalledProcessError:
                    continue
        if found_chrome_candidate:
            return False, "Found Google Chrome/Chromium window(s) but could not focus one."
    return focus_chromium_family_window()


def _clean_browser_title(title):
    t = str(title or "").strip()
    for suffix in (" - Brave", " - Google Chrome", " - Chromium", " - Chrome"):
        if t.endswith(suffix):
            t = t[: -len(suffix)].rstrip()
    return t.strip() or "Untitled"


def _collect_tab_titles(tab_count):
    """Best-effort title probe by selecting each tab once."""
    titles = [f"Tab {i + 1}" for i in range(tab_count)]
    if not is_active_chromium_family_window():
        return titles
    for idx in range(tab_count):
        if not select_chromium_tab(idx, tab_count=tab_count):
            continue
        time.sleep(0.12)
        win = active_window_geometry()
        if win is None:
            continue
        title = _clean_browser_title(win.get("title", ""))
        if title:
            titles[idx] = title
    return titles


def _choose_next_game_target(tab_count, current_target=None):
    choices = list(range(tab_count))
    if current_target is not None and len(choices) > 1:
        choices = [c for c in choices if int(c) != int(current_target)]
    return int(random.choice(choices))


def _show_game_prompt_window(
    *,
    target_title,
    browser_label,
    target_idx,
    tab_count,
    score,
    hold_progress,
    screen_w,
    screen_h,
    paused=False,
):
    canvas = np.zeros((180, 980, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)
    if paused:
        headline = "Focus Chromium Window"
        sub = f"Game paused. Window: {browser_label}"
    else:
        headline = f"LOOK AT TAB {int(target_idx) + 1}/{int(tab_count)}"
        sub = f"Title: {target_title}"

    cv2.putText(
        canvas,
        headline,
        (22, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.10,
        (0, 0, 255),
        4,
    )
    cv2.putText(
        canvas,
        sub,
        (22, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (0, 0, 255),
        3,
    )
    cv2.putText(
        canvas,
        f"Window: {browser_label}",
        (22, 146),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (210, 210, 210),
        1,
    )
    cv2.putText(
        canvas,
        f"Score: {int(score)}   Hold: {hold_progress:.2f}/{GAME_HIT_SECONDS:.2f}s",
        (22, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (210, 210, 210),
        1,
    )
    cv2.imshow(GAME_PROMPT_WIN, canvas)
    x = max(0, (int(screen_w) - canvas.shape[1]) // 2)
    y = max(0, (int(screen_h) - canvas.shape[0]) // 2)
    cv2.moveWindow(GAME_PROMPT_WIN, x, y)
    if not getattr(_show_game_prompt_window, "_opaque_applied", False):
        if _force_window_opaque(GAME_PROMPT_WIN):
            _show_game_prompt_window._opaque_applied = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-file", type=str, default=TAB_SETTINGS_FILE)
    ap.add_argument("--show-camera", action="store_true")
    ap.add_argument("--game-mode", action="store_true", help="show camera+points and a center-screen target-title prompt")
    ap.add_argument("--show-status-window", action="store_true")
    ap.add_argument("--allow-any-window", action="store_true")
    ap.add_argument("--tabs", type=int, default=0, help="optional runtime tab count for local status mapping")
    ap.add_argument("--disable-switch-publisher", action="store_true")
    ap.add_argument("--switch-publisher-port", type=int, default=DEFAULT_SWITCH_PORT)
    ap.add_argument("--disable-browser-overlay", action="store_true")
    ap.add_argument("--browser-overlay-port", type=int, default=DEFAULT_OVERLAY_PORT)
    args = ap.parse_args()

    if not os.path.exists(args.model_file):
        raise SystemExit(
            f"Tab model not found at {args.model_file}. Run: uv run python calibration.py"
        )
    model = load_tab_model(args.model_file)
    model_tab_count = int(model.get("calibration_tab_count") or model.get("tab_count") or len(model.get("centers", {})))
    if model_tab_count < 2:
        raise SystemExit("Invalid tab model; recalibrate with calibration.py")
    runtime_tab_count = int(args.tabs) if int(args.tabs) >= 2 else model_tab_count

    monitor = get_monitors()[0]
    sw = monitor.width
    sh = monitor.height
    if args.allow_any_window:
        _ = tab_rectangles(sw, TOP_H, runtime_tab_count)
    else:
        if args.game_mode:
            ok, focus_err = _focus_game_browser_prefer_chrome()
        else:
            ok, focus_err = focus_chromium_family_window()
        if not ok:
            raise SystemExit(focus_err)
        _, err = chromium_tab_rectangles(sw, TOP_H, runtime_tab_count)
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
    vote_hist = deque(maxlen=9)
    win = "Tab Tracker Status"
    last_conf = 0.0
    last_reported = None
    smoothed_pos = None
    stable_active = None
    game_tab_titles = None
    game_target_idx = None
    game_hit_t0 = None
    game_score = 0
    game_window_id = None
    game_window_label = "unknown"
    game_browser_hint = "google-chrome"
    switch_publisher = None
    overlay_publisher = None
    if not args.disable_switch_publisher:
        try:
            switch_publisher = SwitchStatePublisher(args.switch_publisher_port)
            switch_publisher.start()
            switch_publisher.update(
                running=True,
                tab_count=int(runtime_tab_count),
                browser_active=True,
                predicted_position=0.0,
            )
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
    if args.game_mode and (not args.disable_browser_overlay):
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
                f"continuing without tab red-dot overlay ({exc})"
            )
            overlay_publisher = None

    try:
        while True:
            browser_active = True
            if not args.allow_any_window:
                browser_active = is_active_chromium_family_window()
            current_win_id = active_window_id()
            current_win = active_window_geometry()
            if args.game_mode and browser_active and game_window_id is None and current_win_id:
                game_window_id = str(current_win_id)
                game_window_label = _clean_browser_title((current_win or {}).get("title", ""))
                klass = str((current_win or {}).get("class", "")).lower()
                if "brave" in klass:
                    game_browser_hint = "brave"
                elif "google-chrome" in klass or "chromium" in klass:
                    game_browser_hint = "google-chrome"

            # In game mode, progression is locked to the initially active browser window.
            game_window_active = True
            if args.game_mode and game_window_id is not None:
                game_window_active = bool(current_win_id and str(current_win_id) == str(game_window_id))

            ret, frame = cap.read()
            if not ret:
                continue
            paused_for_window = (not browser_active) or (not game_window_active)
            if paused_for_window:
                if switch_publisher is not None:
                    switch_publisher.update(
                        running=True,
                        predicted_tab=0,
                        predicted_position=0.0,
                        confidence=0.0,
                        tab_count=int(runtime_tab_count),
                        browser_active=False,
                    )
                if overlay_publisher is not None:
                    overlay_publisher.update(
                        enabled=False,
                        target_tab=0,
                        phase="paused",
                        browser_hint=game_browser_hint,
                    )
                if args.show_status_window:
                    disp = np.zeros((160, 420, 3), dtype=np.uint8)
                    _draw_status(disp, None, 0.0)
                    cv2.imshow(win, disp)
                    cv2.moveWindow(win, 16, 16)
                if args.show_camera:
                    cv2.imshow("Tab Tracker Camera", frame)
                if args.game_mode:
                    _show_debug_camera_window(frame, None, sw, sh)
                    _show_game_prompt_window(
                        target_title="-",
                        browser_label=game_window_label,
                        target_idx=0,
                        tab_count=runtime_tab_count,
                        score=game_score,
                        hold_progress=0.0,
                        screen_w=sw,
                        screen_h=sh,
                        paused=True,
                    )
                if args.show_status_window or args.show_camera or args.game_mode:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                continue
            result = tracker.process(frame, apply_head_comp=True)
            active = stable_active
            landmarks = None
            if result is not None:
                (h, v), landmarks = result
                pred, conf, _ = predict_tab(model, h, v, tab_count=runtime_tab_count)
                if pred is not None:
                    pred_pos = predict_tab_position(model, h, v)
                    if pred_pos is None:
                        pred_pos = float((int(pred) + 0.5) / float(runtime_tab_count))
                    if smoothed_pos is None:
                        smoothed_pos = float(pred_pos)
                    else:
                        smoothed_pos = float(
                            (POSITION_EMA_ALPHA * float(pred_pos))
                            + ((1.0 - POSITION_EMA_ALPHA) * float(smoothed_pos))
                        )
                    smooth_idx = int(np.clip(smoothed_pos * runtime_tab_count, 0, runtime_tab_count - 1))
                    if conf >= MIN_CONFIDENCE_FOR_SWITCH:
                        vote_hist.append(int(smooth_idx))
                    elif stable_active is not None:
                        vote_hist.append(int(stable_active))
                    if vote_hist:
                        vals, counts = np.unique(np.asarray(vote_hist, dtype=int), return_counts=True)
                        best_i = int(np.argmax(counts))
                        voted_idx = int(vals[best_i])
                        voted_ratio = float(counts[best_i]) / float(len(vote_hist))
                        if stable_active is None or voted_idx == stable_active:
                            stable_active = int(voted_idx)
                        elif voted_ratio >= SWITCH_MAJORITY_THRESHOLD and conf >= MIN_CONFIDENCE_FOR_SWITCH:
                            stable_active = int(voted_idx)
                    active = stable_active
                    last_conf = conf
                    publish_pos = float(smoothed_pos if smoothed_pos is not None else pred_pos)
                    if switch_publisher is not None:
                        switch_publisher.update(
                            running=True,
                            predicted_tab=int(active) + 1 if active is not None else 0,
                            predicted_position=float(publish_pos),
                            confidence=float(last_conf),
                            tab_count=int(runtime_tab_count),
                            browser_active=True,
                        )
                    if active is not None and active != last_reported:
                        print(f"Predicted tab={active + 1} confidence={last_conf:.2f}")
                        last_reported = active

            if args.game_mode:
                if game_tab_titles is None and game_window_active:
                    game_tab_titles = _collect_tab_titles(runtime_tab_count)
                if game_target_idx is None:
                    game_target_idx = _choose_next_game_target(runtime_tab_count, current_target=None)
                if overlay_publisher is not None and game_target_idx is not None:
                    overlay_publisher.update(
                        enabled=True,
                        target_tab=int(game_target_idx) + 1,
                        trial_index=int(game_score) + 1,
                        trial_total=0,
                        phase="game",
                        auto_activate=False,
                        mark_target_only=True,
                        browser_hint=game_browser_hint,
                    )
                if active is not None and int(active) == int(game_target_idx):
                    if game_hit_t0 is None:
                        game_hit_t0 = time.monotonic()
                    hold_for = float(time.monotonic() - game_hit_t0)
                    if hold_for >= GAME_HIT_SECONDS:
                        game_score += 1
                        game_target_idx = _choose_next_game_target(runtime_tab_count, current_target=game_target_idx)
                        game_hit_t0 = None
                else:
                    game_hit_t0 = None

            if args.show_status_window:
                disp = np.zeros((160, 420, 3), dtype=np.uint8)
                _draw_status(disp, active, last_conf)
                cv2.imshow(win, disp)
                cv2.moveWindow(win, 16, 16)
            if args.show_camera:
                cv2.imshow("Tab Tracker Camera", frame)
            if args.game_mode:
                _show_debug_camera_window(frame, landmarks, sw, sh)
                current_hold = 0.0 if game_hit_t0 is None else float(time.monotonic() - game_hit_t0)
                current_hold = float(min(current_hold, GAME_HIT_SECONDS))
                target_title = "-"
                if game_tab_titles is not None and game_target_idx is not None:
                    target_title = str(game_tab_titles[int(game_target_idx)])
                _show_game_prompt_window(
                    target_title=target_title,
                    browser_label=game_window_label,
                    target_idx=(game_target_idx or 0),
                    tab_count=runtime_tab_count,
                    score=game_score,
                    hold_progress=current_hold,
                    screen_w=sw,
                    screen_h=sh,
                    paused=False,
                )

            if args.show_status_window or args.show_camera or args.game_mode:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        if overlay_publisher is not None:
            overlay_publisher.update(
                enabled=False,
                target_tab=0,
                phase="idle",
                browser_hint=game_browser_hint,
            )
            overlay_publisher.stop()
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
        try:
            cv2.destroyWindow(DEBUG_CAMERA_WIN)
        except cv2.error:
            pass
        try:
            cv2.destroyWindow(GAME_PROMPT_WIN)
        except cv2.error:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
