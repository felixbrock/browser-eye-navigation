#!/usr/bin/env python3
"""Collect browser tab click training data with webcam eye/head features."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from pynput import mouse
from PIL import ImageGrab
from screeninfo import get_monitors

from gaze_core import EyeTracker, WEBCAM_H, WEBCAM_INDEX, WEBCAM_W

CHROMIUM_TOKENS = ("brave", "chrome", "chromium")
TAB_STRIP_MAX_Y = 140
CLICK_MATCH_MAX_AGE_MS = 1200
EVENT_BACKLOOK_MS = 1200
DEFAULT_TAB_STRIP_LEFT_INSET_PX = 32.0
DEFAULT_TAB_STRIP_RIGHT_INSET_PX = 132.0
DEFAULT_TAB_STRIP_HEIGHT_PX = 36.0
PINNED_TAB_WIDTH_PX = 44.0
STRIP_CAPTURE_HEIGHT_PX = 52
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "train.jsonl"


def now_ms() -> int:
    return int(time.time() * 1000)


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sanitize_tab_strip(tab_strip):
    if not isinstance(tab_strip, dict):
        return None
    return {
        "left_px": _as_float(tab_strip.get("left_px")),
        "right_px": _as_float(tab_strip.get("right_px")),
        "width_px": _as_float(tab_strip.get("width_px")),
        "height_px": _as_float(tab_strip.get("height_px")),
        "right_inset_px": _as_float(tab_strip.get("right_inset_px")),
    }


def sanitize_tab_candidates(candidates):
    if not isinstance(candidates, list):
        return []

    sanitized = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        bounds_px = candidate.get("bounds_px") if isinstance(candidate.get("bounds_px"), dict) else {}
        bounds_norm = candidate.get("bounds_norm") if isinstance(candidate.get("bounds_norm"), dict) else {}
        sanitized.append(
            {
                "tab_id": _as_int(candidate.get("tab_id")),
                "index": _as_int(candidate.get("index")),
                "title": str(candidate.get("title") or ""),
                "is_active": bool(candidate.get("is_active", False)),
                "is_pinned": bool(candidate.get("is_pinned", False)),
                "bounds_px": {
                    "left": _as_float(bounds_px.get("left")),
                    "right": _as_float(bounds_px.get("right")),
                    "center": _as_float(bounds_px.get("center")),
                    "width": _as_float(bounds_px.get("width")),
                    "top": _as_float(bounds_px.get("top")),
                    "height": _as_float(bounds_px.get("height")),
                },
                "bounds_norm": {
                    "left": _as_float(bounds_norm.get("left")),
                    "right": _as_float(bounds_norm.get("right")),
                    "center": _as_float(bounds_norm.get("center")),
                    "width": _as_float(bounds_norm.get("width")),
                },
            }
        )
    return sanitized


def browser_hint_for_window(window, fallback_hint):
    text = f"{window.get('class') or ''} {window.get('title') or ''}".lower()
    if "brave" in text:
        return "brave"
    if "google-chrome" in text or "google chrome" in text:
        return "google-chrome"
    if "chrome" in text:
        return "google-chrome"
    if "chromium" in text:
        return "chromium"
    return fallback_hint


def _normalize_strip(strip, window_width: float):
    hint = sanitize_tab_strip(strip)
    if hint is None:
        right_inset = DEFAULT_TAB_STRIP_RIGHT_INSET_PX
        width = max(1.0, window_width - DEFAULT_TAB_STRIP_LEFT_INSET_PX - right_inset)
        return {
            "left_px": DEFAULT_TAB_STRIP_LEFT_INSET_PX,
            "right_px": DEFAULT_TAB_STRIP_LEFT_INSET_PX + width,
            "width_px": width,
            "height_px": DEFAULT_TAB_STRIP_HEIGHT_PX,
            "right_inset_px": right_inset,
        }

    left = _as_float(hint.get("left_px")) or DEFAULT_TAB_STRIP_LEFT_INSET_PX
    right = _as_float(hint.get("right_px"))
    width = _as_float(hint.get("width_px"))
    if right is None and width is not None:
        right = left + width
    if right is None:
        right = max(left + 1.0, window_width - DEFAULT_TAB_STRIP_RIGHT_INSET_PX)
    right = min(max(right, left + 1.0), window_width)
    width = max(1.0, right - left)
    return {
        "left_px": left,
        "right_px": right,
        "width_px": width,
        "height_px": _as_float(hint.get("height_px")) or DEFAULT_TAB_STRIP_HEIGHT_PX,
        "right_inset_px": max(0.0, window_width - right),
    }


def _default_source_candidates(event):
    tab_count = max(1, _as_int(event.get("tab_count")) or 1)
    clicked_index = _as_int(event.get("tab_index")) or 0
    clicked_title = str(event.get("tab_title") or "")
    clicked_tab_id = _as_int(event.get("tab_id"))
    return [
        {
            "tab_id": clicked_tab_id if index == clicked_index else index,
            "index": index,
            "title": clicked_title if index == clicked_index else "",
            "is_active": index == clicked_index,
            "is_pinned": False,
            "bounds_px": {"left": None, "right": None, "center": None, "width": None, "top": 0.0, "height": DEFAULT_TAB_STRIP_HEIGHT_PX},
            "bounds_norm": {"left": None, "right": None, "center": None, "width": None},
        }
        for index in range(tab_count)
    ]


def capture_strip_image(window):
    if not window:
        return None
    width = _as_int(window.get("width"))
    height = _as_int(window.get("height"))
    x = _as_int(window.get("x"))
    y = _as_int(window.get("y"))
    if None in (width, height, x, y) or width <= 0 or height <= 0:
        return None
    bbox = (x, y, x + width, y + min(height, TAB_STRIP_MAX_Y))
    try:
        image = ImageGrab.grab(bbox=bbox)
    except Exception:
        return None
    if image is None:
        return None
    return np.asarray(image.convert("RGB"))


def detect_strip_bounds_from_image(image, hint_strip, window_width: float):
    strip = dict(hint_strip)
    if image is None or image.size == 0:
        return strip, False

    gray = cv2.cvtColor(image[: min(image.shape[0], STRIP_CAPTURE_HEIGHT_PX)], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 32, 96)
    col_energy = edges.mean(axis=0).astype(np.float32)
    if not len(col_energy):
        return strip, False

    smooth = np.convolve(col_energy, np.ones(17, dtype=np.float32) / 17.0, mode="same")
    threshold = max(2.0, float(np.percentile(smooth, 65)))
    active = np.flatnonzero(smooth >= threshold)
    if len(active) < 8:
        return strip, False

    detected_left = max(0.0, float(active[0]) - 4.0)
    detected_right = min(window_width, float(active[-1]) + 4.0)

    if detected_right - detected_left < 200.0:
        return strip, False

    hint_left = _as_float(strip.get("left_px")) or DEFAULT_TAB_STRIP_LEFT_INSET_PX
    hint_right = _as_float(strip.get("right_px")) or max(hint_left + 1.0, window_width - DEFAULT_TAB_STRIP_RIGHT_INSET_PX)

    strip["left_px"] = max(0.0, min(hint_left, detected_left + 24.0))
    strip["right_px"] = min(window_width, max(hint_right, detected_right - 40.0))
    strip["width_px"] = max(1.0, strip["right_px"] - strip["left_px"])
    strip["right_inset_px"] = max(0.0, window_width - strip["right_px"])
    return strip, True


def build_candidates_from_strip(source_candidates, strip):
    candidates = []
    ordered = sorted(source_candidates, key=lambda item: (_as_int(item.get("index")) or 0, _as_int(item.get("tab_id")) or 0))
    pinned_count = sum(1 for candidate in ordered if candidate.get("is_pinned"))
    normal_count = max(0, len(ordered) - pinned_count)
    strip_left = _as_float(strip.get("left_px")) or DEFAULT_TAB_STRIP_LEFT_INSET_PX
    strip_width = max(1.0, _as_float(strip.get("width_px")) or 1.0)
    strip_height = _as_float(strip.get("height_px")) or DEFAULT_TAB_STRIP_HEIGHT_PX
    pinned_width = PINNED_TAB_WIDTH_PX if pinned_count else 0.0
    normal_width = (
        max(1.0, (strip_width - (pinned_count * pinned_width)) / normal_count)
        if normal_count
        else 0.0
    )

    cursor = strip_left
    for candidate in ordered:
        width = pinned_width if candidate.get("is_pinned") else normal_width
        width = max(1.0, width)
        left = cursor
        right = left + width
        center = left + (width * 0.5)
        cursor = right
        candidates.append(
            {
                "tab_id": _as_int(candidate.get("tab_id")),
                "index": _as_int(candidate.get("index")),
                "title": str(candidate.get("title") or ""),
                "is_active": bool(candidate.get("is_active", False)),
                "is_pinned": bool(candidate.get("is_pinned", False)),
                "bounds_px": {
                    "left": float(left),
                    "right": float(right),
                    "center": float(center),
                    "width": float(width),
                    "top": 0.0,
                    "height": float(strip_height),
                },
                "bounds_norm": {
                    "left": float((left - strip_left) / strip_width),
                    "right": float((right - strip_left) / strip_width),
                    "center": float((center - strip_left) / strip_width),
                    "width": float(width / strip_width),
                },
            }
        )
    return candidates


def shift_layout(candidates, strip, shift_px):
    if not candidates or abs(shift_px) < 1e-3:
        return candidates, strip, 0.0

    min_shift = -min(candidate["bounds_px"]["left"] for candidate in candidates)
    max_shift = _as_float(strip.get("right_inset_px"))
    if max_shift is None:
        max_shift = 0.0
    applied = float(np.clip(shift_px, min_shift, max_shift))

    shifted = []
    for candidate in candidates:
        bounds_px = dict(candidate["bounds_px"])
        bounds_px["left"] += applied
        bounds_px["right"] += applied
        bounds_px["center"] += applied
        shifted.append({**candidate, "bounds_px": bounds_px, "bounds_norm": dict(candidate["bounds_norm"])})

    shifted_strip = dict(strip)
    shifted_strip["left_px"] = (_as_float(strip.get("left_px")) or 0.0) + applied
    shifted_strip["right_px"] = (_as_float(strip.get("right_px")) or 0.0) + applied
    shifted_strip["width_px"] = max(1.0, shifted_strip["right_px"] - shifted_strip["left_px"])
    shifted_strip["right_inset_px"] = max(0.0, shifted_strip["right_inset_px"] - applied if strip.get("right_inset_px") is not None else 0.0)

    strip_left = shifted_strip["left_px"]
    strip_width = max(1.0, shifted_strip["width_px"])
    for candidate in shifted:
        bounds_px = candidate["bounds_px"]
        candidate["bounds_norm"] = {
            "left": float((bounds_px["left"] - strip_left) / strip_width),
            "right": float((bounds_px["right"] - strip_left) / strip_width),
            "center": float((bounds_px["center"] - strip_left) / strip_width),
            "width": float(bounds_px["width"] / strip_width),
        }
    return shifted, shifted_strip, applied


def compute_geometry_metrics(candidates, clicked_index, click_window_x):
    clicked_candidate = next((candidate for candidate in candidates if candidate.get("index") == clicked_index), None)
    nearest_candidate = None
    if candidates:
        nearest_candidate = min(
            candidates,
            key=lambda candidate: abs(click_window_x - (candidate["bounds_px"]["center"] or 0.0)),
        )

    if clicked_candidate is None:
        return {
            "click_window_x": click_window_x,
            "click_inside_reported_tab": False,
            "nearest_tab_index": nearest_candidate.get("index") if nearest_candidate else None,
            "click_offset_px": None,
        }

    bounds = clicked_candidate["bounds_px"]
    inside = bool(bounds["left"] <= click_window_x <= bounds["right"])
    return {
        "click_window_x": click_window_x,
        "click_inside_reported_tab": inside,
        "nearest_tab_index": nearest_candidate.get("index") if nearest_candidate else None,
        "click_offset_px": float(click_window_x - bounds["center"]),
    }


def build_collector_layout(window, click, event, source_candidates, source_strip):
    window_width = float(_as_float(window.get("width")) or 1.0)
    strip = _normalize_strip(source_strip, window_width)
    screen_capture = capture_strip_image(window)
    strip, measured_from_image = detect_strip_bounds_from_image(screen_capture, strip, window_width)
    candidates = build_candidates_from_strip(source_candidates, strip)

    clicked_index = _as_int(event.get("tab_index"))
    click_window_x = float((_as_int(click.get("desktop_x")) or 0) - (_as_int(window.get("x")) or 0))
    geometry_source = "measured" if measured_from_image else "extension"
    alignment_shift_px = 0.0

    if candidates and clicked_index is not None:
        clicked_candidate = next((candidate for candidate in candidates if candidate.get("index") == clicked_index), None)
        if clicked_candidate is not None:
            left = clicked_candidate["bounds_px"]["left"]
            right = clicked_candidate["bounds_px"]["right"]
            if not (left <= click_window_x <= right):
                alignment_shift = click_window_x - clicked_candidate["bounds_px"]["center"]
                candidates, strip, alignment_shift_px = shift_layout(candidates, strip, alignment_shift)

    geometry = compute_geometry_metrics(candidates, clicked_index, click_window_x)
    geometry.update(
        {
            "source": geometry_source,
            "screen_capture_succeeded": bool(screen_capture is not None),
            "image_measurement_succeeded": bool(measured_from_image),
            "alignment_shift_px": float(alignment_shift_px),
        }
    )
    return strip, candidates, geometry


def run_command(args):
    try:
        proc = subprocess.run(args, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return proc.stdout


def active_window_snapshot():
    if shutil.which("xdotool") is None:
        return None

    win_id = run_command(["xdotool", "getactivewindow"])
    klass = run_command(["xdotool", "getactivewindow", "getwindowclassname"])
    title = run_command(["xdotool", "getactivewindow", "getwindowname"])
    geom = run_command(["xdotool", "getactivewindow", "getwindowgeometry", "--shell"])
    if not win_id or not klass or not geom:
        return None

    fields = {}
    for line in geom.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip().upper()] = value.strip()

    try:
        x = int(fields.get("X", "0"))
        y = int(fields.get("Y", "0"))
        width = int(fields.get("WIDTH", "0"))
        height = int(fields.get("HEIGHT", "0"))
    except ValueError:
        return None

    browser_class = klass.strip()
    browser_title = (title or "").strip()
    lowered = f"{browser_class} {browser_title}".lower()
    return {
        "x11_window_id": win_id.strip(),
        "class": browser_class,
        "title": browser_title,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "is_chromium": any(token in lowered for token in CHROMIUM_TOKENS),
    }


def monitor_for_point(x: int, y: int):
    for mon in get_monitors():
        mon_x = getattr(mon, "x", 0)
        mon_y = getattr(mon, "y", 0)
        if mon_x <= x < mon_x + mon.width and mon_y <= y < mon_y + mon.height:
            return {
                "x": int(mon_x),
                "y": int(mon_y),
                "width": int(mon.width),
                "height": int(mon.height),
                "name": getattr(mon, "name", None),
                "is_primary": bool(getattr(mon, "is_primary", False)),
            }
    return None


class CollectorServer:
    """Receive browser tab activation events from the unpacked extension."""

    def __init__(self, port: int, event_queue: queue.Queue):
        self.port = int(port)
        self.event_queue = event_queue
        self.server = None
        self.thread = None

    def start(self):
        event_queue = self.event_queue

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != "/tab-event":
                    self.send_response(404)
                    self.end_headers()
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length)
                    payload = json.loads(raw.decode("utf-8"))
                except Exception:
                    self.send_response(400)
                    self.end_headers()
                    return
                payload["received_ts_ms"] = now_ms()
                event_queue.put(payload)
                self.send_response(200)
                self.end_headers()

            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def end_headers(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                super().end_headers()

            def log_message(self, fmt, *args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        self.thread = None


class DataCollector:
    def __init__(self, args):
        self.args = args
        self.event_queue: queue.Queue = queue.Queue()
        self.server = CollectorServer(args.port, self.event_queue)
        self.sample_buffer = deque()
        self.click_buffer = deque(maxlen=256)
        self.stop_event = threading.Event()
        self.output_fp = None
        self.mouse_listener = None

    def _trim_samples(self, current_ts_ms: int):
        min_ts = current_ts_ms - max(self.args.buffer_ms, EVENT_BACKLOOK_MS)
        while self.sample_buffer and self.sample_buffer[0]["timestamp_ms"] < min_ts:
            self.sample_buffer.popleft()

    def _trim_clicks(self, current_ts_ms: int):
        min_ts = current_ts_ms - EVENT_BACKLOOK_MS
        while self.click_buffer and self.click_buffer[0]["timestamp_ms"] < min_ts:
            self.click_buffer.popleft()

    def _on_click(self, x, y, button, pressed):
        if not pressed or button != mouse.Button.left:
            return
        ts_ms = now_ms()
        win = active_window_snapshot()
        monitor = monitor_for_point(int(x), int(y))
        local_x = int(x - monitor["x"]) if monitor else None
        local_y = int(y - monitor["y"]) if monitor else None
        click = {
            "timestamp_ms": ts_ms,
            "desktop_x": int(x),
            "desktop_y": int(y),
            "monitor": monitor,
            "click_x": local_x,
            "click_y": local_y,
            "window": win,
            "matched": False,
        }
        self.click_buffer.append(click)
        self._trim_clicks(ts_ms)

    def _start_mouse_listener(self):
        self.mouse_listener = mouse.Listener(on_click=self._on_click)
        self.mouse_listener.daemon = True
        self.mouse_listener.start()

    def _recent_samples_for(self, ts_ms: int):
        lower = ts_ms - self.args.buffer_ms
        return [sample for sample in self.sample_buffer if lower <= sample["timestamp_ms"] <= ts_ms]

    def _match_click(self, event):
        event_ts_ms = int(event.get("event_ts_ms") or event.get("received_ts_ms") or now_ms())
        self._trim_clicks(event_ts_ms)
        best = None
        best_age = None
        for click in reversed(self.click_buffer):
            if click["matched"]:
                continue
            age = event_ts_ms - click["timestamp_ms"]
            if age < 0 or age > CLICK_MATCH_MAX_AGE_MS:
                continue
            window = click.get("window") or {}
            if not window.get("is_chromium"):
                continue
            if click.get("click_y") is None or click["click_y"] > TAB_STRIP_MAX_Y:
                continue
            if best is None or age < best_age:
                best = click
                best_age = age
        if best is None:
            return None
        best["matched"] = True
        return best, int(best_age)

    def _entry_from_event(self, event, click, click_age_ms):
        ts_ms = int(event.get("event_ts_ms") or event.get("received_ts_ms") or now_ms())
        samples = self._recent_samples_for(click["timestamp_ms"])
        face_samples = sum(1 for sample in samples if sample.get("face_detected"))
        window = click.get("window") or {}
        monitor = click.get("monitor") or {}
        source_tab_strip = sanitize_tab_strip(event.get("tab_strip"))
        source_tab_candidates = sanitize_tab_candidates(event.get("tab_candidates")) or _default_source_candidates(event)
        tab_strip, tab_candidates, geometry = build_collector_layout(
            window,
            click,
            event,
            source_tab_candidates,
            source_tab_strip,
        )
        browser_hint = browser_hint_for_window(window, event.get("browser_hint"))

        entry = {
            "timestamp": ts_ms / 1000.0,
            "timestamp_ms": ts_ms,
            "browser_hint": browser_hint,
            "browser_window_id": {
                "chrome_window_id": event.get("chrome_window_id"),
                "x11_window_id": window.get("x11_window_id"),
            },
            "trigger": event.get("trigger"),
            "tab_event": {
                "tab_id": event.get("tab_id"),
                "clicked_tab_index": event.get("tab_index"),
                "tab_count": event.get("tab_count"),
                "clicked_tab_title": event.get("tab_title"),
            },
            "tab_strip": tab_strip,
            "tab_candidates": tab_candidates,
            "geometry": geometry,
            "monitor": {
                "x": monitor.get("x"),
                "y": monitor.get("y"),
                "width": monitor.get("width"),
                "height": monitor.get("height"),
                "name": monitor.get("name"),
                "is_primary": monitor.get("is_primary"),
            },
            "window": {
                "x": window.get("x"),
                "y": window.get("y"),
                "width": window.get("width"),
                "height": window.get("height"),
                "class": window.get("class"),
                "title": window.get("title"),
            },
            "click": {
                "desktop_x": click.get("desktop_x"),
                "desktop_y": click.get("desktop_y"),
                "x": click.get("click_x"),
                "y": click.get("click_y"),
            },
            "capture_quality": {
                "matched_click_age_ms": click_age_ms,
                "sample_count": len(samples),
                "face_sample_count": face_samples,
                "face_detected_recently": face_samples > 0,
                "top_strip_click": bool(click.get("click_y") is not None and click["click_y"] <= TAB_STRIP_MAX_Y),
                "geometry_source": geometry.get("source"),
                "geometry_click_inside_reported_tab": geometry.get("click_inside_reported_tab"),
            },
            "pre_click_samples": samples,
        }
        return entry

    def _handle_tab_event(self, event):
        matched = self._match_click(event)
        if matched is None:
            return
        click, click_age_ms = matched
        entry = self._entry_from_event(event, click, click_age_ms)
        self.output_fp.write(json.dumps(entry, separators=(",", ":")) + "\n")
        self.output_fp.flush()

    def _drain_events(self):
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                return
            self._handle_tab_event(event)

    def run(self):
        output_path = os.path.abspath(self.args.output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.output_fp = open(output_path, "a", encoding="utf-8", buffering=1)

        self.server.start()
        self._start_mouse_listener()

        tracker = EyeTracker()
        cap = cv2.VideoCapture(self.args.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_H)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {self.args.camera_index}")

        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                ts_ms = now_ms()
                if ok:
                    sample = tracker.sample(frame)
                    sample["timestamp_ms"] = ts_ms
                    self.sample_buffer.append(sample)
                    self._trim_samples(ts_ms)
                self._drain_events()
        finally:
            cap.release()
            tracker.close()
            self.server.stop()
            if self.mouse_listener is not None:
                self.mouse_listener.stop()
            if self.output_fp is not None:
                self.output_fp.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=WEBCAM_INDEX)
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--buffer-ms", type=int, default=800)
    return parser.parse_args()


def main():
    args = parse_args()
    collector = DataCollector(args)

    def _handle_signal(_signum, _frame):
        collector.stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    collector.run()


if __name__ == "__main__":
    main()
