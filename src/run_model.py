#!/usr/bin/env python3
"""Run live browser-tab inference for the focused browser window."""

from __future__ import annotations

import argparse
import json
import pickle
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

from gaze_core import EyeTracker, WEBCAM_H, WEBCAM_INDEX, WEBCAM_W
from train import aggregate_sequence_features, build_example_features

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "tab_ranker_v2.pkl"
DEFAULT_RUNTIME_PORT = 8768
DEFAULT_BUFFER_MS = 800
CHROMIUM_TOKENS = ("brave", "chrome", "chromium")
DEBUG_LOG_EVERY_REQUEST = True


def now_ms() -> int:
    return int(time.time() * 1000)


def _safe_int(value, default: int | None = None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float | None = None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
    if not win_id or not klass or not title or not geom:
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
    browser_title = title.strip()
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


def browser_family(label: str | None) -> str | None:
    text = str(label or "").lower()
    if "brave" in text:
        return "brave"
    if "chrome" in text:
        return "google-chrome"
    if "chromium" in text:
        return "chromium"
    return None


def debug_event(event: str, **fields):
    print(json.dumps({"event": event, **fields}))


def sanitize_layout(payload):
    tab_strip = payload.get("tab_strip") if isinstance(payload.get("tab_strip"), dict) else {}
    raw_candidates = payload.get("tab_candidates") if isinstance(payload.get("tab_candidates"), list) else []
    candidates = []
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue
        bounds_px = candidate.get("bounds_px") if isinstance(candidate.get("bounds_px"), dict) else {}
        bounds_norm = candidate.get("bounds_norm") if isinstance(candidate.get("bounds_norm"), dict) else {}
        candidates.append(
            {
                "tab_id": _safe_int(candidate.get("tab_id")),
                "index": _safe_int(candidate.get("index"), 0),
                "title": str(candidate.get("title") or ""),
                "is_active": bool(candidate.get("is_active", False)),
                "is_pinned": bool(candidate.get("is_pinned", False)),
                "bounds_px": {
                    "left": float(_safe_float(bounds_px.get("left"), 0.0)),
                    "right": float(_safe_float(bounds_px.get("right"), 0.0)),
                    "center": float(_safe_float(bounds_px.get("center"), 0.0)),
                    "width": float(_safe_float(bounds_px.get("width"), 1.0)),
                    "top": float(_safe_float(bounds_px.get("top"), 0.0)),
                    "height": float(_safe_float(bounds_px.get("height"), 36.0)),
                },
                "bounds_norm": {
                    "left": float(_safe_float(bounds_norm.get("left"), 0.0)),
                    "right": float(_safe_float(bounds_norm.get("right"), 0.0)),
                    "center": float(_safe_float(bounds_norm.get("center"), 0.0)),
                    "width": float(_safe_float(bounds_norm.get("width"), 0.0)),
                },
            }
        )
    return {
        "tab_strip": {
            "left_px": float(_safe_float(tab_strip.get("left_px"), 0.0)),
            "right_px": float(_safe_float(tab_strip.get("right_px"), 1.0)),
            "width_px": float(_safe_float(tab_strip.get("width_px"), 1.0)),
            "height_px": float(_safe_float(tab_strip.get("height_px"), 36.0)),
            "right_inset_px": float(_safe_float(tab_strip.get("right_inset_px"), 0.0)),
        },
        "tab_candidates": candidates,
    }


class SampleStore:
    def __init__(self, buffer_ms: int):
        self.buffer_ms = int(buffer_ms)
        self.samples = deque()
        self.lock = threading.Lock()

    def add(self, sample):
        with self.lock:
            self.samples.append(sample)
            min_ts = sample["timestamp_ms"] - self.buffer_ms
            while self.samples and self.samples[0]["timestamp_ms"] < min_ts:
                self.samples.popleft()

    def recent(self):
        with self.lock:
            if not self.samples:
                return []
            latest_ts = self.samples[-1]["timestamp_ms"]
            min_ts = latest_ts - self.buffer_ms
            return [sample for sample in self.samples if sample["timestamp_ms"] >= min_ts]


class RuntimeState:
    def __init__(self, args, model_bundle):
        self.args = args
        self.model = model_bundle["model"]
        self.feature_names = list(model_bundle["feature_names"])
        self.sample_store = SampleStore(args.buffer_ms)
        self.stop_event = threading.Event()
        self.request_count = 0

    def evaluate_request(self, payload):
        self.request_count += 1
        active_window = active_window_snapshot()
        if active_window is None or not active_window.get("is_chromium"):
            return self._response(
                reason="no_focused_browser_window",
                debug={"request_count": self.request_count},
            )

        if not self._matches_focused_window(payload, active_window):
            return self._response(
                reason="focused_window_mismatch",
                debug={
                    "request_count": self.request_count,
                    "active_window_title": active_window.get("title"),
                    "active_window_class": active_window.get("class"),
                    "payload_active_tab_title": payload.get("active_tab_title"),
                    "payload_browser_hint": payload.get("browser_hint"),
                },
            )

        samples = self.sample_store.recent()
        face_samples = [sample for sample in samples if sample.get("face_detected")]
        if len(face_samples) < self.args.min_face_samples:
            return self._response(
                reason="insufficient_face_samples",
                debug={
                    "request_count": self.request_count,
                    "face_samples": len(face_samples),
                    "min_face_samples": self.args.min_face_samples,
                },
            )

        layout = sanitize_layout(payload)
        candidates = layout["tab_candidates"]
        if not candidates:
            return self._response(
                reason="no_tab_candidates",
                debug={"request_count": self.request_count},
            )

        sequence = aggregate_sequence_features(samples)
        tab_strip = layout["tab_strip"]
        tab_count = max(1, len(candidates))
        x = np.asarray(
            [
                [build_example_features(sequence, candidate, tab_strip, tab_count)[name] for name in self.feature_names]
                for candidate in candidates
            ],
            dtype=np.float32,
        )
        probabilities = self.model.predict_proba(x)[:, 1]
        best_ix = int(np.argmax(probabilities))
        best_candidate = candidates[best_ix]
        best_score = float(probabilities[best_ix])
        active_tab_id = _safe_int(payload.get("active_tab_id"))

        return self._response(
            predicted_tab_id=best_candidate.get("tab_id"),
            predicted_index=best_candidate.get("index"),
            predicted_score=best_score,
            reason="predicted_only" if best_candidate.get("tab_id") != active_tab_id else "already_active",
            debug={
                "request_count": self.request_count,
                "active_tab_id": active_tab_id,
                "best_tab_id": best_candidate.get("tab_id"),
                "best_index": best_candidate.get("index"),
                "best_score": round(best_score, 4),
            },
        )

    def _response(
        self,
        predicted_tab_id=None,
        predicted_index=None,
        predicted_score=None,
        reason="noop",
        debug=None,
    ):
        response = {
            "predicted_tab_id": predicted_tab_id,
            "predicted_index": predicted_index,
            "predicted_score": predicted_score,
            "reason": reason,
        }
        if debug is not None:
            response["debug"] = debug
        if DEBUG_LOG_EVERY_REQUEST:
            debug_event("runtime_response", **response)
        return response

    def _matches_focused_window(self, payload, active_window):
        request_browser = browser_family(payload.get("browser_hint"))
        active_browser = browser_family(active_window.get("class"))

        request_window = payload.get("window") if isinstance(payload.get("window"), dict) else {}
        if not request_window.get("focused", False):
            return False

        geometry_score = 0
        if abs((_safe_int(request_window.get("left"), 0) or 0) - active_window["x"]) <= 64:
            geometry_score += 1
        if abs((_safe_int(request_window.get("top"), 0) or 0) - active_window["y"]) <= 96:
            geometry_score += 1
        if abs((_safe_int(request_window.get("width"), 0) or 0) - active_window["width"]) <= 128:
            geometry_score += 1
        if abs((_safe_int(request_window.get("height"), 0) or 0) - active_window["height"]) <= 128:
            geometry_score += 1

        title_score = 0
        active_title = str(active_window.get("title") or "").lower()
        payload_title = str(payload.get("active_tab_title") or "").lower()
        if payload_title and payload_title in active_title:
            title_score = 2

        browser_score = 0
        if request_browser is not None and active_browser is not None and request_browser == active_browser:
            browser_score = 2

        # Title + geometry should be enough to accept the match even when the extension mislabels Brave as Chrome.
        return (geometry_score + title_score + browser_score) >= 4


class RuntimeServer:
    def __init__(self, port: int, runtime_state: RuntimeState):
        self.port = int(port)
        self.runtime_state = runtime_state
        self.server = None
        self.thread = None

    def start(self):
        runtime_state = self.runtime_state

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != "/runtime-window-state":
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
                response = runtime_state.evaluate_request(payload)
                body = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--camera-index", type=int, default=WEBCAM_INDEX)
    parser.add_argument("--buffer-ms", type=int, default=DEFAULT_BUFFER_MS)
    parser.add_argument("--min-face-samples", type=int, default=5)
    return parser.parse_args()


def load_model_bundle(path: Path):
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict) or "model" not in bundle or "feature_names" not in bundle:
        raise ValueError(f"Unexpected model bundle shape in {path}")
    return bundle


def main():
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    model_bundle = load_model_bundle(model_path)
    runtime_state = RuntimeState(args, model_bundle)
    server = RuntimeServer(DEFAULT_RUNTIME_PORT, runtime_state)

    def handle_signal(_signum, _frame):
        runtime_state.stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    tracker = EyeTracker()
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_H)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera_index}")

    server.start()

    debug_event(
        "runtime_started",
        model_path=str(model_path),
        port=DEFAULT_RUNTIME_PORT,
    )

    try:
        while not runtime_state.stop_event.is_set():
            ok, frame = cap.read()
            ts_ms = now_ms()
            if not ok:
                continue
            sample = tracker.sample(frame)
            sample["timestamp_ms"] = ts_ms
            runtime_state.sample_store.add(sample)
    finally:
        cap.release()
        tracker.close()
        server.stop()


if __name__ == "__main__":
    main()
