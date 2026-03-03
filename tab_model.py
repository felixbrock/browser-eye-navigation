#!/usr/bin/env python3
"""Shared model utilities for tab-focused gaze classification."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone

import numpy as np


TAB_SETTINGS_FILE = os.path.expanduser("~/.config/gaze_tab_settings.json")
CHROMIUM_TITLE_TOKENS = ("chromium", "brave", "chrome")
BROWSER_CLASS_PATTERNS = (
    "Brave-browser",
    "brave-browser",
    "Google-chrome",
    "google-chrome",
    "Chromium",
    "chromium",
)
I3_FOCUS_CLASS_PATTERNS = (
    "brave-browser",
    "Brave-browser",
    "google-chrome",
    "Google-chrome",
    "chromium",
    "Chromium",
)
I3_FOCUS_APP_IDS = (
    "brave-browser",
    "google-chrome",
    "chromium",
)


def fit_tab_model(samples_by_tab):
    """Fit centroid + pooled covariance model from per-tab iris samples."""
    centers = {}
    all_points = []
    total_samples = 0
    for tab_id, samples in samples_by_tab.items():
        pts = np.asarray(samples, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 6:
            continue
        center = np.median(pts, axis=0)
        centers[int(tab_id)] = [float(center[0]), float(center[1])]
        all_points.append(pts)
        total_samples += int(len(pts))

    if len(centers) < 2:
        return None

    joined = np.vstack(all_points)
    cov = np.cov(joined.T) + np.eye(2) * 1e-4
    inv_cov = np.linalg.pinv(cov)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tab_count": len(centers),
        "total_samples": total_samples,
        "centers": {str(k): v for k, v in sorted(centers.items())},
        "inv_cov": inv_cov.tolist(),
        "cov": cov.tolist(),
    }


def predict_tab(model, h, v):
    """Return (tab_id, confidence, score_map)."""
    centers = model.get("centers", {})
    if not centers:
        return None, 0.0, {}
    x = np.asarray([float(h), float(v)], dtype=float)
    inv_cov = np.asarray(model.get("inv_cov"), dtype=float)
    if inv_cov.shape != (2, 2):
        inv_cov = np.eye(2, dtype=float)

    scores = {}
    for tab_id, center in centers.items():
        c = np.asarray(center, dtype=float)
        d = x - c
        m2 = float(d.T @ inv_cov @ d)
        scores[int(tab_id)] = m2

    best_tab = min(scores.keys(), key=lambda k: scores[k])
    sorted_scores = sorted(scores.values())
    if len(sorted_scores) >= 2:
        margin = sorted_scores[1] - sorted_scores[0]
    else:
        margin = 0.0
    confidence = float(1.0 - np.exp(-max(0.0, margin)))
    return best_tab, float(np.clip(confidence, 0.0, 1.0)), scores


def save_tab_model(model, path=TAB_SETTINGS_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_tab_model(path=TAB_SETTINGS_FILE):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tab_rectangles(screen_w, top_h, tab_count, margin=24, gap=8, origin_x=0, origin_y=0, width=None):
    area_w = int(width if width is not None else screen_w)
    usable_w = max(1, area_w - 2 * margin - gap * (tab_count - 1))
    tab_w = max(60, usable_w // max(tab_count, 1))
    rects = []
    x = origin_x + margin
    y_top = origin_y + 18
    for idx in range(tab_count):
        x1 = x
        x2 = min(origin_x + area_w - margin, x1 + tab_w)
        rects.append((x1, y_top, x2, y_top + top_h))
        x = x2 + gap
    return rects


def active_window_geometry():
    """Return active window metadata from xdotool, or None."""
    if shutil.which("xdotool") is None:
        return None
    try:
        class_proc = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowclassname"],
            check=True,
            capture_output=True,
            text=True,
        )
        title_proc = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowname"],
            check=True,
            capture_output=True,
            text=True,
        )
        geom_proc = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowgeometry", "--shell"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    klass = class_proc.stdout.strip()
    title = title_proc.stdout.strip()
    fields = {}
    for line in geom_proc.stdout.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        fields[k.strip().upper()] = v.strip()
    try:
        x = int(fields.get("X", "0"))
        y = int(fields.get("Y", "0"))
        w = int(fields.get("WIDTH", "0"))
        h = int(fields.get("HEIGHT", "0"))
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return {"title": title, "class": klass, "x": x, "y": y, "width": w, "height": h}


def is_active_chromium_family_window():
    """True if the currently active window appears to be Chromium-family."""
    win = active_window_geometry()
    if win is None:
        return False
    title = str(win.get("title", "")).lower()
    klass = str(win.get("class", "")).lower()
    return any(tok in title or tok in klass for tok in CHROMIUM_TITLE_TOKENS)


def active_window_id():
    """Return active X11 window id as string, or None."""
    if shutil.which("xdotool") is None:
        return None
    try:
        proc = subprocess.run(
            ["xdotool", "getactivewindow"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    win_id = proc.stdout.strip()
    return win_id or None


def focus_chromium_family_window():
    """Focus a visible Brave/Chrome/Chromium window (i3 first, xdotool fallback)."""

    def _is_chromium_active():
        win = active_window_geometry()
        if win is None:
            return False
        title = str(win.get("title", "")).lower()
        klass = str(win.get("class", "")).lower()
        return any(tok in title or tok in klass for tok in CHROMIUM_TITLE_TOKENS)

    def _iter_i3_nodes(node):
        yield node
        for child in node.get("nodes", []) or []:
            yield from _iter_i3_nodes(child)
        for child in node.get("floating_nodes", []) or []:
            yield from _iter_i3_nodes(child)

    # Prefer i3-native focusing when available.
    if shutil.which("i3-msg") is not None:
        # Fast path: class criteria focus command(s).
        for klass in I3_FOCUS_CLASS_PATTERNS:
            try:
                subprocess.run(
                    ["i3-msg", f'[class="{klass}"] focus'],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                time.sleep(0.08)
                if _is_chromium_active():
                    return True, None
            except Exception:
                continue
        # Robust path: inspect i3 tree and focus exact container id.
        try:
            tree_proc = subprocess.run(
                ["i3-msg", "-t", "get_tree"],
                check=True,
                capture_output=True,
                text=True,
            )
            tree = json.loads(tree_proc.stdout)
            candidates = []
            for node in _iter_i3_nodes(tree):
                con_id = node.get("id")
                if not con_id:
                    continue
                name = str(node.get("name") or "").lower()
                app_id = str(node.get("app_id") or "").lower()
                win_props = node.get("window_properties") or {}
                wclass = str(win_props.get("class") or "").lower()
                if (
                    any(p.lower() == app_id for p in I3_FOCUS_APP_IDS)
                    or any(p.lower() == wclass for p in I3_FOCUS_CLASS_PATTERNS)
                    or any(tok in name or tok in wclass or tok in app_id for tok in CHROMIUM_TITLE_TOKENS)
                ):
                    candidates.append(str(con_id))
            for con_id in candidates:
                subprocess.run(
                    ["i3-msg", f'[con_id="{con_id}"] focus'],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                time.sleep(0.08)
                if _is_chromium_active():
                    return True, None
        except Exception:
            pass

    if shutil.which("xdotool") is None:
        return False, "Neither i3-msg focus nor xdotool could focus a Chromium-family window."
    for pattern in BROWSER_CLASS_PATTERNS:
        try:
            proc = subprocess.run(
                ["xdotool", "search", "--class", pattern],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            continue
        ids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if not ids:
            continue
        win_id = ids[-1]
        try:
            subprocess.run(["xdotool", "windowactivate", "--sync", win_id], check=True)
            time.sleep(0.08)
            if _is_chromium_active():
                return True, None
        except subprocess.CalledProcessError:
            continue
    return False, "Could not focus a Brave/Google Chrome/Chromium window (i3-msg + xdotool tried)."


def select_chromium_tab(tab_idx):
    """Select Chromium tab by active window using Ctrl+1..8 (Ctrl+9 for last tab)."""
    win_id = active_window_id()
    if win_id is None:
        return False
    slot = int(tab_idx) + 1
    slot = min(max(slot, 1), 9)
    key = f"ctrl+{slot}"
    try:
        subprocess.run(
            ["xdotool", "key", "--window", str(win_id), key],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def chromium_tab_rectangles(screen_w, top_h, tab_count):
    """Build tab rectangles aligned to active Chromium-family window."""
    win = active_window_geometry()
    if win is None:
        return None, "Could not query active window via xdotool."
    title = str(win.get("title", "")).lower()
    klass = str(win.get("class", "")).lower()
    if not any(tok in title or tok in klass for tok in CHROMIUM_TITLE_TOKENS):
        return None, f'Active window is not Chromium-family: "{win.get("title", "")}" ({win.get("class", "")})'
    rects = tab_rectangles(
        screen_w=screen_w,
        top_h=top_h,
        tab_count=tab_count,
        origin_x=int(win["x"]),
        origin_y=int(win["y"]),
        width=int(win["width"]),
    )
    return rects, None
