#!/usr/bin/env python3
"""Shared model utilities for tab-focused gaze classification."""

from __future__ import annotations

import json
import os
import random
import shutil
import signal
import subprocess
import tempfile
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
DEFAULT_SESSION_URL = "https://en.wikipedia.org/wiki/Eye_tracking"
DEFAULT_SESSION_MIN_TABS = 3
DEFAULT_SESSION_MAX_TABS = 20


def chromium_browser_binary():
    """Return available Chromium-family browser binary, or None."""
    for name in (
        "brave",
        "brave-browser",
        "google-chrome-stable",
        "google-chrome",
        "chromium",
        "chromium-browser",
    ):
        if shutil.which(name):
            return name
    return None


def launch_managed_chromium_session(
    *,
    session_url=DEFAULT_SESSION_URL,
    min_tabs=DEFAULT_SESSION_MIN_TABS,
    max_tabs=DEFAULT_SESSION_MAX_TABS,
    load_overlay_extension=False,
    overlay_extension_dir=None,
):
    """Launch isolated Chromium-family browser instance with random tab count."""
    browser = chromium_browser_binary()
    if browser is None:
        return None, "Could not find Brave/Chrome/Chromium binary on PATH."
    low = int(max(2, min_tabs))
    high = int(max(low, max_tabs))
    tab_count = random.randint(low, high)
    profile_dir = tempfile.mkdtemp(prefix="gaze-tab-session-")

    args = [
        browser,
        "--new-window",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={profile_dir}",
    ]
    if load_overlay_extension and overlay_extension_dir and os.path.isdir(overlay_extension_dir):
        ext = os.path.abspath(overlay_extension_dir)
        args.append(f"--disable-extensions-except={ext}")
        args.append(f"--load-extension={ext}")
    args.extend([session_url] * tab_count)
    try:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(profile_dir, ignore_errors=True)
        return None, f"Failed to launch browser session: {exc}"

    session = {
        "browser": browser,
        "profile_dir": profile_dir,
        "tab_count": int(tab_count),
        "url": session_url,
        "pid": int(proc.pid),
    }
    return session, None


def close_managed_chromium_session(session):
    """Close only the managed Chromium session and remove its profile dir."""
    if not session:
        return
    profile_dir = str(session.get("profile_dir") or "").strip()
    pid = int(session.get("pid") or 0)

    if pid > 0:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            except PermissionError:
                break
            time.sleep(0.25)
            try:
                os.kill(pid, 0)
            except OSError:
                break

    # Sweep any remaining processes tied to this profile only.
    if profile_dir:
        try:
            proc = subprocess.run(
                ["pgrep", "-f", profile_dir],
                check=False,
                capture_output=True,
                text=True,
            )
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rem_pid = int(line)
                except ValueError:
                    continue
                if rem_pid == os.getpid():
                    continue
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.kill(rem_pid, sig)
                    except OSError:
                        break
                    time.sleep(0.10)
                    try:
                        os.kill(rem_pid, 0)
                    except OSError:
                        break
        except Exception:
            pass
        shutil.rmtree(profile_dir, ignore_errors=True)


def focus_managed_chromium_session(session):
    """Focus the window belonging to a managed browser session."""
    if not session:
        return False, "No managed session provided."
    if shutil.which("xdotool") is None:
        return False, "xdotool is required to focus managed browser session."

    profile_dir = str(session.get("profile_dir") or "").strip()
    pids = []
    try:
        root_pid = int(session.get("pid") or 0)
    except (TypeError, ValueError):
        root_pid = 0
    if root_pid > 0:
        pids.append(root_pid)

    if profile_dir:
        try:
            proc = subprocess.run(
                ["pgrep", "-f", profile_dir],
                check=False,
                capture_output=True,
                text=True,
            )
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    p = int(line)
                except ValueError:
                    continue
                if p not in pids:
                    pids.append(p)
        except Exception:
            pass

    win_ids = []
    for pid in pids:
        try:
            proc = subprocess.run(
                ["xdotool", "search", "--onlyvisible", "--pid", str(pid)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            continue
        for line in proc.stdout.splitlines():
            w = line.strip()
            if w:
                win_ids.append(w)

    if not win_ids:
        return False, "Could not find managed browser window by PID."

    for win_id in reversed(win_ids):
        try:
            subprocess.run(
                ["xdotool", "windowactivate", "--sync", str(win_id)],
                check=True,
                capture_output=True,
                text=True,
            )
            time.sleep(0.08)
            if is_active_chromium_family_window():
                return True, None
        except subprocess.CalledProcessError:
            continue

    return False, "Could not activate managed browser window."


def fit_tab_model(samples_by_tab, calibration_tab_count=None):
    """Fit tab model from per-tab iris samples.

    The primary model predicts normalized horizontal tab position in [0, 1],
    which can be remapped to any runtime tab count.
    """
    centers = {}
    all_points = []
    total_samples = 0
    reg_rows = []
    reg_targets = []

    tab_ids = sorted(int(k) for k in samples_by_tab.keys())
    inferred_tab_count = int(max(tab_ids) + 1) if tab_ids else 0
    if calibration_tab_count is None:
        calibration_tab_count = inferred_tab_count
    calibration_tab_count = int(max(2, calibration_tab_count))

    for tab_id, samples in samples_by_tab.items():
        pts = np.asarray(samples, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 6:
            continue
        tab_idx = int(tab_id)
        center = np.median(pts, axis=0)
        centers[tab_idx] = [float(center[0]), float(center[1])]
        all_points.append(pts)
        total_samples += int(len(pts))
        # Model target as normalized tab-center coordinate for portability.
        target_pos = (tab_idx + 0.5) / float(calibration_tab_count)
        reg_rows.extend(np.column_stack([np.ones(len(pts)), pts]).tolist())
        reg_targets.extend([target_pos] * len(pts))

    if len(centers) < 2:
        return None

    joined = np.vstack(all_points)
    cov = np.cov(joined.T) + np.eye(2) * 1e-4
    inv_cov = np.linalg.pinv(cov)
    X = np.asarray(reg_rows, dtype=float)
    y = np.asarray(reg_targets, dtype=float)
    ridge = np.diag([1e-6, 1e-3, 1e-3])
    weights = np.linalg.pinv(X.T @ X + ridge) @ (X.T @ y)
    pred = X @ weights
    rmse = float(np.sqrt(np.mean((pred - y) ** 2))) if len(y) else 1.0

    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "continuous_tab_position_v1",
        "tab_count": len(centers),
        "calibration_tab_count": calibration_tab_count,
        "total_samples": total_samples,
        "centers": {str(k): v for k, v in sorted(centers.items())},
        "inv_cov": inv_cov.tolist(),
        "cov": cov.tolist(),
        "position_model": {
            "weights": [float(w) for w in weights.tolist()],
            "rmse": float(rmse),
        },
    }


def _predict_normalized_position(model, h, v):
    pm = model.get("position_model") or {}
    weights = np.asarray(pm.get("weights", []), dtype=float)
    if weights.shape != (3,):
        return None
    x = np.asarray([1.0, float(h), float(v)], dtype=float)
    pos = float(np.clip(float(x @ weights), 0.0, 1.0))
    return pos


def _resolve_runtime_tab_count(model, tab_count):
    if tab_count is not None:
        return max(2, int(tab_count))
    model_count = int(model.get("calibration_tab_count") or model.get("tab_count") or 0)
    if model_count >= 2:
        return model_count
    return max(2, len(model.get("centers", {})))


def predict_tab_position(model, h, v):
    """Return normalized tab position in [0, 1], or None for legacy models."""
    return _predict_normalized_position(model, h, v)


def predict_tab(model, h, v, tab_count=None):
    """Return (tab_id, confidence, score_map)."""
    norm_pos = _predict_normalized_position(model, h, v)
    if norm_pos is not None:
        runtime_tabs = _resolve_runtime_tab_count(model, tab_count)
        idx = int(norm_pos * runtime_tabs)
        idx = min(max(idx, 0), runtime_tabs - 1)
        slot_pos = (norm_pos * runtime_tabs) - (idx + 0.5)
        boundary_margin = max(0.0, 0.5 - abs(slot_pos)) * 2.0
        rmse = float((model.get("position_model") or {}).get("rmse", 0.12))
        rmse_factor = float(np.exp(-min(1.0, max(0.0, rmse * 8.0))))
        confidence = float(np.clip(boundary_margin * rmse_factor, 0.0, 1.0))
        centers = np.arange(runtime_tabs, dtype=float) + 0.5
        center_norm = centers / float(runtime_tabs)
        score_map = {int(i): float(abs(norm_pos - c)) for i, c in enumerate(center_norm)}
        return idx, confidence, score_map

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


def select_chromium_tab(tab_idx, tab_count=None):
    """Select Chromium tab by active window.

    Tabs 1..8 use direct shortcuts. Tabs >=9 use Ctrl+8 then step right.
    """
    win_id = active_window_id()
    if win_id is None:
        return False
    slot = max(1, int(tab_idx) + 1)

    try:
        if slot <= 8:
            subprocess.run(
                ["xdotool", "key", "--window", str(win_id), f"ctrl+{slot}"],
                check=True,
                capture_output=True,
                text=True,
            )
            return True

        # Chromium maps Ctrl+9 to "last tab", so anchor at a stable known tab.
        subprocess.run(
            ["xdotool", "key", "--window", str(win_id), "ctrl+8"],
            check=True,
            capture_output=True,
            text=True,
        )
        if slot == 8:
            return True

        # Move right from tab 8 for tab indexes >= 9.
        steps = max(0, slot - 8)
        for _ in range(steps):
            subprocess.run(
                ["xdotool", "key", "--window", str(win_id), "ctrl+Page_Down"],
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
