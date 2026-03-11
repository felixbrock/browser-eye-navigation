#!/usr/bin/env python3
"""Webcam eye and head feature extraction for training-data collection."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

WEBCAM_INDEX = 0
WEBCAM_W = 640
WEBCAM_H = 480
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"

R_OUTER = 33
R_INNER = 133
R_TOP = 159
R_BOTTOM = 145
R_IRIS_RING = (469, 470, 471, 472)

L_INNER = 362
L_OUTER = 263
L_TOP = 386
L_BOTTOM = 374
L_IRIS_RING = (474, 475, 476, 477)

NOSE_TIP = 1
CHIN = 152
MOUTH_TOP = 13
MOUTH_BOTTOM = 14


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


class EyeTracker:
    """Extract webcam eye and head features using MediaPipe Face Landmarker."""

    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            print(f"ERROR: Model not found at {MODEL_PATH}")
            print("Download it with:")
            print(
                f'  curl -L -o "{MODEL_PATH}" '
                '"https://storage.googleapis.com/mediapipe-models/'
                'face_landmarker/face_landmarker/float16/latest/face_landmarker.task"'
            )
            sys.exit(1)

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._frame_ts_ms = 0
        self._head_anchor: tuple[float, float, float] | None = None

    def close(self) -> None:
        self.landmarker.close()

    def _iris_center(self, lm, ring_indices):
        xs = [lm[i].x for i in ring_indices]
        ys = [lm[i].y for i in ring_indices]
        return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))

    def _eye_ratio(self, lm, iris_ring, outer, inner, top, bottom):
        ix, iy = self._iris_center(lm, iris_ring)
        left_x = min(lm[outer].x, lm[inner].x)
        right_x = max(lm[outer].x, lm[inner].x)
        top_y = min(lm[top].y, lm[bottom].y)
        bottom_y = max(lm[top].y, lm[bottom].y)
        width = max(1e-7, right_x - left_x)
        height = max(1e-7, bottom_y - top_y)
        horizontal = (ix - left_x) / width
        vertical = (iy - top_y) / height
        openness = height / max(1e-7, width)
        return {
            "iris_center": {"x": float(ix), "y": float(iy)},
            "horizontal_ratio": _clip01(horizontal),
            "vertical_ratio": _clip01(vertical),
            "openness": float(openness),
        }

    def _head_pose(self, lm):
        left_outer = lm[L_OUTER]
        right_outer = lm[R_OUTER]
        nose = lm[NOSE_TIP]
        chin = lm[CHIN]
        mouth_top = lm[MOUTH_TOP]
        mouth_bottom = lm[MOUTH_BOTTOM]

        eye_mid_x = float((left_outer.x + right_outer.x) * 0.5)
        eye_mid_y = float((left_outer.y + right_outer.y) * 0.5)
        inter_eye = float(np.hypot(right_outer.x - left_outer.x, right_outer.y - left_outer.y))
        face_height = float(abs(chin.y - nose.y))
        mouth_open = float(abs(mouth_bottom.y - mouth_top.y))
        yaw = float((nose.x - eye_mid_x) / max(inter_eye, 1e-7))
        pitch = float((nose.y - eye_mid_y) / max(inter_eye, 1e-7))
        roll = float(math.atan2(right_outer.y - left_outer.y, right_outer.x - left_outer.x))

        if self._head_anchor is None:
            self._head_anchor = (eye_mid_x, eye_mid_y, inter_eye)

        anchor_x, anchor_y, anchor_scale = self._head_anchor
        self._head_anchor = (
            (anchor_x * 0.97) + (eye_mid_x * 0.03),
            (anchor_y * 0.97) + (eye_mid_y * 0.03),
            (anchor_scale * 0.97) + (inter_eye * 0.03),
        )
        dx = float((eye_mid_x - anchor_x) / max(anchor_scale, 1e-7))
        dy = float((eye_mid_y - anchor_y) / max(anchor_scale, 1e-7))
        dscale = float(np.log(max(inter_eye, 1e-7) / max(anchor_scale, 1e-7)))

        return {
            "position": {
                "x": eye_mid_x,
                "y": eye_mid_y,
                "z": inter_eye,
                "dx_from_anchor": dx,
                "dy_from_anchor": dy,
                "scale_from_anchor": dscale,
            },
            "rotation": {
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
            },
            "face_metrics": {
                "inter_eye_distance": inter_eye,
                "face_height": face_height,
                "mouth_open": mouth_open,
            },
        }

    def sample(self, frame):
        """Return a JSON-serializable sample dictionary for the frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts_ms += 33
        result = self.landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        if not result.face_landmarks:
            return {
                "face_detected": False,
                "gaze": None,
                "left_eye": None,
                "right_eye": None,
                "head": None,
            }

        lm = result.face_landmarks[0]
        left_eye = self._eye_ratio(lm, L_IRIS_RING, L_OUTER, L_INNER, L_TOP, L_BOTTOM)
        right_eye = self._eye_ratio(lm, R_IRIS_RING, R_OUTER, R_INNER, R_TOP, R_BOTTOM)
        gaze_x = float((left_eye["horizontal_ratio"] + right_eye["horizontal_ratio"]) * 0.5)
        gaze_y = float((left_eye["vertical_ratio"] + right_eye["vertical_ratio"]) * 0.5)
        head = self._head_pose(lm)

        return {
            "face_detected": True,
            "gaze": {
                "x": _clip01(gaze_x),
                "y": _clip01(gaze_y),
                "space": "camera_normalized_uncalibrated",
            },
            "left_eye": left_eye,
            "right_eye": right_eye,
            "head": head,
        }
