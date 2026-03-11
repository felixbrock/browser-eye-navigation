#!/usr/bin/env python3
"""Train a browser-tab ranker from collected webcam and tab-click data."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "scikit-learn is required to run training. "
        "Install project dependencies with `UV_CACHE_DIR=.uv-cache uv sync`."
    ) from exc

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "train.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_NAME = "tab_ranker"

LEGACY_TAB_STRIP_LEFT_INSET_PX = 32.0
LEGACY_TAB_STRIP_RIGHT_INSET_PX = 132.0
LEGACY_TAB_STRIP_HEIGHT_PX = 36.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument(
        "--min-face-samples",
        type=int,
        default=5,
        help="Skip events with fewer recent face-detected samples than this threshold.",
    )
    return parser.parse_args()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def load_entries(path: Path):
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return entries


def estimate_tab_candidates(entry):
    window = entry.get("window") or {}
    tab_event = entry.get("tab_event") or {}
    window_width = max(1.0, _safe_float(window.get("width"), 1.0))
    tab_count = max(1, _safe_int(tab_event.get("tab_count"), 1))
    clicked_index = _safe_int(tab_event.get("clicked_tab_index"), 0)
    strip_left = LEGACY_TAB_STRIP_LEFT_INSET_PX
    strip_right = max(LEGACY_TAB_STRIP_RIGHT_INSET_PX, round(window_width * 0.08))
    strip_width = max(1.0, window_width - strip_left - strip_right)
    slot_width = strip_width / tab_count

    candidates = []
    for index in range(tab_count):
        left = strip_left + (index * slot_width)
        right = left + slot_width
        center = left + (slot_width * 0.5)
        candidates.append(
            {
                "tab_id": index,
                "index": index,
                "title": "",
                "is_active": index == clicked_index,
                "is_pinned": False,
                "bounds_px": {
                    "left": left,
                    "right": right,
                    "center": center,
                    "width": slot_width,
                    "top": 0.0,
                    "height": LEGACY_TAB_STRIP_HEIGHT_PX,
                },
                "bounds_norm": {
                    "left": clamp01((left - strip_left) / strip_width),
                    "right": clamp01((right - strip_left) / strip_width),
                    "center": clamp01((center - strip_left) / strip_width),
                    "width": clamp01(slot_width / strip_width),
                },
            }
        )

    return {
        "tab_strip": {
            "left_px": strip_left,
            "right_px": strip_left + strip_width,
            "width_px": strip_width,
            "height_px": LEGACY_TAB_STRIP_HEIGHT_PX,
            "right_inset_px": strip_right,
        },
        "tab_candidates": candidates,
        "used_legacy_layout": True,
    }


def get_candidate_layout(entry):
    tab_candidates = entry.get("tab_candidates")
    tab_strip = entry.get("tab_strip")
    if isinstance(tab_candidates, list) and tab_candidates and isinstance(tab_strip, dict):
        return {
            "tab_strip": tab_strip,
            "tab_candidates": tab_candidates,
            "used_legacy_layout": False,
        }
    return estimate_tab_candidates(entry)


def summarize_series(values: list[float], prefix: str):
    if not values:
        return {
            f"{prefix}_last": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_delta": 0.0,
        }

    arr = np.asarray(values, dtype=np.float32)
    return {
        f"{prefix}_last": float(arr[-1]),
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_delta": float(arr[-1] - arr[0]),
    }


def aggregate_sequence_features(samples):
    valid_samples = [sample for sample in samples if sample.get("face_detected")]
    features = {
        "sample_count": float(len(samples)),
        "face_sample_count": float(len(valid_samples)),
        "face_detected_fraction": float(len(valid_samples) / len(samples)) if samples else 0.0,
    }

    feature_series = {
        "gaze_x": [],
        "gaze_y": [],
        "left_open": [],
        "right_open": [],
        "head_pitch": [],
        "head_yaw": [],
        "head_roll": [],
        "head_dx": [],
        "head_dy": [],
        "head_scale": [],
        "inter_eye": [],
        "mouth_open": [],
    }

    for sample in valid_samples:
        gaze = sample.get("gaze") or {}
        left_eye = sample.get("left_eye") or {}
        right_eye = sample.get("right_eye") or {}
        head = sample.get("head") or {}
        rotation = head.get("rotation") or {}
        position = head.get("position") or {}
        face_metrics = head.get("face_metrics") or {}

        feature_series["gaze_x"].append(_safe_float(gaze.get("x")))
        feature_series["gaze_y"].append(_safe_float(gaze.get("y")))
        feature_series["left_open"].append(_safe_float(left_eye.get("openness")))
        feature_series["right_open"].append(_safe_float(right_eye.get("openness")))
        feature_series["head_pitch"].append(_safe_float(rotation.get("pitch")))
        feature_series["head_yaw"].append(_safe_float(rotation.get("yaw")))
        feature_series["head_roll"].append(_safe_float(rotation.get("roll")))
        feature_series["head_dx"].append(_safe_float(position.get("dx_from_anchor")))
        feature_series["head_dy"].append(_safe_float(position.get("dy_from_anchor")))
        feature_series["head_scale"].append(_safe_float(position.get("scale_from_anchor")))
        feature_series["inter_eye"].append(_safe_float(face_metrics.get("inter_eye_distance")))
        feature_series["mouth_open"].append(_safe_float(face_metrics.get("mouth_open")))

    for prefix, values in feature_series.items():
        features.update(summarize_series(values, prefix))

    return features


def candidate_features(candidate, tab_strip, tab_count: int):
    bounds_norm = candidate.get("bounds_norm") or {}
    bounds_px = candidate.get("bounds_px") or {}
    index = _safe_int(candidate.get("index"))
    title = str(candidate.get("title") or "")
    strip_width_px = max(1.0, _safe_float(tab_strip.get("width_px"), 1.0))

    return {
        "candidate_index": float(index),
        "candidate_index_norm": float(index / max(tab_count - 1, 1)),
        "candidate_left_norm": clamp01(_safe_float(bounds_norm.get("left"))),
        "candidate_right_norm": clamp01(_safe_float(bounds_norm.get("right"))),
        "candidate_center_norm": clamp01(_safe_float(bounds_norm.get("center"))),
        "candidate_width_norm": clamp01(_safe_float(bounds_norm.get("width"))),
        "candidate_left_px": _safe_float(bounds_px.get("left")),
        "candidate_center_px": _safe_float(bounds_px.get("center")),
        "candidate_width_px_norm": clamp01(_safe_float(bounds_px.get("width")) / strip_width_px),
        "candidate_is_pinned": 1.0 if candidate.get("is_pinned") else 0.0,
        "candidate_title_len": float(len(title)),
        "candidate_title_len_norm": min(len(title), 120) / 120.0,
        "tab_count": float(tab_count),
        "tab_count_norm": min(tab_count, 50) / 50.0,
    }


def build_example_features(sequence, candidate, tab_strip, tab_count: int):
    features = dict(sequence)
    features.update(candidate_features(candidate, tab_strip, tab_count))

    gaze_last = features["gaze_x_last"]
    gaze_mean = features["gaze_x_mean"]
    center = features["candidate_center_norm"]
    left = features["candidate_left_norm"]
    right = features["candidate_right_norm"]

    features["gaze_last_to_center_abs"] = abs(gaze_last - center)
    features["gaze_mean_to_center_abs"] = abs(gaze_mean - center)
    features["gaze_last_minus_center"] = gaze_last - center
    features["gaze_mean_minus_center"] = gaze_mean - center
    features["candidate_contains_gaze_last"] = 1.0 if left <= gaze_last <= right else 0.0
    features["candidate_contains_gaze_mean"] = 1.0 if left <= gaze_mean <= right else 0.0
    return features


def entry_to_examples(entry, min_face_samples: int):
    capture_quality = entry.get("capture_quality") or {}
    if _safe_int(capture_quality.get("face_sample_count")) < min_face_samples:
        return []

    layout = get_candidate_layout(entry)
    tab_strip = layout["tab_strip"]
    candidates = layout["tab_candidates"]
    if not candidates:
        return []

    sequence = aggregate_sequence_features(entry.get("pre_click_samples") or [])
    tab_event = entry.get("tab_event") or {}
    clicked_tab_id = tab_event.get("tab_id")
    clicked_tab_index = _safe_int(tab_event.get("clicked_tab_index"), -1)
    tab_count = len(candidates)
    event_id = f"{_safe_int(entry.get('timestamp_ms'))}:{clicked_tab_index}"

    examples = []
    for candidate in candidates:
        candidate_id = candidate.get("tab_id")
        candidate_index = _safe_int(candidate.get("index"), -1)
        label = int(
            (clicked_tab_id is not None and candidate_id == clicked_tab_id)
            or candidate_index == clicked_tab_index
        )
        features = build_example_features(sequence, candidate, tab_strip, tab_count)
        examples.append(
            {
                "event_id": event_id,
                "label": label,
                "features": features,
                "candidate_index": candidate_index,
                "used_legacy_layout": layout["used_legacy_layout"],
            }
        )
    return examples


def build_dataset(entries, min_face_samples: int):
    examples = []
    skipped_events = 0
    legacy_events = 0

    for entry in entries:
        event_examples = entry_to_examples(entry, min_face_samples)
        if not event_examples:
            skipped_events += 1
            continue
        examples.extend(event_examples)
        legacy_events += int(event_examples[0]["used_legacy_layout"])

    if not examples:
        raise ValueError("No usable training examples were found in the dataset.")

    feature_names = sorted(examples[0]["features"].keys())
    x = np.asarray(
        [[example["features"][name] for name in feature_names] for example in examples],
        dtype=np.float32,
    )
    y = np.asarray([example["label"] for example in examples], dtype=np.int32)
    event_ids = np.asarray([example["event_id"] for example in examples], dtype=object)
    candidate_indices = np.asarray([example["candidate_index"] for example in examples], dtype=np.int32)

    return {
        "X": x,
        "y": y,
        "event_ids": event_ids,
        "candidate_indices": candidate_indices,
        "feature_names": feature_names,
        "num_events": len(set(event_ids.tolist())),
        "num_examples": len(examples),
        "skipped_events": skipped_events,
        "legacy_events": legacy_events,
    }


def split_events(event_ids, test_size: float, random_seed: int):
    unique_events = np.unique(event_ids)
    if len(unique_events) < 4 or test_size <= 0.0:
        return unique_events, np.asarray([], dtype=object)

    train_events, test_events = train_test_split(
        unique_events,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
    )
    return np.asarray(train_events, dtype=object), np.asarray(test_events, dtype=object)


def sample_weights(y):
    positives = max(1, int((y == 1).sum()))
    negatives = max(1, int((y == 0).sum()))
    positive_weight = negatives / positives
    return np.where(y == 1, positive_weight, 1.0).astype(np.float32)


def fit_model(x_train, y_train, random_seed: int):
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=10,
        random_state=random_seed,
    )
    model.fit(x_train, y_train, sample_weight=sample_weights(y_train))
    return model


def score_event_rankings(probabilities, labels, event_ids):
    grouped = defaultdict(list)
    for probability, label, event_id in zip(probabilities, labels, event_ids):
        grouped[event_id].append((float(probability), int(label)))

    top1_hits = 0
    top3_hits = 0
    reciprocal_ranks = []
    for rows in grouped.values():
        ranked = sorted(rows, key=lambda item: item[0], reverse=True)
        positive_ranks = [rank for rank, (_, label) in enumerate(ranked, start=1) if label == 1]
        if not positive_ranks:
            continue
        rank = positive_ranks[0]
        top1_hits += int(rank == 1)
        top3_hits += int(rank <= 3)
        reciprocal_ranks.append(1.0 / rank)

    total_events = max(1, len(grouped))
    return {
        "events": len(grouped),
        "top1_accuracy": top1_hits / total_events,
        "top3_accuracy": top3_hits / total_events,
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }


def evaluate_model(model, x_test, y_test, event_ids_test):
    if len(x_test) == 0:
        return {"held_out": False}

    probabilities = model.predict_proba(x_test)[:, 1]
    candidate_accuracy = float(((probabilities >= 0.5).astype(np.int32) == y_test).mean())
    ranking_metrics = score_event_rankings(probabilities, y_test, event_ids_test)
    ranking_metrics.update(
        {
            "held_out": True,
            "candidate_accuracy": candidate_accuracy,
            "positive_rate": float(y_test.mean()) if len(y_test) else 0.0,
        }
    )
    return ranking_metrics


def save_artifacts(output_dir: Path, model_name: str, model, feature_names, dataset, metrics):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_name}.pkl"
    metadata_path = output_dir / f"{model_name}.metadata.json"
    metrics_path = output_dir / f"{model_name}.metrics.json"

    with model_path.open("wb") as handle:
        pickle.dump({"model": model, "feature_names": feature_names}, handle)

    metadata = {
        "model_type": type(model).__name__,
        "feature_names": feature_names,
        "num_examples": dataset["num_examples"],
        "num_events": dataset["num_events"],
        "legacy_events": dataset["legacy_events"],
        "skipped_events": dataset["skipped_events"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return model_path, metadata_path, metrics_path


def main():
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    entries = load_entries(input_path)
    dataset = build_dataset(entries, min_face_samples=args.min_face_samples)
    train_events, test_events = split_events(
        dataset["event_ids"],
        test_size=args.test_size,
        random_seed=args.random_seed,
    )

    train_mask = np.isin(dataset["event_ids"], train_events)
    test_mask = np.isin(dataset["event_ids"], test_events)

    model = fit_model(dataset["X"][train_mask], dataset["y"][train_mask], args.random_seed)
    metrics = evaluate_model(
        model,
        dataset["X"][test_mask],
        dataset["y"][test_mask],
        dataset["event_ids"][test_mask],
    )
    metrics.update(
        {
            "input_path": str(input_path),
            "train_events": int(len(np.unique(dataset["event_ids"][train_mask]))),
            "test_events": int(len(np.unique(dataset["event_ids"][test_mask]))),
            "num_examples": dataset["num_examples"],
            "num_events": dataset["num_events"],
            "legacy_events": dataset["legacy_events"],
            "skipped_events": dataset["skipped_events"],
        }
    )

    model_path, metadata_path, metrics_path = save_artifacts(
        output_dir,
        args.model_name,
        model,
        dataset["feature_names"],
        dataset,
        metrics,
    )

    print(json.dumps(
        {
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "metrics_path": str(metrics_path),
            "metrics": metrics,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
