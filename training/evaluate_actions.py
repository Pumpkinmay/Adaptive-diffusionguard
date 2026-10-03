"""Offline action accuracy, JSON validity, calibration, and latency metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from training.schemas import VALID_ACTIONS, ActionOutput


def macro_f1(labels: list[str], predictions: list[str]) -> float:
    scores: list[float] = []
    for action in sorted(VALID_ACTIONS):
        tp = sum(y == action and p == action for y, p in zip(labels, predictions))
        fp = sum(y != action and p == action for y, p in zip(labels, predictions))
        fn = sum(y == action and p != action for y, p in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return mean(scores)


def evaluate(rows: list[dict[str, Any]]) -> dict[str, float]:
    labels: list[str] = []
    predictions: list[str] = []
    correctness: list[float] = []
    confidences: list[float] = []
    latencies: list[float] = []
    valid = 0
    for row in rows:
        label = str(row["label"])
        labels.append(label)
        try:
            parsed = ActionOutput.parse_json(str(row["output"]))
            prediction = parsed.action
            confidence = parsed.confidence
            valid += 1
        except (ValueError, TypeError, json.JSONDecodeError):
            prediction = "__invalid__"
            confidence = 0.0
        predictions.append(prediction)
        correctness.append(float(prediction == label))
        confidences.append(confidence)
        latencies.append(float(row.get("latency_seconds", 0.0)))
    if not rows:
        raise ValueError("evaluation input is empty")
    brier = mean((confidence - correct) ** 2 for confidence, correct in zip(confidences, correctness))
    return {
        "action_macro_f1": macro_f1(labels, predictions),
        "json_valid_rate": valid / len(rows),
        "brier_score": brier,
        "mean_inference_seconds": mean(latencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(evaluate(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
