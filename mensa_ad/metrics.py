from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScoreNormalizer:
    minimum: float
    maximum: float

    @classmethod
    def fit(cls, scores: np.ndarray) -> "ScoreNormalizer":
        values = np.asarray(scores, dtype=np.float64)
        if values.size == 0:
            raise ValueError("cannot fit a score normalizer on an empty array")
        return cls(minimum=float(np.min(values)), maximum=float(np.max(values)))

    def transform(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64)
        denominator = self.maximum - self.minimum
        if denominator <= 1e-12:
            return np.zeros_like(values, dtype=np.float64)
        normalized = (values - self.minimum) / denominator
        return np.clip(normalized, 0.0, 1.0)


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    metrics: dict[str, float]
    counts: dict[str, int]


def _validated_score_label_arrays(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    score_values = np.asarray(scores, dtype=np.float64).ravel()
    raw_label_values = np.asarray(labels, dtype=np.float64).ravel()
    if score_values.shape[0] != raw_label_values.shape[0]:
        raise ValueError("scores and labels must have the same length")
    if not np.isin(raw_label_values, [0.0, 1.0]).all():
        raise ValueError("labels must be binary labels encoded as 0 or 1")
    label_values = raw_label_values.astype(np.int64)
    return score_values, label_values


def confusion_counts(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, int]:
    score_values, label_values = _validated_score_label_arrays(scores, labels)
    predictions = score_values > threshold
    positives = label_values == 1
    negatives = label_values == 0
    return {
        "tp": int(np.logical_and(predictions, positives).sum()),
        "fp": int(np.logical_and(predictions, negatives).sum()),
        "tn": int(np.logical_and(~predictions, negatives).sum()),
        "fn": int(np.logical_and(~predictions, positives).sum()),
    }


def compute_binary_metrics(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "f1": float(f1),
    }


def roc_auc_score(scores: np.ndarray, labels: np.ndarray) -> float | None:
    score_values, label_values = _validated_score_label_arrays(scores, labels)
    positive_count = int((label_values == 1).sum())
    negative_count = int((label_values == 0).sum())
    if positive_count == 0 or negative_count == 0:
        return None

    order = np.argsort(score_values, kind="mergesort")
    sorted_scores = score_values[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[start:end] = average_rank
        start = end

    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    positive_rank_sum = float(original_ranks[label_values == 1].sum())
    auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )
    return float(auc)


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray, steps: int = 1001) -> ThresholdResult:
    if steps < 2:
        raise ValueError("steps must be at least 2")
    score_values, label_values = _validated_score_label_arrays(scores, labels)
    thresholds = np.linspace(0.0, 1.0, num=steps, dtype=np.float64)

    best_threshold = 0.0
    order = np.argsort(score_values)
    sorted_scores = score_values[order]
    sorted_labels = label_values[order]
    cumulative_positives = np.cumsum(sorted_labels == 1)
    cumulative_negatives = np.cumsum(sorted_labels == 0)
    total_positives = int(cumulative_positives[-1]) if sorted_labels.size else 0
    total_negatives = int(cumulative_negatives[-1]) if sorted_labels.size else 0

    def counts_at_threshold(threshold: float) -> dict[str, int]:
        cutoff = int(np.searchsorted(sorted_scores, threshold, side="right"))
        positives_below_or_equal = int(cumulative_positives[cutoff - 1]) if cutoff else 0
        negatives_below_or_equal = int(cumulative_negatives[cutoff - 1]) if cutoff else 0
        return {
            "tp": total_positives - positives_below_or_equal,
            "fp": total_negatives - negatives_below_or_equal,
            "tn": negatives_below_or_equal,
            "fn": positives_below_or_equal,
        }

    best_counts = counts_at_threshold(best_threshold)
    best_metrics = compute_binary_metrics(best_counts)

    for threshold in thresholds:
        counts = counts_at_threshold(float(threshold))
        metrics = compute_binary_metrics(counts)
        if (metrics["f1"], metrics["recall"], -metrics["fpr"]) > (
            best_metrics["f1"],
            best_metrics["recall"],
            -best_metrics["fpr"],
        ):
            best_threshold = float(threshold)
            best_counts = counts
            best_metrics = metrics

    return ThresholdResult(threshold=best_threshold, metrics=best_metrics, counts=best_counts)
