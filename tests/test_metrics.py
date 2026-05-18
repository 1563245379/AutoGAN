import numpy as np
import pytest

from mensa_ad.metrics import (
    ScoreNormalizer,
    best_f1_threshold,
    compute_binary_metrics,
    confusion_counts,
    roc_auc_score,
)


def test_score_normalizer_maps_scores_to_zero_one_range():
    normalizer = ScoreNormalizer.fit(np.array([2.0, 4.0, 6.0]))

    normalized = normalizer.transform(np.array([1.0, 2.0, 4.0, 8.0]))

    assert normalized.tolist() == [0.0, 0.0, 0.5, 1.0]


def test_best_f1_threshold_prefers_separation_between_normal_and_anomaly_scores():
    scores = np.array([0.05, 0.10, 0.20, 0.85, 0.90, 0.95])
    labels = np.array([0, 0, 0, 1, 1, 1])

    result = best_f1_threshold(scores, labels, steps=101)

    assert 0.20 <= result.threshold < 0.85
    assert result.metrics["f1"] == 1.0


def test_confusion_counts_and_metrics_use_anomaly_as_positive_class():
    scores = np.array([0.1, 0.7, 0.8, 0.2])
    labels = np.array([0, 1, 0, 1])

    counts = confusion_counts(scores, labels, threshold=0.5)
    metrics = compute_binary_metrics(counts)

    assert counts == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["fpr"] == 0.5
    assert metrics["f1"] == 0.5


def test_confusion_counts_flattens_column_scores_without_broadcasting():
    scores = np.array([[0.1], [0.7], [0.8], [0.2]])
    labels = np.array([0, 1, 0, 1])

    counts = confusion_counts(scores, labels, threshold=0.5)

    assert counts == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_confusion_counts_rejects_non_binary_labels():
    scores = np.array([0.1, 0.9])
    labels = np.array([0, 2])

    with pytest.raises(ValueError, match="binary labels"):
        confusion_counts(scores, labels, threshold=0.5)


def test_confusion_counts_rejects_fractional_labels_before_casting():
    scores = np.array([0.1, 0.9])
    labels = np.array([0.0, 0.5])

    with pytest.raises(ValueError, match="binary labels"):
        confusion_counts(scores, labels, threshold=0.5)


def test_roc_auc_score_is_one_for_perfect_ranking():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])

    assert roc_auc_score(scores, labels) == 1.0


def test_roc_auc_score_is_zero_for_reversed_ranking():
    scores = np.array([0.8, 0.9, 0.1, 0.2])
    labels = np.array([0, 0, 1, 1])

    assert roc_auc_score(scores, labels) == 0.0


def test_roc_auc_score_returns_none_for_single_class_labels():
    scores = np.array([0.1, 0.2, 0.3])
    labels = np.array([0, 0, 0])

    assert roc_auc_score(scores, labels) is None
