import numpy as np
import pandas as pd
import pytest

from mensa_ad.data import TabularAnomalyDataset, load_labeled_frame, split_normal_train_val


def test_load_labeled_frame_validates_and_merges_labels(tmp_path):
    features = tmp_path / "features.csv"
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "id": [10, 11, 12],
            "f0": [0.1, -0.2, 0.3],
            "f1": [1.0, -1.0, 0.0],
        }
    ).to_csv(features, index=False)
    pd.DataFrame({"id": [12, 10, 11], "anomaly": [0, 0, 1]}).to_csv(labels, index=False)

    loaded = load_labeled_frame(features, labels, expected_features=2)

    assert loaded.feature_columns == ["f0", "f1"]
    assert loaded.frame["anomaly"].tolist() == [0, 1, 0]


def test_load_labeled_frame_rejects_fractional_anomaly_labels(tmp_path):
    features = tmp_path / "features.csv"
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "id": [10, 11],
            "f0": [0.1, -0.2],
            "f1": [1.0, -1.0],
        }
    ).to_csv(features, index=False)
    pd.DataFrame({"id": [10, 11], "anomaly": [0, 0.5]}).to_csv(labels, index=False)

    with pytest.raises(ValueError, match="anomaly labels must be binary"):
        load_labeled_frame(features, labels, expected_features=2)


def test_split_normal_train_val_removes_anomalies_from_weight_training(tmp_path):
    features = tmp_path / "features.csv"
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "a": [0.0, 0.1, 0.2, 0.3, 0.4],
            "b": [0.4, 0.3, 0.2, 0.1, 0.0],
        }
    ).to_csv(features, index=False)
    pd.DataFrame({"id": [1, 2, 3, 4, 5], "anomaly": [0, 1, 0, 1, 0]}).to_csv(labels, index=False)
    loaded = load_labeled_frame(features, labels, expected_features=2)

    train_df, val_df = split_normal_train_val(loaded.frame, val_fraction=0.34, seed=7)

    assert set(train_df["anomaly"].unique()) == {0}
    assert set(val_df["anomaly"].unique()) == {0}
    assert len(train_df) + len(val_df) == 3


def test_tabular_dataset_returns_float_features_and_int_labels(tmp_path):
    frame = pd.DataFrame(
        {
            "id": [1, 2],
            "f0": [0.25, -0.25],
            "f1": [1.0, -1.0],
            "anomaly": [0, 1],
        }
    )
    dataset = TabularAnomalyDataset(frame, ["f0", "f1"])

    features, label, sample_id = dataset[1]

    assert features.shape == (2,)
    assert features.dtype.name == "float32"
    assert isinstance(label, np.integer)
    assert isinstance(sample_id, np.integer)
    assert int(label) == 1
    assert int(sample_id) == 2


def test_tabular_dataset_replaces_missing_features_with_neutral_value():
    frame = pd.DataFrame(
        {
            "id": [1],
            "f0": [np.nan],
            "f1": [0.5],
            "anomaly": [0],
        }
    )
    dataset = TabularAnomalyDataset(frame, ["f0", "f1"])

    features, _, _ = dataset[0]

    assert np.all(np.isfinite(features))
    assert features.tolist() == [0.0, 0.5]
