from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LabeledFrame:
    frame: pd.DataFrame
    feature_columns: list[str]


def load_labeled_frame(
    features_csv: str | Path,
    labels_csv: str | Path,
    expected_features: int | None,
) -> LabeledFrame:
    features = pd.read_csv(features_csv)
    labels = pd.read_csv(labels_csv)

    required_feature_columns = {"id"}
    required_label_columns = {"id", "anomaly"}
    missing_feature_columns = required_feature_columns.difference(features.columns)
    missing_label_columns = required_label_columns.difference(labels.columns)
    if missing_feature_columns:
        raise ValueError(f"features file is missing columns: {sorted(missing_feature_columns)}")
    if missing_label_columns:
        raise ValueError(f"labels file is missing columns: {sorted(missing_label_columns)}")
    if features["id"].duplicated().any():
        raise ValueError("features file contains duplicate id values")
    if labels["id"].duplicated().any():
        raise ValueError("labels file contains duplicate id values")

    feature_columns = [column for column in features.columns if column != "id"]
    if expected_features is not None and len(feature_columns) != expected_features:
        raise ValueError(f"expected {expected_features} features, found {len(feature_columns)}")

    merged = features.merge(labels[["id", "anomaly"]], on="id", how="left", validate="one_to_one")
    if merged["anomaly"].isna().any():
        missing_count = int(merged["anomaly"].isna().sum())
        raise ValueError(f"{missing_count} feature rows do not have labels")

    for column in feature_columns:
        merged[column] = pd.to_numeric(merged[column], errors="raise")
    merged["anomaly"] = pd.to_numeric(merged["anomaly"], errors="raise")
    if not set(merged["anomaly"].unique()).issubset({0, 1}):
        raise ValueError("anomaly labels must be binary values 0 or 1")
    merged["anomaly"] = merged["anomaly"].astype(int)

    return LabeledFrame(frame=merged, feature_columns=feature_columns)


def split_normal_train_val(
    frame: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0.0, 1.0)")

    normal = frame.loc[frame["anomaly"].eq(0)].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if normal.empty:
        raise ValueError("no normal rows are available for training")

    val_count = int(round(len(normal) * val_fraction))
    if val_fraction > 0.0 and val_count == 0 and len(normal) > 1:
        val_count = 1
    if val_count >= len(normal):
        val_count = len(normal) - 1

    val_df = normal.iloc[:val_count].reset_index(drop=True)
    train_df = normal.iloc[val_count:].reset_index(drop=True)
    return train_df, val_df


class TabularAnomalyDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, feature_columns: Sequence[str]) -> None:
        self.features = frame.loc[:, list(feature_columns)].to_numpy(dtype=np.float32, copy=True)
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=1.0, neginf=-1.0)
        self.labels = frame.loc[:, "anomaly"].to_numpy(dtype=np.int64, copy=True)
        self.ids = frame.loc[:, "id"].to_numpy(dtype=np.int64, copy=True)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.int64, np.int64]:
        return self.features[index], self.labels[index], self.ids[index]
