from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
import pytest
import torch

import main as main_module
from main import parse_args, run_evaluate, run_train, validate_args
from mensa_ad.models import DiscriminatorEncoder, GeneratorDecoder


def test_parse_args_exposes_val_score_interval(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mensa_ad.cli",
            "--epochs",
            "2",
            "--batch-size",
            "4096",
            "--score-samples",
            "1",
            "--val-score-interval",
            "3",
        ],
    )

    args = parse_args()

    assert args.epochs == 2
    assert args.batch_size == 4096
    assert args.score_samples == 1
    assert args.val_score_interval == 3


def test_parse_args_rejects_removed_score_alpha(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mensa_ad.cli", "--score-alpha", "0.5"])

    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_exposes_evaluate_mode_and_checkpoint_paths(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mensa_ad.cli",
            "--mode",
            "evaluate",
            "--checkpoint",
            "outputs/mensa_ad/model.pt",
            "--threshold-file",
            "outputs/mensa_ad/threshold.json",
        ],
    )

    args = parse_args()

    assert args.mode == "evaluate"
    assert args.checkpoint == "outputs/mensa_ad/model.pt"
    assert args.threshold_file == "outputs/mensa_ad/threshold.json"


def test_validate_args_rejects_threshold_steps_below_two():
    args = argparse.Namespace(threshold_steps=1, normal_quantile=0.995)

    with pytest.raises(ValueError, match="threshold_steps"):
        validate_args(args)


@pytest.mark.parametrize("normal_quantile", [-0.1, 1.1])
def test_validate_args_rejects_normal_quantile_outside_unit_interval(normal_quantile):
    args = argparse.Namespace(threshold_steps=2, normal_quantile=normal_quantile)

    with pytest.raises(ValueError, match="normal_quantile"):
        validate_args(args)


def test_validate_args_rejects_nan_normal_quantile():
    args = argparse.Namespace(threshold_steps=2, normal_quantile=float("nan"))

    with pytest.raises(ValueError, match="normal_quantile"):
        validate_args(args)


def test_validate_args_requires_checkpoint_for_evaluate_mode():
    args = argparse.Namespace(mode="evaluate", checkpoint=None, threshold_file="threshold.json")

    with pytest.raises(ValueError, match="checkpoint"):
        validate_args(args)


def test_validate_args_requires_threshold_file_for_evaluate_mode():
    args = argparse.Namespace(mode="evaluate", checkpoint="model.pt", threshold_file=None)

    with pytest.raises(ValueError, match="threshold_file"):
        validate_args(args)


def test_run_evaluate_loads_checkpoint_and_writes_metrics(tmp_path):
    feature_columns = ["f0", "f1"]
    train_csv = tmp_path / "train.csv"
    train_labels_csv = tmp_path / "train_label.csv"
    test_csv = tmp_path / "test.csv"
    labels_csv = tmp_path / "test_label.csv"
    checkpoint_path = tmp_path / "model.pt"
    threshold_path = tmp_path / "threshold.json"
    out_dir = tmp_path / "eval"

    pd.DataFrame(
        {
            "id": [20, 21],
            "f0": [0.4, -0.4],
            "f1": [-0.4, 0.4],
        }
    ).to_csv(train_csv, index=False)
    pd.DataFrame({"id": [20, 21], "anomaly": [1, 0]}).to_csv(train_labels_csv, index=False)
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "f0": [-0.2, 0.1, 0.3, -0.1],
            "f1": [0.2, -0.1, -0.3, 0.1],
        }
    ).to_csv(test_csv, index=False)
    pd.DataFrame({"id": [1, 2, 3, 4], "anomaly": [0, 1, 0, 1]}).to_csv(labels_csv, index=False)

    generator = GeneratorDecoder(noise_dim=10, output_dim=2)
    discriminator = DiscriminatorEncoder(input_dim=2, latent_dim=16)
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "feature_columns": feature_columns,
            "config": {
                "input_dim": 2,
                "noise_dim": 10,
                "latent_dim": 16,
                "batch_size": 2,
                "epochs": 1,
                "dropout": 0.2,
                "score_samples": 1,
                "device": "cpu",
                "seed": 123,
            },
            "history": [],
        },
        checkpoint_path,
    )
    threshold_path.write_text(
        '{"threshold": 0.5, "score_minimum": 0.0, "score_maximum": 1.0}',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        checkpoint=str(checkpoint_path),
        threshold_file=str(threshold_path),
        train=str(train_csv),
        train_labels=str(train_labels_csv),
        test=str(test_csv),
        test_labels=str(labels_csv),
        out=str(out_dir),
        batch_size=2,
        device="cpu",
        score_samples=1,
    )

    result = run_evaluate(args)

    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "predictions.csv").exists()
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "auc_roc" in metrics["test_metrics"]
    assert "auc_roc" in result["test_metrics"]
    assert result["evaluated_rows"] == 5
    assert metrics["train_anomalies_added_to_test"] == 1


def test_run_train_appends_training_anomalies_to_test_outputs(tmp_path, monkeypatch):
    train_csv = tmp_path / "train.csv"
    train_labels_csv = tmp_path / "train_label.csv"
    test_csv = tmp_path / "test.csv"
    test_labels_csv = tmp_path / "test_label.csv"
    out_dir = tmp_path / "train_out"

    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "f0": [0.0, 0.1, 0.2, 0.3],
            "f1": [0.3, 0.2, 0.1, 0.0],
        }
    ).to_csv(train_csv, index=False)
    pd.DataFrame({"id": [1, 2, 3, 4], "anomaly": [0, 1, 0, 1]}).to_csv(train_labels_csv, index=False)
    pd.DataFrame(
        {
            "id": [10, 11],
            "f0": [-0.1, -0.2],
            "f1": [0.1, 0.2],
        }
    ).to_csv(test_csv, index=False)
    pd.DataFrame({"id": [10, 11], "anomaly": [0, 1]}).to_csv(test_labels_csv, index=False)

    class FakeTrainState:
        generator = object()
        discriminator = object()
        history = [{"epoch": 1.0, "discriminator_loss": 0.0, "generator_loss": 0.0}]

        def checkpoint(self, feature_columns):
            return {"feature_columns": feature_columns, "config": {}, "history": self.history}

    monkeypatch.setattr(main_module, "train_mensa", lambda **_: FakeTrainState())
    monkeypatch.setattr(
        main_module,
        "score_dataset",
        lambda dataset, *_: dataset.ids.astype(float),
    )
    args = argparse.Namespace(
        train=str(train_csv),
        train_labels=str(train_labels_csv),
        test=str(test_csv),
        test_labels=str(test_labels_csv),
        out=str(out_dir),
        input_dim=None,
        val_fraction=0.0,
        seed=123,
        batch_size=2,
        epochs=1,
        lr=0.0002,
        score_samples=1,
        device="cpu",
        val_score_interval=0,
        threshold_strategy="labeled_bruteforce",
        threshold_steps=11,
        normal_quantile=0.995,
    )

    run_train(args)

    predictions = pd.read_csv(out_dir / "predictions.csv")
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert predictions["id"].tolist() == [10, 11, 2, 4]
    assert predictions["label"].tolist() == [0, 1, 1, 1]
    assert metrics["train_anomalies_added_to_test"] == 2
