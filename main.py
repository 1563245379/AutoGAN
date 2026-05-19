from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mensa_ad.data import TabularAnomalyDataset, load_labeled_frame, split_normal_train_val
from mensa_ad.metrics import (
    ScoreNormalizer,
    best_f1_threshold,
    compute_binary_metrics,
    confusion_counts,
    roc_auc_score,
)
from mensa_ad.models import DiscriminatorEncoder, GeneratorDecoder
from mensa_ad.training import TrainConfig, resolve_device, score_dataset, set_seed, train_mensa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate MENSA Autoencoder-GAN anomaly detection.")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train")
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--train-labels", default="data/train_label.csv")
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--test-labels", default="data/test_label.csv")
    parser.add_argument("--out", default="outputs/mensa_ad")
    parser.add_argument("--checkpoint")
    parser.add_argument("--threshold-file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-samples", type=int, default=4)
    parser.add_argument("--threshold-steps", type=int, default=1001)
    parser.add_argument("--input-dim", type=int, default=None)
    parser.add_argument(
        "--threshold-strategy",
        choices=["labeled_bruteforce", "normal_quantile"],
        default="labeled_bruteforce",
    )
    parser.add_argument("--normal-quantile", type=float, default=0.995)
    parser.add_argument("--val-score-interval", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    mode = getattr(args, "mode", "train")
    if mode == "evaluate":
        if not getattr(args, "checkpoint", None):
            raise ValueError("checkpoint is required when mode is evaluate")
        if not getattr(args, "threshold_file", None):
            raise ValueError("threshold_file is required when mode is evaluate")
        return

    if args.threshold_steps < 2:
        raise ValueError("threshold_steps must be at least 2")
    if not math.isfinite(args.normal_quantile) or not 0.0 <= args.normal_quantile <= 1.0:
        raise ValueError("normal_quantile must be a finite value between 0.0 and 1.0")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _config_from_checkpoint(checkpoint: dict, args: argparse.Namespace) -> TrainConfig:
    config_payload = checkpoint.get("config", {})
    allowed_config_keys = {field.name for field in fields(TrainConfig)}
    config = TrainConfig(**{key: value for key, value in config_payload.items() if key in allowed_config_keys})
    return replace(
        config,
        batch_size=args.batch_size,
        score_samples=args.score_samples,
        device=args.device,
    )


def _load_trained_models(
    checkpoint_path: str | Path,
    args: argparse.Namespace,
) -> tuple[GeneratorDecoder, DiscriminatorEncoder, TrainConfig, list[str]]:
    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    feature_columns = list(checkpoint["feature_columns"])
    config = _config_from_checkpoint(checkpoint, args)
    generator = GeneratorDecoder(
        noise_dim=config.noise_dim,
        output_dim=config.input_dim,
        dropout=config.dropout,
    ).to(device)
    discriminator = DiscriminatorEncoder(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        dropout=config.dropout,
    ).to(device)
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    return generator, discriminator, config, feature_columns


def _load_threshold(threshold_file: str | Path) -> tuple[float, ScoreNormalizer, dict]:
    payload = json.loads(Path(threshold_file).read_text(encoding="utf-8"))
    threshold = float(payload["threshold"])
    minimum = float(payload["score_minimum"])
    maximum = float(payload["score_maximum"])
    if not all(math.isfinite(value) for value in (threshold, minimum, maximum)):
        raise ValueError("threshold file must contain finite threshold and score range values")
    return threshold, ScoreNormalizer(minimum=minimum, maximum=maximum), payload


def run_train(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loaded = load_labeled_frame(args.train, args.train_labels, expected_features=args.input_dim)
    test_loaded = load_labeled_frame(args.test, args.test_labels, expected_features=len(train_loaded.feature_columns))
    if train_loaded.feature_columns != test_loaded.feature_columns:
        raise ValueError("train and test feature columns do not match")

    normal_train_df, normal_val_df = split_normal_train_val(
        train_loaded.frame,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    train_dataset = TabularAnomalyDataset(normal_train_df, train_loaded.feature_columns)
    val_dataset = TabularAnomalyDataset(normal_val_df, train_loaded.feature_columns)
    full_train_dataset = TabularAnomalyDataset(train_loaded.frame, train_loaded.feature_columns)
    test_dataset = TabularAnomalyDataset(test_loaded.frame, train_loaded.feature_columns)

    config = TrainConfig(
        input_dim=len(train_loaded.feature_columns),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        score_samples=args.score_samples,
        device=args.device,
        seed=args.seed,
        val_score_interval=args.val_score_interval,
    )
    state = train_mensa(train_dataset=train_dataset, val_dataset=val_dataset, config=config)

    raw_train_scores = score_dataset(full_train_dataset, state.generator, state.discriminator, config)
    raw_test_scores = score_dataset(test_dataset, state.generator, state.discriminator, config)
    normalizer = ScoreNormalizer.fit(raw_train_scores)
    train_scores = normalizer.transform(raw_train_scores)
    test_scores = normalizer.transform(raw_test_scores)

    train_labels = train_loaded.frame["anomaly"].to_numpy(dtype=np.int64)
    test_labels = test_loaded.frame["anomaly"].to_numpy(dtype=np.int64)

    if args.threshold_strategy == "labeled_bruteforce":
        threshold_result = best_f1_threshold(train_scores, train_labels, steps=args.threshold_steps)
        threshold = threshold_result.threshold
        threshold_source = {
            "strategy": args.threshold_strategy,
            "train_metrics_at_threshold": threshold_result.metrics,
            "train_counts_at_threshold": threshold_result.counts,
        }
    else:
        normal_train_scores = train_scores[train_labels == 0]
        threshold = float(np.quantile(normal_train_scores, args.normal_quantile))
        threshold_source = {
            "strategy": args.threshold_strategy,
            "normal_quantile": args.normal_quantile,
        }

    test_counts = confusion_counts(test_scores, test_labels, threshold)
    test_metrics = compute_binary_metrics(test_counts)
    test_metrics["auc_roc"] = roc_auc_score(test_scores, test_labels)
    predictions = pd.DataFrame(
        {
            "id": test_loaded.frame["id"].to_numpy(dtype=np.int64),
            "label": test_labels,
            "score_raw": raw_test_scores,
            "score": test_scores,
            "prediction": (test_scores > threshold).astype(int),
        }
    )

    pd.DataFrame(state.history).to_csv(out_dir / "history.csv", index=False)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    write_json(
        out_dir / "metrics.json",
        {
            "test_counts": test_counts,
            "test_metrics": test_metrics,
            "threshold": threshold,
            "threshold_source": threshold_source,
            "train_rows_for_weight_training": len(normal_train_df),
            "val_rows_for_weight_training": len(normal_val_df),
            "feature_columns": train_loaded.feature_columns,
        },
    )
    write_json(
        out_dir / "threshold.json",
        {
            "threshold": threshold,
            "score_minimum": normalizer.minimum,
            "score_maximum": normalizer.maximum,
            "threshold_source": threshold_source,
        },
    )
    torch.save(state.checkpoint(train_loaded.feature_columns), out_dir / "model.pt")

    result = {"threshold": threshold, "test_metrics": test_metrics}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_evaluate(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    generator, discriminator, config, feature_columns = _load_trained_models(args.checkpoint, args)
    threshold, normalizer, threshold_payload = _load_threshold(args.threshold_file)
    test_loaded = load_labeled_frame(args.test, args.test_labels, expected_features=len(feature_columns))
    if test_loaded.feature_columns != feature_columns:
        raise ValueError("checkpoint and test feature columns do not match")

    set_seed(config.seed)
    test_dataset = TabularAnomalyDataset(test_loaded.frame, feature_columns)
    raw_test_scores = score_dataset(test_dataset, generator, discriminator, config)
    test_scores = normalizer.transform(raw_test_scores)
    test_labels = test_loaded.frame["anomaly"].to_numpy(dtype=np.int64)
    test_counts = confusion_counts(test_scores, test_labels, threshold)
    test_metrics = compute_binary_metrics(test_counts)
    test_metrics["auc_roc"] = roc_auc_score(test_scores, test_labels)
    predictions = pd.DataFrame(
        {
            "id": test_loaded.frame["id"].to_numpy(dtype=np.int64),
            "label": test_labels,
            "score_raw": raw_test_scores,
            "score": test_scores,
            "prediction": (test_scores > threshold).astype(int),
        }
    )

    predictions.to_csv(out_dir / "predictions.csv", index=False)
    write_json(
        out_dir / "metrics.json",
        {
            "checkpoint": str(args.checkpoint),
            "threshold_file": str(args.threshold_file),
            "evaluated_rows": int(len(test_loaded.frame)),
            "feature_columns": feature_columns,
            "test_counts": test_counts,
            "test_metrics": test_metrics,
            "threshold": threshold,
        },
    )
    write_json(out_dir / "threshold.json", threshold_payload)

    result = {
        "threshold": threshold,
        "test_metrics": test_metrics,
        "evaluated_rows": int(len(test_loaded.frame)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.mode == "evaluate":
        run_evaluate(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
