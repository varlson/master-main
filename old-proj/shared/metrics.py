from __future__ import annotations

from typing import Iterable

import numpy as np

from shared.loaders import denormalize_predictions


EPS = 1e-8
REGRESSION_METRICS = ("mae", "rmse", "mape", "smape", "wape")


def _to_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float32)


def denormalize_arrays(
    predictions,
    targets,
    normalization_stats: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    preds_np = _to_numpy(predictions)
    targets_np = _to_numpy(targets)

    if normalization_stats:
        preds_np = denormalize_predictions(preds_np, normalization_stats)
        targets_np = denormalize_predictions(targets_np, normalization_stats)

    return preds_np, targets_np


def compute_regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)

    err = y_pred_np - y_true_np
    abs_err = np.abs(err)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    denom_mape = np.maximum(np.abs(y_true_np), EPS)
    mape = float(np.mean(abs_err / denom_mape) * 100.0)

    denom_smape = np.maximum(np.abs(y_true_np) + np.abs(y_pred_np), EPS)
    smape = float(np.mean((2.0 * abs_err) / denom_smape) * 100.0)

    wape_denom = float(np.sum(np.abs(y_true_np)))
    wape = float((np.sum(abs_err) / max(wape_denom, EPS)) * 100.0)

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "wape": wape,
    }


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def confidence_interval_95(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("confidence_interval_95 requer ao menos um valor.")

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    margin = float(1.96 * std / np.sqrt(arr.size)) if arr.size > 1 else 0.0

    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }


def summarize_metric_dicts(
    metric_dicts: list[dict[str, float]],
    *,
    source_metric_names: tuple[str, ...] = REGRESSION_METRICS,
    output_prefix: str,
) -> dict[str, float]:
    summary: dict[str, float] = {}

    for metric_name in source_metric_names:
        values = [
            float(metrics[metric_name])
            for metrics in metric_dicts
            if metric_name in metrics and metrics[metric_name] is not None
        ]
        if not values:
            continue

        stats = confidence_interval_95(values)
        for stat_name, stat_value in stats.items():
            summary[f"{output_prefix}_{metric_name}_{stat_name}"] = stat_value

    return summary
