from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-8


def _to_3d(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    if data.ndim == 4:
        if data.shape[-1] != 1:
            raise ValueError(
                f"Esperado ultimo eixo de features = 1 para visualizacao. Shape recebido: {data.shape}"
            )
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(
            f"Esperado tensor 3D [samples, horizon, nodes]. Shape recebido: {data.shape}"
        )
    return data


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.maximum(np.abs(y_true), EPS)
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100.0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = _safe_mape(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def _sample_indices(total: int, max_points: int) -> np.ndarray:
    if total <= max_points:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, num=max_points, dtype=np.int64)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _metrics_by_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    horizon = y_true.shape[1]
    rows: list[dict[str, Any]] = []
    for h in range(horizon):
        m = _metrics(y_true[:, h, :], y_pred[:, h, :])
        rows.append(
            {
                "horizon": h + 1,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "mape": m["mape"],
            }
        )
    return pd.DataFrame(rows)


def _metrics_by_node(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    num_nodes = y_true.shape[2]
    rows: list[dict[str, Any]] = []
    for node_id in range(num_nodes):
        m = _metrics(y_true[:, :, node_id], y_pred[:, :, node_id])
        rows.append(
            {
                "node": node_id,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "mape": m["mape"],
            }
        )
    return pd.DataFrame(rows)


def _plot_real_vs_pred_for_nodes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    node_indices: np.ndarray,
    output_file: Path,
    horizon_index: int,
    max_points: int,
    title_prefix: str,
) -> None:
    sampled = _sample_indices(y_true.shape[0], max_points)
    n_nodes = len(node_indices)
    cols = 2 if n_nodes > 1 else 1
    rows = int(np.ceil(n_nodes / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), squeeze=False)
    for i, node in enumerate(node_indices):
        ax = axes[i // cols][i % cols]
        true_series = y_true[sampled, horizon_index, node]
        pred_series = y_pred[sampled, horizon_index, node]
        mae_node = np.mean(np.abs(true_series - pred_series))

        ax.plot(sampled, true_series, label="Real", linewidth=1.5)
        ax.plot(sampled, pred_series, label="Previsto", linewidth=1.5, alpha=0.85)
        ax.set_title(f"No {int(node)} | MAE={mae_node:.4f}")
        ax.set_xlabel("Indice temporal (amostrado)")
        ax.set_ylabel("Valor")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")

    total_axes = rows * cols
    for j in range(n_nodes, total_axes):
        fig.delaxes(axes[j // cols][j % cols])

    fig.suptitle(f"{title_prefix} - Real vs Previsto (horizonte={horizon_index + 1})")
    _save_figure(fig, output_file)


def _plot_scatter_real_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    node_indices: np.ndarray,
    output_file: Path,
    horizon_index: int,
    max_points: int,
    title_prefix: str,
) -> None:
    sampled = _sample_indices(y_true.shape[0], max_points)
    n_nodes = len(node_indices)
    cols = 2 if n_nodes > 1 else 1
    rows = int(np.ceil(n_nodes / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), squeeze=False)
    for i, node in enumerate(node_indices):
        ax = axes[i // cols][i % cols]
        true_vals = y_true[sampled, horizon_index, node]
        pred_vals = y_pred[sampled, horizon_index, node]

        min_v = float(min(np.min(true_vals), np.min(pred_vals)))
        max_v = float(max(np.max(true_vals), np.max(pred_vals)))
        if abs(max_v - min_v) < 1e-8:
            max_v = min_v + 1.0

        ss_res = float(np.sum((true_vals - pred_vals) ** 2))
        ss_tot = float(np.sum((true_vals - np.mean(true_vals)) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + EPS)

        ax.scatter(true_vals, pred_vals, alpha=0.55, s=14, color="#1f77b4")
        ax.plot([min_v, max_v], [min_v, max_v], color="#d62728", linestyle="--", linewidth=1.5)
        ax.set_title(f"No {int(node)} | R²={r2:.4f}")
        ax.set_xlabel("Real")
        ax.set_ylabel("Previsto")
        ax.grid(alpha=0.2)

    total_axes = rows * cols
    for j in range(n_nodes, total_axes):
        fig.delaxes(axes[j // cols][j % cols])

    fig.suptitle(f"{title_prefix} - Scatter Real vs Previsto (horizonte={horizon_index + 1})")
    _save_figure(fig, output_file)


def _plot_metrics_by_horizon(df_horizon: pd.DataFrame, output_file: Path, title_prefix: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    metrics = [("mae", "MAE"), ("rmse", "RMSE"), ("mape", "MAPE (%)")]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, (col, label), color in zip(axes, metrics, colors):
        ax.plot(df_horizon["horizon"], df_horizon[col], marker="o", color=color, linewidth=2)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Horizonte")
    fig.suptitle(f"{title_prefix} - Metricas por Horizonte")
    _save_figure(fig, output_file)


def _plot_metrics_by_node(df_node: pd.DataFrame, output_file: Path, title_prefix: str) -> None:
    top_k = min(20, len(df_node))
    top_nodes = df_node.sort_values("mae", ascending=False).head(top_k)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].plot(df_node["node"], df_node["mae"], linewidth=1.2, color="#1f77b4")
    axes[0].set_title("MAE por no (todos os nos)")
    axes[0].set_xlabel("No")
    axes[0].set_ylabel("MAE")
    axes[0].grid(alpha=0.25)

    axes[1].bar(top_nodes["node"].astype(str), top_nodes["mae"], color="#d62728")
    axes[1].set_title(f"Top-{top_k} nos com maior MAE")
    axes[1].set_xlabel("No")
    axes[1].set_ylabel("MAE")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(alpha=0.25, axis="y")

    fig.suptitle(f"{title_prefix} - Metricas por No")
    _save_figure(fig, output_file)


def _moving_average(values: np.ndarray, window: int = 15) -> np.ndarray:
    if len(values) < 3:
        return values
    window = max(3, min(window, len(values)))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def _plot_error_over_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_file: Path,
    title_prefix: str,
    max_points: int,
) -> None:
    err = y_pred - y_true
    mae_t = np.mean(np.abs(err), axis=(1, 2))
    rmse_t = np.sqrt(np.mean(err ** 2, axis=(1, 2)))

    sampled = _sample_indices(len(mae_t), max_points)
    x = sampled

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, mae_t[sampled], color="#1f77b4", alpha=0.6, linewidth=1.2, label="MAE")
    ax.plot(
        x,
        _moving_average(mae_t[sampled], window=25),
        color="#1f77b4",
        linewidth=2.0,
        label="MAE (media movel)",
    )
    ax.plot(x, rmse_t[sampled], color="#ff7f0e", alpha=0.6, linewidth=1.2, label="RMSE")
    ax.plot(
        x,
        _moving_average(rmse_t[sampled], window=25),
        color="#ff7f0e",
        linewidth=2.0,
        label="RMSE (media movel)",
    )
    ax.set_title(f"{title_prefix} - Erro ao Longo do Tempo")
    ax.set_xlabel("Indice temporal (amostrado)")
    ax.set_ylabel("Erro")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    _save_figure(fig, output_file)


def _plot_train_val_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    output_file: Path,
    title_prefix: str,
) -> None:
    if not train_losses:
        return

    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_losses, marker="o", linewidth=2, label="Train Loss")
    if val_losses:
        val_epochs = np.arange(1, len(val_losses) + 1)
        ax.plot(val_epochs, val_losses, marker="o", linewidth=2, label="Val Loss")
    ax.set_title(f"{title_prefix} - Curvas de Treino/Validacao")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    _save_figure(fig, output_file)


def _plot_error_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_file: Path,
    title_prefix: str,
    horizon_index: int,
    max_time_points: int,
) -> None:
    abs_err = np.abs(y_pred[:, horizon_index, :] - y_true[:, horizon_index, :])  # [S, N]
    sampled = _sample_indices(abs_err.shape[0], max_time_points)
    heat = abs_err[sampled].T  # [N, S]

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(heat, cmap="magma", ax=ax, cbar_kws={"label": "|erro|"})
    ax.set_title(f"{title_prefix} - Heatmap de Erro (no x tempo, horizonte={horizon_index + 1})")
    ax.set_xlabel("Tempo (amostrado)")
    ax.set_ylabel("No")
    _save_figure(fig, output_file)


def generate_model_diagnostics(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: str | Path,
    model_name: str,
    dataset_name: str,
    experiment_name: str,
    train_losses: Sequence[float] | None = None,
    val_losses: Sequence[float] | None = None,
    num_nodes_to_plot: int = 4,
    horizon_for_line_and_heatmap: int = 0,
    max_points_line: int = 500,
    max_time_points_heatmap: int = 350,
    results_root: str | Path = "results",
) -> dict[str, Any]:
    """
    Gera relatorio visual de desempenho para um experimento de forecasting.
    """
    sns.set_theme(style="whitegrid")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_root = Path(results_root)
    csv_dir = results_root / "csv"
    json_dir = results_root / "json"
    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    y_true = _to_3d(targets)
    y_pred = _to_3d(predictions)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"targets e predictions com shapes diferentes: {y_true.shape} vs {y_pred.shape}")

    title_prefix = f"{model_name} | {dataset_name} | {experiment_name}"

    overall = _metrics(y_true, y_pred)
    df_horizon = _metrics_by_horizon(y_true, y_pred)
    df_node = _metrics_by_node(y_true, y_pred)

    overall_path = json_dir / f"{experiment_name}_overall_metrics.json"
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    horizon_csv = csv_dir / f"{experiment_name}_metrics_by_horizon.csv"
    node_csv = csv_dir / f"{experiment_name}_metrics_by_node.csv"
    df_horizon.to_csv(horizon_csv, index=False)
    df_node.to_csv(node_csv, index=False)

    node_order = np.argsort(df_node["mae"].to_numpy())[::-1]
    selected_nodes = node_order[: min(num_nodes_to_plot, len(node_order))]

    real_vs_pred_file = out / "real_vs_pred_nodes.png"
    scatter_file = out / "scatter_real_vs_pred.png"
    by_horizon_file = out / "metrics_by_horizon.png"
    by_node_file = out / "metrics_by_node.png"
    err_time_file = out / "error_over_time.png"
    train_val_file = out / "train_val_curves.png"
    heatmap_file = out / "error_heatmap_node_time.png"

    _plot_real_vs_pred_for_nodes(
        y_true=y_true,
        y_pred=y_pred,
        node_indices=selected_nodes,
        output_file=real_vs_pred_file,
        horizon_index=horizon_for_line_and_heatmap,
        max_points=max_points_line,
        title_prefix=title_prefix,
    )
    _plot_metrics_by_horizon(df_horizon, by_horizon_file, title_prefix)
    _plot_metrics_by_node(df_node, by_node_file, title_prefix)
    _plot_scatter_real_vs_pred(
        y_true=y_true,
        y_pred=y_pred,
        node_indices=selected_nodes,
        output_file=scatter_file,
        horizon_index=horizon_for_line_and_heatmap,
        max_points=max_points_line,
        title_prefix=title_prefix,
    )
    _plot_error_over_time(
        y_true=y_true,
        y_pred=y_pred,
        output_file=err_time_file,
        title_prefix=title_prefix,
        max_points=max_points_line,
    )
    _plot_error_heatmap(
        y_true=y_true,
        y_pred=y_pred,
        output_file=heatmap_file,
        title_prefix=title_prefix,
        horizon_index=horizon_for_line_and_heatmap,
        max_time_points=max_time_points_heatmap,
    )

    if train_losses:
        _plot_train_val_curves(
            train_losses=train_losses,
            val_losses=val_losses or [],
            output_file=train_val_file,
            title_prefix=title_prefix,
        )

    generated = [
        overall_path,
        horizon_csv,
        node_csv,
        real_vs_pred_file,
        scatter_file,
        by_horizon_file,
        by_node_file,
        err_time_file,
        heatmap_file,
    ]
    if train_losses:
        generated.append(train_val_file)

    return {
        "output_dir": str(out),
        "overall_metrics": overall,
        "generated_files": [str(path) for path in generated],
    }
