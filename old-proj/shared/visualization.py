from __future__ import annotations

import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-8
DEFAULT_MODEL_ORDER = ["STICformer", "MTGNN", "GraphWaveNet"]
RESULT_METRICS = [
    ("MAE", "test_mae_mean"),
    ("RMSE", "test_rmse_mean"),
    ("WAPE", "test_wape_mean"),
]


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


def _latest_consolidated_csv(results_root: str | Path, prefix: str) -> Path:
    csv_dir = Path(results_root) / "csv"
    candidates = sorted(csv_dir.glob(f"{prefix}*_consolidated_experiments.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum CSV consolidado encontrado em {csv_dir} com prefixo '{prefix}'."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _base_dataset_name(dataset: str) -> str:
    return str(dataset).split("-by-", 1)[0]


def _backbone_method_name(dataset: str) -> str:
    dataset = str(dataset)
    match = re.search(r"-by-(.*?)-with-", dataset)
    if match:
        return match.group(1)
    if "-by-" in dataset:
        return dataset.split("-by-", 1)[1]
    return "original"


def _variant_label(dataset: str, family: str) -> str:
    if family == "original":
        return "original"
    return _backbone_method_name(dataset)


def _sort_models(models: Sequence[str]) -> list[str]:
    preferred = {model: index for index, model in enumerate(DEFAULT_MODEL_ORDER)}
    return sorted(models, key=lambda model: (preferred.get(model, len(preferred)), model))


def _sort_variants(variants: Sequence[str]) -> list[str]:
    preferred = {"original": 0, "disp_fil": 1, "nois_corr": 2, "high_sal": 3}
    return sorted(variants, key=lambda variant: (preferred.get(variant, len(preferred)), variant))


def load_original_backbone_results(
    results_root: str | Path = "results/both",
    original_csv: str | Path | None = None,
    backbone_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Carrega os CSVs consolidados de resultados originais e backbones.

    Por padrao, seleciona automaticamente os arquivos mais recentes com os
    prefixos ``original_all-datasets`` e ``backbone_all-datasets`` em
    ``results_root/csv``.
    """
    results_root = Path(results_root)
    original_path = Path(original_csv) if original_csv else _latest_consolidated_csv(
        results_root, "original_all-datasets"
    )
    backbone_path = Path(backbone_csv) if backbone_csv else _latest_consolidated_csv(
        results_root, "backbone_all-datasets"
    )

    frames = []
    for family, path in [("original", original_path), ("backbone", backbone_path)]:
        frame = pd.read_csv(path)
        frame["experiment_family"] = family
        frame["source_csv"] = str(path)
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True)
    results["base_dataset"] = results["dataset"].map(_base_dataset_name)
    results["backbone_method"] = [
        _backbone_method_name(dataset) if family == "backbone" else "original"
        for dataset, family in zip(results["dataset"], results["experiment_family"])
    ]
    results["graph_variant"] = [
        _variant_label(dataset, family)
        for dataset, family in zip(results["dataset"], results["experiment_family"])
    ]
    return results


class RadarChart:
    """
    Radar chart para comparar modelos em MAE, RMSE e WAPE.

    Como as tres metricas possuem escalas diferentes e todas sao erros, o modo
    padrao converte cada metrica em score relativo ``melhor_valor / valor``.
    Assim, valores mais externos indicam melhor desempenho.
    """

    def __init__(
        self,
        metrics: Sequence[tuple[str, str]] = RESULT_METRICS,
        model_order: Sequence[str] = DEFAULT_MODEL_ORDER,
        normalize: str = "relative_score",
        fill_alpha: float = 0.08,
    ):
        self.metrics = list(metrics)
        self.model_order = list(model_order)
        self.normalize = normalize
        self.fill_alpha = fill_alpha

    def _model_order_for_frame(self, frame: pd.DataFrame) -> list[str]:
        present = [model for model in self.model_order if model in set(frame["model"])]
        extras = sorted(set(frame["model"]) - set(present))
        return present + extras

    def _metric_values(self, frame: pd.DataFrame, model: str) -> list[float]:
        row = frame[frame["model"] == model]
        if row.empty:
            return [float("nan") for _ in self.metrics]
        row = row.iloc[0]
        values = []
        for _, column in self.metrics:
            value = float(row[column])
            if self.normalize == "relative_score":
                best = max(float(frame[column].min()), EPS)
                values.append(best / max(value, EPS))
            elif self.normalize == "minmax_inverse":
                series = frame[column].astype(float)
                min_value = float(series.min())
                max_value = float(series.max())
                if abs(max_value - min_value) < EPS:
                    values.append(1.0)
                else:
                    values.append(1.0 - ((value - min_value) / (max_value - min_value)))
            elif self.normalize == "raw":
                values.append(value)
            else:
                raise ValueError(
                    "normalize deve ser 'relative_score', 'minmax_inverse' ou 'raw'."
                )
        return values

    def plot(
        self,
        frame: pd.DataFrame,
        output_path: str | Path | None = None,
        title: str | None = None,
        ax: plt.Axes | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        if frame.empty:
            raise ValueError("Nao e possivel plotar radar chart com DataFrame vazio.")

        labels = [label for label, _ in self.metrics]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
        else:
            fig = ax.figure

        palette = sns.color_palette("tab10", n_colors=max(len(frame["model"].unique()), 3))
        for color, model in zip(palette, self._model_order_for_frame(frame)):
            values = self._metric_values(frame, model)
            values += values[:1]
            ax.plot(angles, values, color=color, linewidth=2, label=model)
            ax.fill(angles, values, color=color, alpha=self.fill_alpha)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        if self.normalize == "raw":
            ax.set_title(title or "Radar Chart")
        else:
            ax.set_ylim(0.0, 1.05)
            ax.set_title(title or "Radar Chart (score relativo)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15))

        if output_path is not None:
            _save_figure(fig, Path(output_path))

        return fig, ax


def generate_results_radar_charts(
    results_root: str | Path = "results/both",
    output_dir: str | Path | None = None,
    original_csv: str | Path | None = None,
    backbone_csv: str | Path | None = None,
    normalize: str = "relative_score",
) -> list[Path]:
    """
    Gera radar charts comparando os modelos em MAE, RMSE e WAPE.

    A saida padrao cria uma grade por dataset base, com a rede original e cada
    backbone em subplots separados.
    """
    sns.set_theme(style="whitegrid")
    results_root = Path(results_root)
    out = Path(output_dir) if output_dir else results_root / "plots" / "radar_charts"
    out.mkdir(parents=True, exist_ok=True)

    results = load_original_backbone_results(
        results_root=results_root,
        original_csv=original_csv,
        backbone_csv=backbone_csv,
    )
    radar = RadarChart(normalize=normalize)
    generated: list[Path] = []

    for base_dataset, dataset_frame in results.groupby("base_dataset"):
        variants = _sort_variants(dataset_frame["graph_variant"].dropna().unique())
        cols = min(2, len(variants))
        rows = int(math.ceil(len(variants) / cols))
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(7 * cols, 6.5 * rows),
            subplot_kw={"polar": True},
            squeeze=False,
        )

        for index, variant in enumerate(variants):
            ax = axes[index // cols][index % cols]
            variant_frame = dataset_frame[dataset_frame["graph_variant"] == variant]
            radar.plot(
                variant_frame,
                title=f"{base_dataset} | {variant}",
                ax=ax,
            )

        for index in range(len(variants), rows * cols):
            fig.delaxes(axes[index // cols][index % cols])

        fig.suptitle(f"Radar de desempenho - {base_dataset}", y=1.02)
        output_file = out / f"{base_dataset}_radar_original_backbones.png"
        _save_figure(fig, output_file)
        generated.append(output_file)

    return generated


def metric_rank_matrix(
    results: pd.DataFrame,
    metric: str = "test_mae_mean",
    block_col: str = "dataset",
    model_col: str = "model",
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """
    Retorna matriz de ranks por bloco para testes Friedman/Nemenyi.

    Linhas sao datasets/backbones; colunas sao modelos. Rank menor significa
    melhor desempenho quando ``lower_is_better=True``.
    """
    required = {block_col, model_col, metric}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para rank matrix: {sorted(missing)}")

    values = results.pivot_table(
        index=block_col,
        columns=model_col,
        values=metric,
        aggfunc="mean",
    ).dropna(axis=0, how="any")

    if values.empty:
        raise ValueError("Nao ha blocos completos para calcular ranks.")

    ranks = values.rank(axis=1, method="average", ascending=lower_is_better)
    return ranks[_sort_models(list(ranks.columns))]


def friedman_test(
    results: pd.DataFrame,
    metric: str = "test_mae_mean",
    block_col: str = "dataset",
    model_col: str = "model",
    lower_is_better: bool = True,
) -> dict[str, Any]:
    """
    Executa teste de Friedman para comparar modelos em varios blocos.
    """
    ranks = metric_rank_matrix(
        results=results,
        metric=metric,
        block_col=block_col,
        model_col=model_col,
        lower_is_better=lower_is_better,
    )
    n_blocks, n_models = ranks.shape
    rank_sums = ranks.sum(axis=0)
    statistic = (
        12.0
        / (n_blocks * n_models * (n_models + 1.0))
        * float(((rank_sums - (n_blocks * (n_models + 1.0) / 2.0)) ** 2).sum())
    )

    p_value = float("nan")
    try:
        from scipy.stats import chi2, friedmanchisquare

        raw = results.pivot_table(
            index=block_col,
            columns=model_col,
            values=metric,
            aggfunc="mean",
        ).dropna(axis=0, how="any")
        raw = raw[list(ranks.columns)]
        statistic, p_value = friedmanchisquare(
            *[raw[column].to_numpy(dtype=float) for column in raw.columns]
        )
        statistic = float(statistic)
        p_value = float(p_value)
    except Exception:
        try:
            from scipy.stats import chi2

            p_value = float(chi2.sf(statistic, df=n_models - 1))
        except Exception:
            p_value = float("nan")

    return {
        "statistic": statistic,
        "p_value": p_value,
        "n_blocks": n_blocks,
        "n_models": n_models,
        "metric": metric,
        "average_ranks": ranks.mean(axis=0).sort_values(),
        "rank_matrix": ranks,
    }


def _fallback_nemenyi_q(alpha: float, n_models: int) -> float:
    q_alpha_005 = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
    }
    if abs(alpha - 0.05) > 1e-12 or n_models not in q_alpha_005:
        return float("nan")
    return q_alpha_005[n_models]


def nemenyi_test(
    ranks: pd.DataFrame,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Executa pos-teste de Nemenyi a partir de uma matriz de ranks.
    """
    if ranks.empty:
        raise ValueError("Matriz de ranks vazia.")

    n_blocks, n_models = ranks.shape
    average_ranks = ranks.mean(axis=0).sort_values()
    standard_error = math.sqrt(n_models * (n_models + 1.0) / (6.0 * n_blocks))

    try:
        from scipy.stats import studentized_range

        q_alpha = float(studentized_range.ppf(1.0 - alpha, n_models, math.inf) / math.sqrt(2.0))
    except Exception:
        q_alpha = _fallback_nemenyi_q(alpha, n_models)

    critical_difference = q_alpha * standard_error if not math.isnan(q_alpha) else float("nan")

    rows = []
    p_values = pd.DataFrame(np.nan, index=average_ranks.index, columns=average_ranks.index)
    for model in average_ranks.index:
        p_values.loc[model, model] = 1.0

    for model_a, model_b in combinations(average_ranks.index, 2):
        diff = abs(float(average_ranks[model_a] - average_ranks[model_b]))
        p_value = float("nan")
        try:
            from scipy.stats import studentized_range

            q_stat = math.sqrt(2.0) * diff / max(standard_error, EPS)
            p_value = float(studentized_range.sf(q_stat, n_models, math.inf))
        except Exception:
            pass

        significant = (
            diff > critical_difference if not math.isnan(critical_difference) else False
        )
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "rank_diff": diff,
                "p_value": p_value,
                "significant": significant,
            }
        )
        p_values.loc[model_a, model_b] = p_value
        p_values.loc[model_b, model_a] = p_value

    return {
        "average_ranks": average_ranks,
        "critical_difference": critical_difference,
        "standard_error": standard_error,
        "alpha": alpha,
        "pairwise": pd.DataFrame(rows),
        "p_values": p_values,
    }


def _nonsignificant_groups(
    average_ranks: pd.Series,
    pairwise: pd.DataFrame,
    alpha: float,
) -> list[list[str]]:
    if pairwise.empty:
        return []

    ordered = list(average_ranks.sort_values().index)
    p_lookup = {}
    for _, row in pairwise.iterrows():
        key = frozenset((row["model_a"], row["model_b"]))
        p_lookup[key] = row["p_value"]

    groups: list[list[str]] = []
    for start in range(len(ordered)):
        for end in range(start + 2, len(ordered) + 1):
            group = ordered[start:end]
            ok = True
            for model_a, model_b in combinations(group, 2):
                p_value = p_lookup.get(frozenset((model_a, model_b)), float("nan"))
                if pd.notna(p_value):
                    ok = ok and float(p_value) >= alpha
                else:
                    ok = False
            if ok:
                groups.append(group)

    maximal = []
    for group in groups:
        group_set = set(group)
        if not any(group_set < set(other) for other in groups):
            maximal.append(group)
    return maximal


# def plot_critical_difference_diagram(
#     average_ranks: pd.Series,
#     critical_difference: float,
#     output_path: str | Path,
#     title: str = "Critical Difference Diagram",
#     pairwise: pd.DataFrame | None = None,
#     alpha: float = 0.05,
# ) -> Path:
#     """
#     Plota diagrama de diferenca critica para ranks medios dos modelos.
#     """
#     ranks = average_ranks.sort_values()
#     models = list(ranks.index)
#     n_models = len(models)

#     fig_height = max(3.5, 0.7 * n_models + 2.0)
#     fig, ax = plt.subplots(figsize=(10, fig_height))
#     ax.set_title(title)
#     ax.set_xlim(0.5, n_models + 0.5)
#     ax.set_ylim(-1.0, n_models + 1.2)
#     ax.set_xlabel("Rank medio (menor e melhor)")
#     ax.set_yticks([])
#     ax.grid(axis="x", alpha=0.25)

#     y_base = np.arange(n_models, 0, -1)
#     for y, model in zip(y_base, models):
#         rank = float(ranks[model])
#         ax.plot(rank, y, "o", color="#1f77b4", markersize=8)
#         ax.hlines(y, xmin=rank, xmax=n_models + 0.25, color="#888888", linewidth=1)
#         ax.text(n_models + 0.32, y, f"{model} ({rank:.2f})", va="center", fontsize=10)

#     if not math.isnan(critical_difference):
#         cd_start = 1.0
#         cd_end = min(n_models + 0.25, cd_start + critical_difference)
#         y_cd = n_models + 0.65
#         ax.hlines(y_cd, cd_start, cd_end, color="#d62728", linewidth=3)
#         ax.vlines([cd_start, cd_end], y_cd - 0.08, y_cd + 0.08, color="#d62728", linewidth=2)
#         ax.text(
#             (cd_start + cd_end) / 2.0,
#             y_cd + 0.18,
#             f"CD={critical_difference:.2f}",
#             ha="center",
#             va="bottom",
#             color="#d62728",
#         )

#     if pairwise is not None and not pairwise.empty:
#         groups = _nonsignificant_groups(ranks, pairwise, alpha)
#         for index, group in enumerate(groups):
#             group_ranks = [float(ranks[model]) for model in group]
#             y = -0.25 - (0.18 * index)
#             ax.hlines(
#                 y,
#                 min(group_ranks),
#                 max(group_ranks),
#                 color="#222222",
#                 linewidth=5,
#                 alpha=0.85,
#             )

#     output_path = Path(output_path)
#     _save_figure(fig, output_path)
#     return output_path

def plot_critical_difference_diagram(
    average_ranks: pd.Series,
    critical_difference: float,
    output_path: str | Path,
    title: str = "Critical Difference Diagram",
    pairwise: pd.DataFrame | None = None,
    alpha: float = 0.05,
) -> Path:
    ranks = average_ranks.sort_values()
    models = list(ranks.index)
    n = len(models)

    left_models  = models[: n // 2]
    right_models = models[n // 2 :][::-1]
    n_left  = len(left_models)
    n_right = len(right_models)

    groups = (
        _nonsignificant_groups(ranks, pairwise, alpha)
        if pairwise is not None and not pairwise.empty
        else []
    )
    n_groups = len(groups)

    # ── tamanho da figura ────────────────────────────────────────────────────
    label_rows = max(n_left, n_right)
    fig_h = 2.5 + label_rows * 0.35 + n_groups * 0.25
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.set_title(title, fontsize=12, pad=10)
    ax.axis("off")

    # ── espaço de coordenadas: eixo em y=0 ──────────────────────────────────
    rank_min = float(ranks.min())
    rank_max = float(ranks.max())
    pad = (rank_max - rank_min) * 0.15 or 0.5

    ax.set_xlim(rank_min - pad, rank_max + pad)

    # y: acima do eixo → positivo, abaixo → negativo
    y_top    =  2.5
    y_bottom = -(label_rows * 0.6 + n_groups * 0.35 + 0.5)
    ax.set_ylim(y_bottom, y_top)

    AXIS_Y   = 0.0          # eixo principal
    LABEL_DY = 0.55         # espaço entre labels (em unidades de y)
    GROUP_DY = 0.30         # espaço entre grupos empilhados

    # ── eixo principal ───────────────────────────────────────────────────────
    ax.axhline(AXIS_Y, xmin=0, xmax=1, color="black", linewidth=1.2, alpha=0.7)

    tick_vals = np.arange(np.ceil(rank_min), np.floor(rank_max) + 1)
    for t in tick_vals:
        ax.plot([t, t], [AXIS_Y - 0.06, AXIS_Y + 0.06], color="black", lw=1)
        ax.text(t, AXIS_Y + 0.12, f"{t:.0f}",
                ha="center", va="bottom", fontsize=9)

    ax.text((rank_min + rank_max) / 2, AXIS_Y - 0.20,
            "rank médio (menor é melhor)",
            ha="center", va="top", fontsize=9, color="gray")

    # ── CD bar ───────────────────────────────────────────────────────────────
    if critical_difference and not np.isnan(critical_difference):
        cd_y  = y_top - 0.6
        cd_x0 = rank_min
        cd_x1 = rank_min + critical_difference
        bar_h = 0.10

        ax.plot([cd_x0, cd_x1], [cd_y, cd_y], color="black", lw=3)
        for xv in (cd_x0, cd_x1):
            ax.plot([xv, xv], [cd_y - bar_h, cd_y + bar_h], color="black", lw=2)
        ax.text((cd_x0 + cd_x1) / 2, cd_y + bar_h + 0.1,
                f"CD = {critical_difference:.2f}",
                ha="center", va="bottom", fontsize=10)

    # ── labels e linhas em L ─────────────────────────────────────────────────
    colors = plt.cm.tab10.colors
    color_map: dict[str, tuple] = {}

    x_left_margin  = rank_min - pad + 0.05
    x_right_margin = rank_max + pad - 0.05

    for i, model in enumerate(left_models):
        r = float(ranks[model])
        c = colors[i % len(colors)]
        color_map[model] = c
        y_l = AXIS_Y - (i + 1) * LABEL_DY

        ax.plot(r, AXIS_Y, "o", color=c, ms=5, zorder=4)
        ax.plot([r, r],                [AXIS_Y, y_l], color=c, lw=1.3)
        ax.plot([x_left_margin, r],    [y_l, y_l],    color=c, lw=1.3)
        ax.text(x_left_margin - 0.05, y_l,
                f"{model} ({r:.2f})",
                ha="right", va="center", fontsize=9, color=c)

    for i, model in enumerate(right_models):
        r = float(ranks[model])
        c = colors[(n_left + i) % len(colors)]
        color_map[model] = c
        y_l = AXIS_Y - (i + 1) * LABEL_DY

        ax.plot(r, AXIS_Y, "o", color=c, ms=5, zorder=4)
        ax.plot([r, r],                [AXIS_Y, y_l], color=c, lw=1.3)
        ax.plot([r, x_right_margin],   [y_l, y_l],    color=c, lw=1.3)
        ax.text(x_right_margin + 0.05, y_l,
                f"({r:.2f}) {model}",
                ha="left", va="center", fontsize=9, color=c)

    # ── grupos não-significativos ─────────────────────────────────────────────
    # Desenhados SOBRE o eixo (y=AXIS_Y) e empilhados para BAIXO quando colidem.
    # Coordenadas são em unidades de rank (x) e y direto — sem transforms.
    if groups:
        occupied: dict[float, list[tuple[float, float]]] = {}

        for group in groups:
            g_ranks = [float(ranks[m]) for m in group]
            gx0, gx1 = min(g_ranks), max(g_ranks)

            # encontra nível sem colisão
            level = 0
            while True:
                y_g   = AXIS_Y - level * GROUP_DY  # ← negativo = abaixo do eixo
                slots = occupied.get(y_g, [])
                clash = any(not (gx1 < s0 - 0.01 or gx0 > s1 + 0.01)
                            for s0, s1 in slots)
                if not clash:
                    break
                level += 1

            occupied.setdefault(y_g, []).append((gx0, gx1))

            ax.plot([gx0, gx1], [y_g, y_g],
                    color="black", lw=6,
                    solid_capstyle="round",
                    alpha=0.65, zorder=2)

        # legenda
        leg_y = y_bottom + 0.15
        ax.plot([rank_min, rank_min + (rank_max - rank_min) * 0.08],
                [leg_y, leg_y],
                color="black", lw=6, solid_capstyle="round", alpha=0.65)
        ax.text(rank_min + (rank_max - rank_min) * 0.10, leg_y,
                "sem diferença significativa",
                va="center", fontsize=9, color="gray")

    output_path = Path(output_path)
    _save_figure(fig, output_path)
    return output_path


def generate_statistical_comparison_plots(
    results_root: str | Path = "results/both",
    output_dir: str | Path | None = None,
    metric: str = "test_mae_mean",
    alpha: float = 0.05,
    original_csv: str | Path | None = None,
    backbone_csv: str | Path | None = None,
) -> dict[str, Any]:
    """
    Executa Friedman/Nemenyi e salva diagramas de diferenca critica.
    """
    results_root = Path(results_root)
    out = Path(output_dir) if output_dir else results_root / "plots" / "statistical_tests"
    out.mkdir(parents=True, exist_ok=True)
    results = load_original_backbone_results(
        results_root=results_root,
        original_csv=original_csv,
        backbone_csv=backbone_csv,
    )

    sections = {
        "original": results[results["experiment_family"] == "original"],
        "backbone": results[results["experiment_family"] == "backbone"],
        "all": results,
    }
    outputs: dict[str, Any] = {}

    for section_name, frame in sections.items():
        if frame.empty:
            continue
        friedman = friedman_test(frame, metric=metric, block_col="dataset")
        nemenyi = nemenyi_test(friedman["rank_matrix"], alpha=alpha)

        ranks_csv = out / f"{section_name}_{metric}_rank_matrix.csv"
        friedman["rank_matrix"].to_csv(ranks_csv)
        pairwise_csv = out / f"{section_name}_{metric}_nemenyi_pairwise.csv"
        nemenyi["pairwise"].to_csv(pairwise_csv, index=False)

        cd_path = plot_critical_difference_diagram(
            average_ranks=nemenyi["average_ranks"],
            critical_difference=nemenyi["critical_difference"],
            output_path=out / f"{section_name}_{metric}_critical_difference.png",
            title=f"Critical Difference - {section_name} ({metric})",
            pairwise=nemenyi["pairwise"],
            alpha=alpha,
        )

        outputs[section_name] = {
            "friedman": {
                "statistic": friedman["statistic"],
                "p_value": friedman["p_value"],
                "n_blocks": friedman["n_blocks"],
                "n_models": friedman["n_models"],
            },
            "critical_difference": nemenyi["critical_difference"],
            "rank_matrix_csv": str(ranks_csv),
            "nemenyi_pairwise_csv": str(pairwise_csv),
            "critical_difference_plot": str(cd_path),
        }

    return outputs
