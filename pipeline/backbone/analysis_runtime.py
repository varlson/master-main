#!/usr/bin/env python3
# encoding: utf-8

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Iterable

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.bootstrap import WORKSPACE_ROOT, ensure_workspace_root_on_path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORKSPACE_ROOT
DATA_DIR = PROJECT_ROOT / "data"
GRAPHML_DIR = DATA_DIR / "GraphML"
NPY_DIR = DATA_DIR / "npy"
PKL_DIR = DATA_DIR / "pkl"
OUTPUT_ROOT = WORKSPACE_ROOT / "outputs" / "analysis"
DEFAULT_DATASET_LIST = ["metr-la", "pems-bay"]
DEFAULT_METHODS = [
    "disp_fil",
    "nois_corr",
    "high_sal",
    "doub_stoch",
    "glanb",
    "h_backbone",
    "marg_likelihood",
]
DEFAULT_ALPHA = 0.1


def _alpha_cut_label(alpha: float) -> str:
    return f"alpah_filter{str(alpha).replace('.', '_')}"


def _is_alpha_backbone_name(name: str) -> bool:
    return "-with-alpah_filter" in name or "-with-alpha_filter" in name


def _prepare_imports() -> None:
    ensure_workspace_root_on_path()
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))


_prepare_imports()

from shared.graph_analysis import (  # noqa: E402
    CENTRALITY_METRICS,
    GraphAnalysisResult,
    analyze_graph,
    compare_centralities,
    load_graph,
    robustness_curve,
    summarize_robustness,
)

try:  # noqa: E402
    from shared.utils import dataset_backbone_combinations, show_graph  # type: ignore
except Exception:  # noqa: E402
    show_graph = None

    def dataset_backbone_combinations(
        methods=[
            "disp_fil",
            "nois_corr",
            "high_sal",
            "doub_stoch",
            "glanb",
            "h_backbone",
            "marg_likelihood",
        ],
        alpha=0.1,
        percentile=None,
    ):
        cut = _alpha_cut_label(alpha)
        return [(method, cut) for method in methods]


def _load_runtime_defaults() -> tuple[list[str], float]:
    datasets = list(DEFAULT_DATASET_LIST)
    alpha = DEFAULT_ALPHA

    try:
        from config import ALPHA, DATASET_LIST

        datasets = list(DATASET_LIST)
        alpha = ALPHA
    except Exception as exc:
        print(f"Warning: could not import config.py ({exc}). Using local defaults.")

    return datasets, alpha


def _read_backbone_names_file() -> list[str]:
    names_file = NPY_DIR / "backbone_data_names.txt"
    if not names_file.exists():
        return []

    with names_file.open("r", encoding="utf-8") as file_obj:
        return [line.strip() for line in file_obj if line.strip()]


def _slugify(text: str) -> str:
    cleaned = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "graph"


def _label_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("-adj_mx"):
        return stem[: -len("-adj_mx")]
    return stem


def _resolve_named_graph_path(name: str) -> Path:
    candidates = [
        GRAPHML_DIR / f"{name}.GraphML",
        GRAPHML_DIR / f"{name}.graphml",
        NPY_DIR / f"{name}-adj_mx.npy",
        NPY_DIR / f"{name}.npy",
        PKL_DIR / f"{name}-adj_mx.pkl",
        PKL_DIR / f"{name}.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Nao foi possivel resolver caminho para a rede '{name}'.")


def _resolve_original_path(dataset: str) -> Path:
    candidates = [
        GRAPHML_DIR / f"{dataset}.GraphML",
        GRAPHML_DIR / f"{dataset}.graphml",
        PKL_DIR / f"{dataset}-adj_mx.pkl",
        NPY_DIR / f"{dataset}-adj_mx.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Rede original nao encontrada para o dataset '{dataset}'.")


def _discover_backbone_names(
    *,
    dataset: str,
    requested_names: list[str] | None,
    methods: list[str],
    alpha: float,
) -> list[str]:
    if requested_names:
        return [name for name in requested_names if name.startswith(f"{dataset}-by-")]

    names: list[str] = []
    names.extend(
        name
        for name in _read_backbone_names_file()
        if name.startswith(f"{dataset}-by-") and _is_alpha_backbone_name(name)
    )

    for method, cut in dataset_backbone_combinations(
        methods=methods,
        alpha=alpha,
    ):
        names.append(f"{dataset}-by-{method}-with-{cut}")

    for graphml_file in GRAPHML_DIR.glob(f"{dataset}-by-*.GraphML"):
        if _is_alpha_backbone_name(graphml_file.stem):
            names.append(graphml_file.stem)
    for npy_file in NPY_DIR.glob(f"{dataset}-by-*-adj_mx.npy"):
        name = _label_from_path(npy_file)
        if _is_alpha_backbone_name(name):
            names.append(name)

    unique_names = []
    for name in names:
        if name not in unique_names:
            try:
                _resolve_named_graph_path(name)
                unique_names.append(name)
            except FileNotFoundError:
                continue
    return unique_names


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analisa redes originais e backbones para gerar tabelas e plots comparativos."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets a analisar no modo automatico (padrao: DATASET_LIST do config.py).",
    )
    parser.add_argument(
        "--network-names",
        nargs="+",
        default=None,
        help="Lista opcional de nomes completos das redes backbone a considerar.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "disp_fil",
            "nois_corr",
            "high_sal",
            "doub_stoch",
            "glanb",
            "h_backbone",
            "marg_likelihood",
        ],
        default=DEFAULT_METHODS,
        help="Metodos usados para inferir nomes de backbone alpha quando --network-names nao for informado.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha usado na inferencia dos nomes de backbone.",
    )
    parser.add_argument(
        "--original-path",
        type=Path,
        default=None,
        help="Caminho explicito da rede original para analise customizada.",
    )
    parser.add_argument(
        "--backbone-paths",
        nargs="+",
        type=Path,
        default=None,
        help="Caminhos explicitos dos backbones para analise customizada.",
    )
    parser.add_argument(
        "--dataset-label",
        default="custom",
        help="Label do dataset quando usar --original-path/--backbone-paths.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Quantidade de hubs usados na comparacao de centralidade.",
    )
    parser.add_argument(
        "--robustness-steps",
        type=int,
        default=20,
        help="Numero de passos na curva de robustez.",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=10,
        help="Numero de repeticoes para a robustez por remocao aleatoria.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Tag opcional para a pasta de saida.",
    )
    return parser


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_dataframe(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def _ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(values) == 0:
        return np.array([]), np.array([])
    ordered = np.sort(values.astype(float))
    unique = np.unique(ordered)
    ccdf = np.array([(ordered >= value).mean() for value in unique], dtype=float)
    return unique, ccdf


def _pct_delta(new_value: float, old_value: float) -> float:
    if pd.isna(old_value) or abs(old_value) < 1e-12:
        return float("nan")
    return ((new_value / old_value) - 1.0) * 100.0


def _float_text(value, precision: int = 4) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{float(value):.{precision}f}"


def _signed_float_text(value, precision: int = 4, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "nan"
    number = float(value)
    sign = "+" if number > 0 else ""
    return f"{sign}{number:.{precision}f}{suffix}"


def _int_text(value) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return str(int(value))


def _infer_findings(row: pd.Series) -> list[str]:
    findings: list[str] = []
    edge_reduction = row.get("edge_reduction_pct", float("nan"))
    path_delta = row.get("avg_path_length_delta_pct", float("nan"))
    clustering_delta = row.get("average_clustering_delta_pct", float("nan"))
    modularity_delta = row.get("modularity_delta", float("nan"))
    centrality_mean = row.get("mean_centrality_spearman", float("nan"))
    targeted_robustness_delta = row.get("targeted_auc_lcc_delta_pct", float("nan"))

    if pd.notna(edge_reduction) and edge_reduction > 40:
        findings.append("removeu uma fração grande das arestas e tornou a rede bem mais esparsa")
    elif pd.notna(edge_reduction) and edge_reduction > 15:
        findings.append("reduziu a conectividade de forma moderada, preservando parte relevante das arestas")

    if pd.notna(clustering_delta) and clustering_delta < -20:
        findings.append("reduziu fortemente a redundância local e os triângulos")
    elif pd.notna(clustering_delta) and clustering_delta > 10:
        findings.append("preservou ou reforçou a estrutura local em torno de grupos densos")

    if pd.notna(path_delta) and path_delta > 20:
        findings.append("aumentou os caminhos médios, sugerindo perda de atalhos estruturais")
    elif pd.notna(path_delta) and path_delta < -10:
        findings.append("encurtou os caminhos médios no componente gigante")

    if pd.notna(modularity_delta) and modularity_delta > 0.05:
        findings.append("acentuou a separação entre comunidades")
    elif pd.notna(modularity_delta) and modularity_delta < -0.05:
        findings.append("enfraqueceu a estrutura comunitária")

    if pd.notna(centrality_mean) and centrality_mean >= 0.85:
        findings.append("preservou bem o ranking relativo das centralidades")
    elif pd.notna(centrality_mean) and centrality_mean < 0.60:
        findings.append("alterou de forma relevante quais nós aparecem como mais centrais")

    if pd.notna(targeted_robustness_delta) and targeted_robustness_delta < -20:
        findings.append("ficou mais frágil a ataques direcionados em hubs")
    elif pd.notna(targeted_robustness_delta) and targeted_robustness_delta > 10:
        findings.append("ganhou robustez relativa sob remoção direcionada de hubs")

    if not findings:
        findings.append("manteve um perfil estrutural relativamente próximo ao da rede original")
    return findings


def _safe_spring_layout(graph: nx.Graph) -> dict:
    if graph.number_of_nodes() == 0:
        return {}

    try:
        return nx.spring_layout(graph, seed=42, weight="weight")
    except Exception:
        return nx.spring_layout(graph, seed=42)


def _comparison_layout(
    original_graph: nx.Graph,
    backbone_graph: nx.Graph,
    original_pos: dict,
) -> dict:
    if set(backbone_graph.nodes()).issubset(original_pos.keys()):
        return original_pos

    combined_graph = nx.compose(original_graph, backbone_graph)
    return _safe_spring_layout(combined_graph)


def _graph_node_size(*graphs: nx.Graph) -> float:
    max_nodes = max((graph.number_of_nodes() for graph in graphs), default=1)
    return max(8.0, min(24.0, 1800.0 / max(max_nodes, 1)))


def _save_graph_comparison_plot(
    *,
    original_result: GraphAnalysisResult,
    backbone_result: GraphAnalysisResult,
    output_path: Path,
    pos: dict,
) -> None:
    titles = [
        f"Original: {original_result.name}",
        f"Backbone: {backbone_result.name}",
    ]
    node_size = _graph_node_size(original_result.graph, backbone_result.graph)

    if show_graph is not None:
        show_graph(
            [original_result.graph, backbone_result.graph],
            titles=titles,
            figsize=(14, 6),
            node_size=node_size,
            alpha=0.12,
            fig_shape=(1, 2),
            pos=pos,
            save_path=output_path,
            show=False,
            close=True,
        )
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    graph_pairs = [original_result.graph, backbone_result.graph]
    for ax, graph, title in zip(axes, graph_pairs, titles):
        nx.draw_networkx_nodes(graph, pos, node_size=node_size, ax=ax)
        nx.draw_networkx_edges(graph, pos, alpha=0.12, ax=ax)
        ax.set_title(title)
        ax.text(
            0.5,
            -0.08,
            f"Nos: {graph.number_of_nodes()} | Arestas: {graph.number_of_edges()}",
            ha="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.set_axis_off()

    _save_figure(fig, output_path)


def _plot_original_vs_backbone_networks(
    *,
    original_result: GraphAnalysisResult,
    backbone_results: list[GraphAnalysisResult],
    plots_dir: Path,
) -> None:
    if not backbone_results:
        return

    original_pos = _safe_spring_layout(original_result.graph)
    for backbone_result in backbone_results:
        pos = _comparison_layout(
            original_graph=original_result.graph,
            backbone_graph=backbone_result.graph,
            original_pos=original_pos,
        )
        _save_graph_comparison_plot(
            original_result=original_result,
            backbone_result=backbone_result,
            output_path=plots_dir / (
                f"{original_result.dataset}_{_slugify(backbone_result.name)}"
                "_original_vs_backbone_network.png"
            ),
            pos=pos,
        )


def _plot_degree_ccdf(results: list[GraphAnalysisResult], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette("tab10", n_colors=max(len(results), 3))

    for color, result in zip(palette, results):
        x, y = _ccdf(result.node_metrics["degree"].to_numpy(dtype=float))
        if len(x) == 0:
            continue
        ax.step(x, y, where="post", linewidth=2, label=result.name, color=color)

    ax.set_title("Distribuicao de Grau (CCDF)")
    ax.set_xlabel("Grau")
    ax.set_ylabel("P(X >= x)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    _save_figure(fig, output_path)


def _plot_weight_distribution(results: list[GraphAnalysisResult], output_path: Path) -> None:
    records = []
    for result in results:
        for weight in result.edge_metrics.get("weight", pd.Series(dtype=float)):
            records.append({"graph_name": result.name, "weight": weight})

    if not records:
        return

    frame = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=frame, x="graph_name", y="weight", ax=ax)
    ax.set_title("Distribuicao de Pesos das Arestas")
    ax.set_xlabel("Rede")
    ax.set_ylabel("Peso")
    ax.tick_params(axis="x", rotation=30)
    _save_figure(fig, output_path)


def _plot_clustering_distribution(results: list[GraphAnalysisResult], output_path: Path) -> None:
    frames = [
        result.node_metrics[["graph_name", "clustering"]]
        for result in results
        if not result.node_metrics.empty
    ]
    if not frames:
        return

    frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=frame, x="graph_name", y="clustering", ax=ax)
    ax.set_title("Clustering Local por Rede")
    ax.set_xlabel("Rede")
    ax.set_ylabel("Clustering")
    ax.tick_params(axis="x", rotation=30)
    _save_figure(fig, output_path)


def _plot_path_length_distribution(results: list[GraphAnalysisResult], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("tab10", n_colors=max(len(results), 3))

    for color, result in zip(palette, results):
        if result.path_lengths.empty:
            continue
        lengths = result.path_lengths["path_length"].to_numpy(dtype=float)
        bins = np.arange(lengths.min(), lengths.max() + 2) - 0.5
        ax.hist(
            lengths,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=result.name,
            color=color,
        )

    ax.set_title("Distribuicao dos Comprimentos de Caminho")
    ax.set_xlabel("Comprimento do caminho")
    ax.set_ylabel("Densidade")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    _save_figure(fig, output_path)


def _plot_community_sizes(results: list[GraphAnalysisResult], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("tab10", n_colors=max(len(results), 3))

    for color, result in zip(palette, results):
        if result.community_membership.empty:
            continue
        sizes = (
            result.community_membership[["community_id", "community_size"]]
            .drop_duplicates()
            .sort_values("community_size", ascending=False)["community_size"]
            .to_numpy(dtype=float)
        )
        x = np.arange(1, len(sizes) + 1)
        ax.plot(x, sizes, marker="o", linewidth=2, label=result.name, color=color)

    ax.set_title("Tamanho das Comunidades")
    ax.set_xlabel("Rank da comunidade")
    ax.set_ylabel("Numero de nos")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    _save_figure(fig, output_path)


def _plot_centrality_correlations(correlation_df: pd.DataFrame, output_path: Path) -> None:
    if correlation_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=correlation_df,
        x="metric",
        y="spearman_corr",
        hue="backbone_graph",
        ax=ax,
    )
    ax.set_title("Correlacao de Centralidades vs Rede Original")
    ax.set_xlabel("Centralidade")
    ax.set_ylabel("Spearman")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="best")
    _save_figure(fig, output_path)


def _plot_topk_overlap(overlap_df: pd.DataFrame, output_path: Path) -> None:
    if overlap_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=overlap_df,
        x="metric",
        y="overlap_ratio",
        hue="backbone_graph",
        ax=ax,
    )
    ax.set_title("Sobreposicao dos Top-k Nos Centrais")
    ax.set_xlabel("Centralidade")
    ax.set_ylabel("Overlap ratio")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="best")
    _save_figure(fig, output_path)


def _plot_robustness_curves(curves_df: pd.DataFrame, output_path: Path) -> None:
    if curves_df.empty:
        return

    aggregated = (
        curves_df.groupby(["graph_name", "strategy", "fraction_removed"], as_index=False)
        .agg(
            largest_component_ratio=("largest_component_ratio", "mean"),
            global_efficiency_ratio=("global_efficiency_ratio", "mean"),
        )
        .sort_values(["graph_name", "strategy", "fraction_removed"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    strategies = ["random", "targeted_degree"]
    titles = ["Remocao Aleatoria", "Ataque Direcionado por Grau"]

    for ax, strategy, title in zip(axes, strategies, titles):
        strategy_df = aggregated[aggregated["strategy"] == strategy]
        for graph_name, frame in strategy_df.groupby("graph_name"):
            ax.plot(
                frame["fraction_removed"],
                frame["largest_component_ratio"],
                linewidth=2,
                label=graph_name,
            )
        ax.set_title(title)
        ax.set_xlabel("Fracao de nos removidos")
        ax.set_ylabel("Razao do componente gigante")
        ax.grid(alpha=0.25)

    axes[-1].legend(loc="best")
    _save_figure(fig, output_path)


def _plot_metric_delta_heatmap(comparison_df: pd.DataFrame, output_path: Path) -> None:
    if comparison_df.empty:
        return

    metric_columns = [
        "edge_reduction_pct",
        "density_delta_pct",
        "average_clustering_delta_pct",
        "avg_path_length_delta_pct",
        "modularity_delta",
        "assortativity_delta",
        "targeted_auc_lcc_delta_pct",
    ]
    available = [column for column in metric_columns if column in comparison_df.columns]
    if not available:
        return

    pivot = comparison_df.set_index("backbone_graph")[available]
    renamed = pivot.rename(
        columns={
            "edge_reduction_pct": "edge_red_%",
            "density_delta_pct": "density_%",
            "average_clustering_delta_pct": "clustering_%",
            "avg_path_length_delta_pct": "path_%",
            "modularity_delta": "modularity_d",
            "assortativity_delta": "assortativity_d",
            "targeted_auc_lcc_delta_pct": "robust_target_%",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(renamed, annot=True, fmt=".2f", cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title("Delta Estrutural vs Rede Original")
    ax.set_xlabel("Metrica")
    ax.set_ylabel("Backbone")
    _save_figure(fig, output_path)


def _write_report(
    *,
    output_path: Path,
    dataset_sections: list[dict],
    output_root: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        file_obj.write("# Relatorio de Analise Estrutural de Redes e Backbones\n\n")
        file_obj.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        file_obj.write(f"Pasta de saida: `{output_root}`\n\n")

        for section in dataset_sections:
            dataset = section["dataset"]
            original = section["original_summary"]
            comparisons = section["comparison_df"]
            summary_df = section["summary_df"]
            robustness_summary = section["robustness_summary"]
            summary_by_name = {
                str(row["graph_name"]): row for _, row in summary_df.iterrows()
            }

            file_obj.write(f"## Dataset: {dataset}\n\n")
            file_obj.write("### Rede Original\n\n")
            file_obj.write(
                f"- Nos: {int(original['num_nodes'])} | Arestas: {int(original['num_edges'])} | "
                f"Densidade: {_float_text(original['density'])}\n"
            )
            file_obj.write(
                f"- Grau medio: {_float_text(original['average_degree'])} | "
                f"Forca media: {_float_text(original['average_strength'])}\n"
            )
            file_obj.write(
                f"- Clustering medio: {_float_text(original['average_clustering'])} | "
                f"Eficiencia global: {_float_text(original['global_efficiency'])} | "
                f"Modularidade: {_float_text(original['modularity'])}\n"
            )
            file_obj.write(
                f"- Caminho medio (gcc): {_float_text(original['avg_path_length_gcc'])} | "
                f"Componentes: {int(original['num_connected_components'])} | "
                f"Assortatividade: {_float_text(original['degree_assortativity'])}\n\n"
            )
            file_obj.write(
                f"- Comunidades: {int(original['num_communities'])} | "
                f"Maior comunidade: {int(original['largest_community_size'])} | "
                f"Razao do componente gigante: {_float_text(original['giant_component_ratio'])}\n\n"
            )

            if comparisons.empty:
                file_obj.write("Nenhum backbone encontrado para comparacao.\n\n")
                continue

            file_obj.write("### Comparacoes com a Rede Original\n\n")
            for _, row in comparisons.iterrows():
                backbone_name = str(row["backbone_graph"])
                backbone_summary = summary_by_name.get(backbone_name)

                file_obj.write(f"#### {backbone_name}\n\n")
                if backbone_summary is not None:
                    file_obj.write("Informacoes basicas do backbone:\n")
                    file_obj.write(
                        f"- Nos: {_int_text(backbone_summary['num_nodes'])} | "
                        f"Delta: {_signed_float_text(row['node_reduction_pct'] * -1.0, 2, '%')}\n"
                    )
                    file_obj.write(
                        f"- Arestas: {_int_text(backbone_summary['num_edges'])} | "
                        f"Delta: {_signed_float_text(row['edge_reduction_pct'] * -1.0, 2, '%')}\n"
                    )
                    file_obj.write(
                        f"- Densidade: {_float_text(backbone_summary['density'])} | "
                        f"Delta: {_signed_float_text(row['density_delta_pct'], 2, '%')}\n"
                    )
                    file_obj.write(
                        f"- Grau medio: {_float_text(backbone_summary['average_degree'])} | "
                        f"Forca media: {_float_text(backbone_summary['average_strength'])}\n"
                    )
                    file_obj.write(
                        f"- Componentes: {_int_text(backbone_summary['num_connected_components'])} | "
                        f"Razao do componente gigante: {_float_text(backbone_summary['giant_component_ratio'])} | "
                        f"Delta GCC: {_signed_float_text(row['giant_component_ratio_delta_pct'], 2, '%')}\n"
                    )
                    file_obj.write(
                        f"- Clustering medio: {_float_text(backbone_summary['average_clustering'])} | "
                        f"Delta: {_signed_float_text(row['average_clustering_delta_pct'], 2, '%')}\n"
                    )
                    file_obj.write(
                        f"- Eficiencia global: {_float_text(backbone_summary['global_efficiency'])} | "
                        f"Delta: {_signed_float_text(row['global_efficiency_delta_pct'], 2, '%')}\n"
                    )
                    file_obj.write(
                        f"- Caminho medio (gcc): {_float_text(backbone_summary['avg_path_length_gcc'])} | "
                        f"Delta: {_signed_float_text(row['avg_path_length_delta_pct'], 2, '%')}\n"
                    )
                    file_obj.write(
                        f"- Modularidade: {_float_text(backbone_summary['modularity'])} | "
                        f"Delta: {_signed_float_text(row['modularity_delta'])}\n"
                    )
                    file_obj.write(
                        f"- Assortatividade: {_float_text(backbone_summary['degree_assortativity'])} | "
                        f"Delta: {_signed_float_text(row['assortativity_delta'])}\n"
                    )
                    file_obj.write(
                        f"- Comunidades: {_int_text(backbone_summary['num_communities'])} | "
                        f"Maior comunidade: {_int_text(backbone_summary['largest_community_size'])}\n"
                    )

                    random_auc = _robustness_lookup(
                        robustness_summary, backbone_name, "random", "auc_lcc_mean"
                    )
                    targeted_auc = _robustness_lookup(
                        robustness_summary, backbone_name, "targeted_degree", "auc_lcc_mean"
                    )
                    file_obj.write(
                        f"- Robustez aleatoria (AUC LCC): {_float_text(random_auc)} | "
                        f"Delta: {_signed_float_text(row['random_auc_lcc_delta_pct'], 2, '%')}\n"
                    )
                    file_obj.write(
                        f"- Robustez alvo (AUC LCC): {_float_text(targeted_auc)} | "
                        f"Delta: {_signed_float_text(row['targeted_auc_lcc_delta_pct'], 2, '%')}\n"
                    )

                file_obj.write("\nComparacao direta com a rede original:\n")
                file_obj.write(
                    f"- Correlacao media das centralidades: {_float_text(row['mean_centrality_spearman'])}\n"
                )
                file_obj.write(
                    f"- Menor correlacao de centralidade entre as metricas: "
                    f"{_float_text(row['min_centrality_spearman'])}\n"
                )
                file_obj.write(
                    f"- Overlap medio top-k: {_float_text(row['mean_topk_overlap'])}\n"
                )
                file_obj.write("- Leitura interpretativa:\n")
                for finding in _infer_findings(row):
                    file_obj.write(f"  - {finding}\n")
                file_obj.write("\n")


def _save_per_graph_outputs(result: GraphAnalysisResult, csv_dir: Path) -> None:
    stem = _slugify(result.name)
    _save_dataframe(result.node_metrics, csv_dir / f"{stem}_node_metrics.csv")
    _save_dataframe(result.edge_metrics, csv_dir / f"{stem}_edge_metrics.csv")
    _save_dataframe(result.path_lengths, csv_dir / f"{stem}_path_lengths.csv")
    _save_dataframe(result.community_membership, csv_dir / f"{stem}_community_membership.csv")


def _build_group_specs_from_paths(
    *,
    dataset_label: str,
    original_path: Path,
    backbone_paths: list[Path],
) -> list[dict]:
    return [
        {
            "dataset": dataset_label,
            "original_path": original_path,
            "backbone_paths": backbone_paths,
        }
    ]


def _build_group_specs_from_datasets(
    *,
    datasets: list[str],
    requested_names: list[str] | None,
    methods: list[str],
    alpha: float,
) -> list[dict]:
    specs: list[dict] = []
    for dataset in datasets:
        original_path = _resolve_original_path(dataset)
        backbone_names = _discover_backbone_names(
            dataset=dataset,
            requested_names=requested_names,
            methods=methods,
            alpha=alpha,
        )
        backbone_paths = [_resolve_named_graph_path(name) for name in backbone_names]
        specs.append(
            {
                "dataset": dataset,
                "original_path": original_path,
                "backbone_paths": backbone_paths,
            }
        )
    return specs


def _robustness_lookup(summary_df: pd.DataFrame, graph_name: str, strategy: str, column: str) -> float:
    if summary_df.empty:
        return float("nan")
    mask = (summary_df["graph_name"] == graph_name) & (summary_df["strategy"] == strategy)
    frame = summary_df.loc[mask]
    if frame.empty:
        return float("nan")
    return float(frame.iloc[0][column])


def _analyze_group(
    *,
    dataset: str,
    original_path: Path,
    backbone_paths: list[Path],
    csv_dir: Path,
    plots_dir: Path,
    top_k: int,
    robustness_steps: int,
    random_trials: int,
) -> dict:
    print(f"\n{'=' * 100}")
    print(f"Analisando dataset: {dataset}")
    print(f"Rede original: {original_path}")
    print(f"Backbones: {len(backbone_paths)}")
    print(f"{'=' * 100}")

    original_result = analyze_graph(
        graph=load_graph(original_path),
        name=_label_from_path(original_path),
        dataset=dataset,
        role="original",
        source_path=original_path,
    )
    results = [original_result]

    for backbone_path in backbone_paths:
        backbone_result = analyze_graph(
            graph=load_graph(backbone_path),
            name=_label_from_path(backbone_path),
            dataset=dataset,
            role="backbone",
            source_path=backbone_path,
        )
        results.append(backbone_result)

    for result in results:
        _save_per_graph_outputs(result, csv_dir)

    summary_df = pd.DataFrame([result.summary for result in results]).sort_values(
        ["role", "graph_name"]
    )
    _save_dataframe(summary_df, csv_dir / f"{dataset}_network_summary.csv")

    robustness_curves = pd.concat(
        [
            robustness_curve(
                graph=result.graph,
                graph_name=result.name,
                dataset=dataset,
                steps=robustness_steps,
                random_trials=random_trials,
            )
            for result in results
        ],
        ignore_index=True,
    )
    robustness_summary = summarize_robustness(robustness_curves)
    _save_dataframe(robustness_curves, csv_dir / f"{dataset}_robustness_curves.csv")
    _save_dataframe(robustness_summary, csv_dir / f"{dataset}_robustness_summary.csv")

    centrality_comparisons: list[pd.DataFrame] = []
    correlation_summaries: list[pd.DataFrame] = []
    overlap_summaries: list[pd.DataFrame] = []
    comparison_rows: list[dict] = []

    original_random_auc = _robustness_lookup(
        robustness_summary, original_result.name, "random", "auc_lcc_mean"
    )
    original_targeted_auc = _robustness_lookup(
        robustness_summary, original_result.name, "targeted_degree", "auc_lcc_mean"
    )

    for result in results[1:]:
        centrality_df, correlation_df, overlap_df = compare_centralities(
            original=original_result,
            backbone=result,
            top_k=top_k,
            metrics=CENTRALITY_METRICS,
        )
        centrality_comparisons.append(centrality_df)
        correlation_summaries.append(correlation_df)
        overlap_summaries.append(overlap_df)

        _save_dataframe(
            centrality_df,
            csv_dir / f"{dataset}_{_slugify(result.name)}_centrality_comparison.csv",
        )

        targeted_auc = _robustness_lookup(
            robustness_summary, result.name, "targeted_degree", "auc_lcc_mean"
        )
        random_auc = _robustness_lookup(
            robustness_summary, result.name, "random", "auc_lcc_mean"
        )

        comparison_rows.append(
            {
                "dataset": dataset,
                "original_graph": original_result.name,
                "backbone_graph": result.name,
                "original_num_nodes": original_result.summary["num_nodes"],
                "backbone_num_nodes": result.summary["num_nodes"],
                "node_reduction_pct": (
                    1.0
                    - (result.summary["num_nodes"] / max(original_result.summary["num_nodes"], 1))
                )
                * 100.0,
                "original_num_edges": original_result.summary["num_edges"],
                "backbone_num_edges": result.summary["num_edges"],
                "edge_reduction_pct": (
                    1.0
                    - (result.summary["num_edges"] / max(original_result.summary["num_edges"], 1))
                )
                * 100.0,
                "original_density": original_result.summary["density"],
                "backbone_density": result.summary["density"],
                "density_delta_pct": _pct_delta(
                    result.summary["density"], original_result.summary["density"]
                ),
                "original_average_degree": original_result.summary["average_degree"],
                "backbone_average_degree": result.summary["average_degree"],
                "original_average_strength": original_result.summary["average_strength"],
                "backbone_average_strength": result.summary["average_strength"],
                "original_average_clustering": original_result.summary["average_clustering"],
                "backbone_average_clustering": result.summary["average_clustering"],
                "average_clustering_delta_pct": _pct_delta(
                    result.summary["average_clustering"],
                    original_result.summary["average_clustering"],
                ),
                "original_local_efficiency_mean": original_result.summary["local_efficiency_mean"],
                "backbone_local_efficiency_mean": result.summary["local_efficiency_mean"],
                "local_efficiency_delta_pct": _pct_delta(
                    result.summary["local_efficiency_mean"],
                    original_result.summary["local_efficiency_mean"],
                ),
                "original_avg_path_length_gcc": original_result.summary["avg_path_length_gcc"],
                "backbone_avg_path_length_gcc": result.summary["avg_path_length_gcc"],
                "avg_path_length_delta_pct": _pct_delta(
                    result.summary["avg_path_length_gcc"],
                    original_result.summary["avg_path_length_gcc"],
                ),
                "original_global_efficiency": original_result.summary["global_efficiency"],
                "backbone_global_efficiency": result.summary["global_efficiency"],
                "global_efficiency_delta_pct": _pct_delta(
                    result.summary["global_efficiency"],
                    original_result.summary["global_efficiency"],
                ),
                "original_num_connected_components": original_result.summary["num_connected_components"],
                "backbone_num_connected_components": result.summary["num_connected_components"],
                "original_giant_component_ratio": original_result.summary["giant_component_ratio"],
                "backbone_giant_component_ratio": result.summary["giant_component_ratio"],
                "giant_component_ratio_delta_pct": _pct_delta(
                    result.summary["giant_component_ratio"],
                    original_result.summary["giant_component_ratio"],
                ),
                "original_modularity": original_result.summary["modularity"],
                "backbone_modularity": result.summary["modularity"],
                "modularity_delta": result.summary["modularity"] - original_result.summary["modularity"],
                "original_assortativity": original_result.summary["degree_assortativity"],
                "backbone_assortativity": result.summary["degree_assortativity"],
                "assortativity_delta": (
                    result.summary["degree_assortativity"]
                    - original_result.summary["degree_assortativity"]
                ),
                "original_num_communities": original_result.summary["num_communities"],
                "backbone_num_communities": result.summary["num_communities"],
                "original_largest_community_size": original_result.summary["largest_community_size"],
                "backbone_largest_community_size": result.summary["largest_community_size"],
                "mean_centrality_spearman": float(correlation_df["spearman_corr"].mean()),
                "min_centrality_spearman": float(correlation_df["spearman_corr"].min()),
                "mean_topk_overlap": float(overlap_df["overlap_ratio"].mean()),
                "random_auc_lcc_original": original_random_auc,
                "random_auc_lcc_backbone": random_auc,
                "random_auc_lcc_delta_pct": _pct_delta(random_auc, original_random_auc),
                "targeted_auc_lcc_original": original_targeted_auc,
                "targeted_auc_lcc_backbone": targeted_auc,
                "targeted_auc_lcc_delta_pct": _pct_delta(targeted_auc, original_targeted_auc),
            }
        )

    centrality_all = pd.concat(centrality_comparisons, ignore_index=True) if centrality_comparisons else pd.DataFrame()
    correlation_all = pd.concat(correlation_summaries, ignore_index=True) if correlation_summaries else pd.DataFrame()
    overlap_all = pd.concat(overlap_summaries, ignore_index=True) if overlap_summaries else pd.DataFrame()
    comparison_df = pd.DataFrame(comparison_rows).sort_values("backbone_graph") if comparison_rows else pd.DataFrame()

    _save_dataframe(centrality_all, csv_dir / f"{dataset}_centrality_comparison_all_nodes.csv")
    _save_dataframe(comparison_df, csv_dir / f"{dataset}_comparison_vs_original.csv")
    _save_dataframe(correlation_all, csv_dir / f"{dataset}_centrality_correlation_summary.csv")
    _save_dataframe(overlap_all, csv_dir / f"{dataset}_centrality_topk_overlap.csv")

    _plot_original_vs_backbone_networks(
        original_result=original_result,
        backbone_results=results[1:],
        plots_dir=plots_dir,
    )
    _plot_degree_ccdf(results, plots_dir / f"{dataset}_degree_ccdf.png")
    _plot_weight_distribution(results, plots_dir / f"{dataset}_weight_distribution.png")
    _plot_clustering_distribution(results, plots_dir / f"{dataset}_clustering_distribution.png")
    _plot_path_length_distribution(results, plots_dir / f"{dataset}_path_length_distribution.png")
    _plot_community_sizes(results, plots_dir / f"{dataset}_community_sizes.png")
    _plot_centrality_correlations(correlation_all, plots_dir / f"{dataset}_centrality_spearman.png")
    _plot_topk_overlap(overlap_all, plots_dir / f"{dataset}_centrality_topk_overlap.png")
    _plot_robustness_curves(robustness_curves, plots_dir / f"{dataset}_robustness_curves.png")
    _plot_metric_delta_heatmap(comparison_df, plots_dir / f"{dataset}_metric_deltas.png")

    return {
        "dataset": dataset,
        "results": results,
        "summary_df": summary_df,
        "comparison_df": comparison_df,
        "centrality_all": centrality_all,
        "correlation_all": correlation_all,
        "overlap_all": overlap_all,
        "robustness_curves": robustness_curves,
        "robustness_summary": robustness_summary,
        "original_summary": original_result.summary,
    }


def run_analysis_pipeline(
    *,
    datasets: list[str] | None = None,
    requested_names: list[str] | None = None,
    methods: list[str] | None = None,
    alpha: float | None = None,
    original_path: Path | None = None,
    backbone_paths: list[Path] | None = None,
    dataset_label: str = "custom",
    top_k: int = 20,
    robustness_steps: int = 20,
    random_trials: int = 10,
    output_root: Path | None = None,
    output_tag: str | None = None,
) -> Path:
    default_datasets, default_alpha = _load_runtime_defaults()
    resolved_datasets = datasets if datasets is not None else default_datasets
    resolved_alpha = default_alpha if alpha is None else alpha
    resolved_methods = methods if methods is not None else DEFAULT_METHODS

    if output_root is None:
        run_stamp = datetime.now().strftime("%d_%m_%Y-%Hh_%M_%S")
        run_tag = _slugify(output_tag) if output_tag else run_stamp
        output_root = OUTPUT_ROOT / run_tag

    csv_dir = output_root / "csv"
    plots_dir = output_root / "plots"
    md_dir = output_root / "md"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    if original_path is not None:
        if not backbone_paths:
            raise ValueError("Informe pelo menos um arquivo em backbone_paths no modo customizado.")
        group_specs = _build_group_specs_from_paths(
            dataset_label=dataset_label,
            original_path=original_path,
            backbone_paths=backbone_paths,
        )
    else:
        group_specs = _build_group_specs_from_datasets(
            datasets=resolved_datasets,
            requested_names=requested_names,
            methods=resolved_methods,
            alpha=resolved_alpha,
        )

    dataset_sections: list[dict] = []
    summary_frames: list[pd.DataFrame] = []
    comparison_frames: list[pd.DataFrame] = []
    correlation_frames: list[pd.DataFrame] = []
    overlap_frames: list[pd.DataFrame] = []
    robustness_curve_frames: list[pd.DataFrame] = []
    robustness_summary_frames: list[pd.DataFrame] = []

    sns.set_theme(style="whitegrid")

    for spec in group_specs:
        section = _analyze_group(
            dataset=spec["dataset"],
            original_path=spec["original_path"],
            backbone_paths=spec["backbone_paths"],
            csv_dir=csv_dir,
            plots_dir=plots_dir,
            top_k=top_k,
            robustness_steps=robustness_steps,
            random_trials=random_trials,
        )
        dataset_sections.append(section)
        summary_frames.append(section["summary_df"])
        if not section["comparison_df"].empty:
            comparison_frames.append(section["comparison_df"])
        if not section["correlation_all"].empty:
            correlation_frames.append(section["correlation_all"])
        if not section["overlap_all"].empty:
            overlap_frames.append(section["overlap_all"])
        robustness_curve_frames.append(section["robustness_curves"])
        robustness_summary_frames.append(section["robustness_summary"])

    if summary_frames:
        _save_dataframe(pd.concat(summary_frames, ignore_index=True), csv_dir / "all_network_summary.csv")
    if comparison_frames:
        _save_dataframe(
            pd.concat(comparison_frames, ignore_index=True),
            csv_dir / "all_comparison_vs_original.csv",
        )
    if correlation_frames:
        _save_dataframe(
            pd.concat(correlation_frames, ignore_index=True),
            csv_dir / "all_centrality_correlation_summary.csv",
        )
    if overlap_frames:
        _save_dataframe(
            pd.concat(overlap_frames, ignore_index=True),
            csv_dir / "all_centrality_topk_overlap.csv",
        )
    if robustness_curve_frames:
        _save_dataframe(
            pd.concat(robustness_curve_frames, ignore_index=True),
            csv_dir / "all_robustness_curves.csv",
        )
    if robustness_summary_frames:
        _save_dataframe(
            pd.concat(robustness_summary_frames, ignore_index=True),
            csv_dir / "all_robustness_summary.csv",
        )

    _write_report(
        output_path=md_dir / "report.md",
        dataset_sections=dataset_sections,
        output_root=output_root,
    )

    print(f"\nAnalise concluida. Resultados salvos em: {output_root}")
    return output_root


def run_single_backbone_analysis(
    *,
    dataset: str,
    original_path: Path,
    backbone_path: Path,
    output_root: Path | None = None,
    top_k: int = 20,
    robustness_steps: int = 20,
    random_trials: int = 10,
) -> Path:
    resolved_output_root = output_root
    if resolved_output_root is None:
        resolved_output_root = OUTPUT_ROOT / _slugify(dataset) / _slugify(_label_from_path(backbone_path))

    return run_analysis_pipeline(
        original_path=original_path,
        backbone_paths=[backbone_path],
        dataset_label=dataset,
        top_k=top_k,
        robustness_steps=robustness_steps,
        random_trials=random_trials,
        output_root=resolved_output_root,
    )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_analysis_pipeline(
        datasets=args.datasets,
        requested_names=args.network_names,
        methods=args.methods,
        alpha=args.alpha,
        original_path=args.original_path,
        backbone_paths=args.backbone_paths,
        dataset_label=args.dataset_label,
        top_k=args.top_k,
        robustness_steps=args.robustness_steps,
        random_trials=args.random_trials,
        output_tag=args.output_tag,
    )


if __name__ == "__main__":
    main()
