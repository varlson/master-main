from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd


EPS = 1e-8
CENTRALITY_METRICS = [
    "degree_centrality",
    "betweenness_centrality",
    "closeness_centrality",
    "eigenvector_centrality",
    "pagerank",
]


@dataclass
class GraphAnalysisResult:
    name: str
    dataset: str
    role: str
    source_path: str
    graph: nx.Graph
    summary: dict
    node_metrics: pd.DataFrame
    edge_metrics: pd.DataFrame
    path_lengths: pd.DataFrame
    community_membership: pd.DataFrame


def _coerce_weight(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _load_adjacency_from_path(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".pkl":
        with path.open("rb") as file_obj:
            data = pickle.load(file_obj, encoding="latin1")
        if isinstance(data, (tuple, list)):
            _, _, adj = data
        else:
            adj = data
        return np.asarray(adj)

    if ext == ".npy":
        return np.asarray(np.load(path, allow_pickle=True))

    if ext == ".npz":
        npz = np.load(path, allow_pickle=True)
        if "adj" in npz:
            return np.asarray(npz["adj"])
        for value in npz.values():
            if isinstance(value, np.ndarray) and value.ndim == 2:
                return np.asarray(value)
        raise ValueError(f"Arquivo {path} nao contem uma matriz 2D de adjacencia.")

    raise ValueError(f"Formato nao suportado para matriz de adjacencia: {path.suffix}")


def _normalize_graph(graph: nx.Graph) -> nx.Graph:
    normalized = nx.Graph()
    normalized.add_nodes_from(str(node) for node in graph.nodes())

    for source, target, data in graph.edges(data=True):
        source_id = str(source)
        target_id = str(target)
        if source_id == target_id:
            continue

        weight = _coerce_weight(data.get("weight", 1.0))
        if normalized.has_edge(source_id, target_id):
            normalized[source_id][target_id]["weight"] += weight
        else:
            normalized.add_edge(source_id, target_id, weight=weight)

    for node in normalized.nodes():
        normalized.nodes[node]["label"] = node

    return normalized


def load_graph(path: str | Path) -> nx.Graph:
    graph_path = Path(path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Arquivo de grafo nao encontrado: {graph_path}")

    ext = graph_path.suffix.lower()
    if ext in {".pkl", ".npy", ".npz"}:
        adj = _load_adjacency_from_path(graph_path)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"Matriz de adjacencia invalida em {graph_path}: {adj.shape}")
        raw_graph = nx.from_numpy_array(adj)
        return _normalize_graph(raw_graph)

    if ext in {".graphml", ".xml"}:
        raw_graph = nx.read_graphml(graph_path)
        return _normalize_graph(raw_graph)

    raise ValueError(f"Formato de grafo nao suportado: {graph_path.suffix}")


def _graph_with_inverse_distance(graph: nx.Graph) -> nx.Graph:
    distance_graph = graph.copy()
    for source, target, data in distance_graph.edges(data=True):
        weight = max(_coerce_weight(data.get("weight", 1.0)), EPS)
        data["distance"] = 1.0 / weight
    return distance_graph


def _largest_component_graph(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        return graph.copy()
    if graph.number_of_edges() == 0:
        node = next(iter(graph.nodes()))
        return graph.subgraph([node]).copy()

    component_nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(component_nodes).copy()


def _edge_overlap_values(graph: nx.Graph) -> dict[tuple[str, str], float]:
    overlaps: dict[tuple[str, str], float] = {}
    for source, target in graph.edges():
        source_neighbors = set(graph.neighbors(source)) - {target}
        target_neighbors = set(graph.neighbors(target)) - {source}
        union = source_neighbors | target_neighbors
        key = tuple(sorted((str(source), str(target))))
        overlaps[key] = len(source_neighbors & target_neighbors) / len(union) if union else 0.0
    return overlaps


def _local_efficiency_by_node(graph: nx.Graph) -> dict[str, float]:
    values: dict[str, float] = {}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if len(neighbors) < 2:
            values[str(node)] = 0.0
            continue
        neighborhood = graph.subgraph(neighbors).copy()
        values[str(node)] = float(nx.global_efficiency(neighborhood))
    return values


def _safe_eigenvector_centrality(graph: nx.Graph) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_edges() == 0:
        return {str(node): 0.0 for node in graph.nodes()}

    try:
        return {
            str(node): float(value)
            for node, value in nx.eigenvector_centrality_numpy(graph, weight="weight").items()
        }
    except Exception:
        try:
            return {
                str(node): float(value)
                for node, value in nx.eigenvector_centrality(
                    graph, weight="weight", max_iter=2000, tol=1e-06
                ).items()
            }
        except Exception:
            return {str(node): 0.0 for node in graph.nodes()}


def _safe_pagerank(graph: nx.Graph) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}
    try:
        return {
            str(node): float(value)
            for node, value in nx.pagerank(graph, weight="weight").items()
        }
    except Exception:
        uniform = 1.0 / max(graph.number_of_nodes(), 1)
        return {str(node): uniform for node in graph.nodes()}


def _safe_connectivity(graph: nx.Graph, mode: str) -> float:
    if graph.number_of_nodes() <= 1:
        return 0.0
    try:
        if mode == "node":
            return float(nx.node_connectivity(graph))
        return float(nx.edge_connectivity(graph))
    except Exception:
        return 0.0


def _safe_assortativity(graph: nx.Graph) -> float:
    if graph.number_of_edges() == 0:
        return float("nan")
    try:
        return float(nx.degree_assortativity_coefficient(graph))
    except Exception:
        return float("nan")


def _community_membership(graph: nx.Graph, graph_name: str, dataset: str, role: str) -> tuple[pd.DataFrame, dict]:
    if graph.number_of_nodes() == 0:
        return (
            pd.DataFrame(
                columns=["graph_name", "dataset", "role", "node", "community_id", "community_size"]
            ),
            {
                "num_communities": 0,
                "modularity": float("nan"),
                "largest_community_size": 0,
                "largest_community_ratio": float("nan"),
            },
        )

    if graph.number_of_edges() == 0:
        communities = [{node} for node in graph.nodes()]
        modularity = float("nan")
    else:
        communities = list(nx.algorithms.community.greedy_modularity_communities(graph, weight="weight"))
        modularity = float(nx.algorithms.community.modularity(graph, communities, weight="weight"))

    rows: list[dict] = []
    largest_size = 0
    for community_id, community_nodes in enumerate(communities):
        community_size = len(community_nodes)
        largest_size = max(largest_size, community_size)
        for node in community_nodes:
            rows.append(
                {
                    "graph_name": graph_name,
                    "dataset": dataset,
                    "role": role,
                    "node": str(node),
                    "community_id": community_id,
                    "community_size": community_size,
                }
            )

    membership = pd.DataFrame(rows)
    summary = {
        "num_communities": len(communities),
        "modularity": modularity,
        "largest_community_size": largest_size,
        "largest_community_ratio": largest_size / max(graph.number_of_nodes(), 1),
    }
    return membership, summary


def _path_length_table(graph: nx.Graph, graph_name: str, dataset: str, role: str) -> pd.DataFrame:
    gcc = _largest_component_graph(graph)
    rows: list[dict] = []
    if gcc.number_of_nodes() <= 1:
        return pd.DataFrame(
            rows,
            columns=["graph_name", "dataset", "role", "source", "target", "path_length"],
        )

    for source, lengths in nx.all_pairs_shortest_path_length(gcc):
        for target, length in lengths.items():
            if str(source) >= str(target):
                continue
            rows.append(
                {
                    "graph_name": graph_name,
                    "dataset": dataset,
                    "role": role,
                    "source": str(source),
                    "target": str(target),
                    "path_length": int(length),
                }
            )

    return pd.DataFrame(rows)


def _path_summary(graph: nx.Graph, path_lengths: pd.DataFrame) -> dict:
    gcc = _largest_component_graph(graph)
    if gcc.number_of_nodes() <= 1:
        return {
            "avg_path_length_gcc": float("nan"),
            "median_path_length_gcc": float("nan"),
            "p90_path_length_gcc": float("nan"),
            "diameter_gcc": float("nan"),
            "avg_weighted_distance_gcc": float("nan"),
            "global_efficiency": float("nan"),
        }

    values = path_lengths["path_length"].to_numpy(dtype=float) if not path_lengths.empty else np.array([])
    distance_graph = _graph_with_inverse_distance(gcc)

    try:
        weighted_distance = float(nx.average_shortest_path_length(distance_graph, weight="distance"))
    except Exception:
        weighted_distance = float("nan")

    return {
        "avg_path_length_gcc": float(values.mean()) if len(values) else float("nan"),
        "median_path_length_gcc": float(np.median(values)) if len(values) else float("nan"),
        "p90_path_length_gcc": float(np.percentile(values, 90)) if len(values) else float("nan"),
        "diameter_gcc": float(nx.diameter(gcc)),
        "avg_weighted_distance_gcc": weighted_distance,
        "global_efficiency": float(nx.global_efficiency(graph)),
    }


def analyze_graph(
    *,
    graph: nx.Graph,
    name: str,
    dataset: str,
    role: str,
    source_path: str | Path,
) -> GraphAnalysisResult:
    normalized = _normalize_graph(graph)
    node_ids = [str(node) for node in normalized.nodes()]
    degree = {str(node): int(value) for node, value in normalized.degree()}
    strength = {str(node): float(value) for node, value in normalized.degree(weight="weight")}
    clustering = {str(node): float(value) for node, value in nx.clustering(normalized).items()}
    weighted_clustering = {
        str(node): float(value)
        for node, value in nx.clustering(normalized, weight="weight").items()
    }
    local_efficiency = _local_efficiency_by_node(normalized)
    degree_centrality = {
        str(node): float(value) for node, value in nx.degree_centrality(normalized).items()
    }
    distance_graph = _graph_with_inverse_distance(normalized)
    betweenness = {
        str(node): float(value)
        for node, value in nx.betweenness_centrality(distance_graph, weight="distance").items()
    }
    closeness = {
        str(node): float(value)
        for node, value in nx.closeness_centrality(distance_graph, distance="distance").items()
    }
    eigenvector = _safe_eigenvector_centrality(normalized)
    pagerank = _safe_pagerank(normalized)
    community_membership, community_summary = _community_membership(
        normalized, name, dataset, role
    )
    community_lookup = {
        str(row["node"]): int(row["community_id"])
        for _, row in community_membership.iterrows()
    }

    node_rows = []
    for node_id in node_ids:
        node_rows.append(
            {
                "graph_name": name,
                "dataset": dataset,
                "role": role,
                "node": node_id,
                "degree": degree[node_id],
                "strength": strength[node_id],
                "clustering": clustering[node_id],
                "weighted_clustering": weighted_clustering[node_id],
                "local_efficiency": local_efficiency[node_id],
                "degree_centrality": degree_centrality[node_id],
                "betweenness_centrality": betweenness[node_id],
                "closeness_centrality": closeness[node_id],
                "eigenvector_centrality": eigenvector[node_id],
                "pagerank": pagerank[node_id],
                "community_id": community_lookup.get(node_id, -1),
            }
        )
    node_metrics = pd.DataFrame(node_rows)

    edge_overlap = _edge_overlap_values(normalized)
    edge_rows = []
    for source, target, data in normalized.edges(data=True):
        key = tuple(sorted((str(source), str(target))))
        edge_rows.append(
            {
                "graph_name": name,
                "dataset": dataset,
                "role": role,
                "source": str(source),
                "target": str(target),
                "weight": _coerce_weight(data.get("weight", 1.0)),
                "edge_overlap": edge_overlap.get(key, 0.0),
            }
        )
    edge_metrics = pd.DataFrame(edge_rows)

    path_lengths = _path_length_table(normalized, name, dataset, role)
    path_summary = _path_summary(normalized, path_lengths)
    gcc = _largest_component_graph(normalized)
    weight_values = edge_metrics["weight"].to_numpy(dtype=float) if not edge_metrics.empty else np.array([])

    summary = {
        "graph_name": name,
        "dataset": dataset,
        "role": role,
        "source_path": str(source_path),
        "num_nodes": normalized.number_of_nodes(),
        "num_edges": normalized.number_of_edges(),
        "density": float(nx.density(normalized)) if normalized.number_of_nodes() > 1 else 0.0,
        "average_degree": float(np.mean(list(degree.values()))) if degree else 0.0,
        "average_strength": float(np.mean(list(strength.values()))) if strength else 0.0,
        "edge_weight_mean": float(weight_values.mean()) if len(weight_values) else float("nan"),
        "edge_weight_median": float(np.median(weight_values)) if len(weight_values) else float("nan"),
        "edge_weight_std": float(weight_values.std()) if len(weight_values) else float("nan"),
        "edge_weight_min": float(weight_values.min()) if len(weight_values) else float("nan"),
        "edge_weight_max": float(weight_values.max()) if len(weight_values) else float("nan"),
        "average_clustering": float(np.mean(list(clustering.values()))) if clustering else 0.0,
        "average_weighted_clustering": (
            float(np.mean(list(weighted_clustering.values()))) if weighted_clustering else 0.0
        ),
        "transitivity": float(nx.transitivity(normalized)) if normalized.number_of_edges() else 0.0,
        "local_efficiency_mean": float(np.mean(list(local_efficiency.values()))) if local_efficiency else 0.0,
        "edge_overlap_mean": (
            float(edge_metrics["edge_overlap"].mean()) if not edge_metrics.empty else 0.0
        ),
        "is_connected": bool(nx.is_connected(normalized)) if normalized.number_of_nodes() else False,
        "num_connected_components": (
            int(nx.number_connected_components(normalized)) if normalized.number_of_nodes() else 0
        ),
        "giant_component_size": gcc.number_of_nodes(),
        "giant_component_ratio": gcc.number_of_nodes() / max(normalized.number_of_nodes(), 1),
        "node_connectivity_gcc": _safe_connectivity(gcc, "node"),
        "edge_connectivity_gcc": _safe_connectivity(gcc, "edge"),
        "degree_assortativity": _safe_assortativity(normalized),
        **community_summary,
        **path_summary,
    }

    return GraphAnalysisResult(
        name=name,
        dataset=dataset,
        role=role,
        source_path=str(source_path),
        graph=normalized,
        summary=summary,
        node_metrics=node_metrics,
        edge_metrics=edge_metrics,
        path_lengths=path_lengths,
        community_membership=community_membership,
    )


def compare_centralities(
    *,
    original: GraphAnalysisResult,
    backbone: GraphAnalysisResult,
    top_k: int = 20,
    metrics: Iterable[str] = CENTRALITY_METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original_df = original.node_metrics.copy()
    backbone_df = backbone.node_metrics.copy()

    merged = original_df.merge(
        backbone_df,
        on="node",
        how="outer",
        suffixes=("_original", "_backbone"),
        indicator=True,
    )
    merged["dataset"] = original.dataset
    merged["original_graph"] = original.name
    merged["backbone_graph"] = backbone.name
    merged["present_in_original"] = merged["_merge"].isin(["left_only", "both"])
    merged["present_in_backbone"] = merged["_merge"].isin(["right_only", "both"])

    correlation_rows: list[dict] = []
    overlap_rows: list[dict] = []
    common = merged[merged["_merge"] == "both"].copy()

    for metric_name in metrics:
        original_col = f"{metric_name}_original"
        backbone_col = f"{metric_name}_backbone"
        valid = common[[original_col, backbone_col]].dropna()
        if valid.empty:
            spearman = float("nan")
            pearson = float("nan")
        else:
            spearman = float(valid[original_col].corr(valid[backbone_col], method="spearman"))
            pearson = float(valid[original_col].corr(valid[backbone_col], method="pearson"))

        correlation_rows.append(
            {
                "dataset": original.dataset,
                "original_graph": original.name,
                "backbone_graph": backbone.name,
                "metric": metric_name,
                "num_common_nodes": len(valid),
                "spearman_corr": spearman,
                "pearson_corr": pearson,
            }
        )

        original_top = set(
            original_df.nlargest(min(top_k, len(original_df)), metric_name)["node"].astype(str)
        )
        backbone_top = set(
            backbone_df.nlargest(min(top_k, len(backbone_df)), metric_name)["node"].astype(str)
        )
        overlap = len(original_top & backbone_top)
        overlap_rows.append(
            {
                "dataset": original.dataset,
                "original_graph": original.name,
                "backbone_graph": backbone.name,
                "metric": metric_name,
                "k": min(top_k, len(original_top), len(backbone_top)),
                "overlap_size": overlap,
                "overlap_ratio": overlap / max(min(top_k, len(original_top), len(backbone_top)), 1),
            }
        )

        merged[f"{metric_name}_delta"] = merged[backbone_col] - merged[original_col]
        merged[f"{metric_name}_abs_delta"] = np.abs(merged[f"{metric_name}_delta"])

    merged = merged.drop(columns=["_merge"])
    correlation_df = pd.DataFrame(correlation_rows)
    overlap_df = pd.DataFrame(overlap_rows)
    return merged, correlation_df, overlap_df


def robustness_curve(
    *,
    graph: nx.Graph,
    graph_name: str,
    dataset: str,
    steps: int = 20,
    random_trials: int = 10,
    random_seed: int = 42,
) -> pd.DataFrame:
    base_graph = _normalize_graph(graph)
    initial_nodes = base_graph.number_of_nodes()
    if initial_nodes == 0:
        return pd.DataFrame(
            columns=[
                "dataset",
                "graph_name",
                "strategy",
                "trial",
                "fraction_removed",
                "nodes_remaining",
                "largest_component_ratio",
                "global_efficiency_ratio",
            ]
        )

    fractions = np.linspace(0.0, 1.0, num=max(steps, 1) + 1)
    baseline_efficiency = float(nx.global_efficiency(base_graph)) if initial_nodes > 1 else 0.0
    rows: list[dict] = []

    def snapshot(current_graph: nx.Graph, strategy: str, trial: int, fraction_removed: float) -> None:
        if current_graph.number_of_nodes() == 0:
            largest_ratio = 0.0
            efficiency_ratio = 0.0
            remaining = 0
        else:
            gcc = _largest_component_graph(current_graph)
            largest_ratio = gcc.number_of_nodes() / initial_nodes
            efficiency = float(nx.global_efficiency(current_graph)) if current_graph.number_of_nodes() > 1 else 0.0
            efficiency_ratio = efficiency / baseline_efficiency if baseline_efficiency > EPS else 0.0
            remaining = current_graph.number_of_nodes()

        rows.append(
            {
                "dataset": dataset,
                "graph_name": graph_name,
                "strategy": strategy,
                "trial": trial,
                "fraction_removed": float(fraction_removed),
                "nodes_remaining": remaining,
                "largest_component_ratio": largest_ratio,
                "global_efficiency_ratio": efficiency_ratio,
            }
        )

    targeted = base_graph.copy()
    removed_count = 0
    snapshot(targeted, "targeted_degree", 0, 0.0)
    for fraction in fractions[1:]:
        target_removed = int(round(fraction * initial_nodes))
        while removed_count < target_removed and targeted.number_of_nodes() > 0:
            node = max(targeted.degree(), key=lambda item: (item[1], str(item[0])))[0]
            targeted.remove_node(node)
            removed_count += 1
        snapshot(targeted, "targeted_degree", 0, fraction)

    rng = np.random.default_rng(random_seed)
    original_nodes = np.array(list(base_graph.nodes()), dtype=object)
    for trial in range(max(random_trials, 1)):
        random_graph = base_graph.copy()
        node_order = list(rng.permutation(original_nodes))
        removed_count = 0
        snapshot(random_graph, "random", trial, 0.0)
        for fraction in fractions[1:]:
            target_removed = int(round(fraction * initial_nodes))
            while removed_count < target_removed and random_graph.number_of_nodes() > 0:
                random_graph.remove_node(str(node_order[removed_count]))
                removed_count += 1
            snapshot(random_graph, "random", trial, fraction)

    return pd.DataFrame(rows)


def summarize_robustness(curves: pd.DataFrame) -> pd.DataFrame:
    if curves.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "graph_name",
                "strategy",
                "auc_lcc_mean",
                "auc_lcc_std",
                "auc_efficiency_mean",
                "auc_efficiency_std",
            ]
        )

    per_trial_rows: list[dict] = []
    for (dataset, graph_name, strategy, trial), frame in curves.groupby(
        ["dataset", "graph_name", "strategy", "trial"], dropna=False
    ):
        ordered = frame.sort_values("fraction_removed")
        per_trial_rows.append(
            {
                "dataset": dataset,
                "graph_name": graph_name,
                "strategy": strategy,
                "trial": trial,
                "auc_lcc": float(
                    np.trapz(
                        ordered["largest_component_ratio"].to_numpy(),
                        ordered["fraction_removed"].to_numpy(),
                    )
                ),
                "auc_efficiency": float(
                    np.trapz(
                        ordered["global_efficiency_ratio"].to_numpy(),
                        ordered["fraction_removed"].to_numpy(),
                    )
                ),
            }
        )

    trial_df = pd.DataFrame(per_trial_rows)
    summary = (
        trial_df.groupby(["dataset", "graph_name", "strategy"], as_index=False)
        .agg(
            auc_lcc_mean=("auc_lcc", "mean"),
            auc_lcc_std=("auc_lcc", "std"),
            auc_efficiency_mean=("auc_efficiency", "mean"),
            auc_efficiency_std=("auc_efficiency", "std"),
        )
        .sort_values(["dataset", "graph_name", "strategy"])
        .reset_index(drop=True)
    )
    return summary
